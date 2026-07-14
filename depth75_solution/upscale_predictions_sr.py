import os
import sys
import glob
import argparse
import time
import csv
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

def upscale_with_swin2sr_batch(pil_images, target_h, target_w, processor, model, device, batch_size=8):
    upscaled_frames = []
    H, W = 480, 832  # Cosmo model shape
    
    for idx in range(0, len(pil_images), batch_size):
        batch_imgs = pil_images[idx : idx + batch_size]
        
        # Process batch images using processor
        inputs = processor(batch_imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Model inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        reconstruction = outputs.reconstruction  # [B, 3, H_pad_x2, W_pad_x2]
        
        # Crop out padding (padding was only added to the right and bottom)
        h_up = 2 * H
        w_up = 2 * W
        reconstruction = reconstruction[:, :, :h_up, :w_up]
        
        # Downscale to target resolution using bicubic downsampling
        final_frames = F.interpolate(reconstruction, size=(target_h, target_w), mode="bicubic", align_corners=False)
        final_frames = final_frames.clamp(0.0, 1.0)
        
        # Convert back to PIL
        for f_idx in range(final_frames.shape[0]):
            frame_pil = TF.to_pil_image(final_frames[f_idx].cpu())
            upscaled_frames.append(frame_pil)
            
    return upscaled_frames

def main():
    parser = argparse.ArgumentParser(description="Batch Swin2SR Upscaler for Cosmo predictions")
    parser.add_argument("--input_dir", type=str, default="/workspace/Cosmo/prediction_epoch3")
    parser.add_argument("--output_dir", type=str, default="/workspace/Cosmo/prediction_epoch3_sr")
    parser.add_argument("--test_csv", type=str, default="/workspace/depth75_solution/test_index_hybrid.csv")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--sr_model", type=str, default="caidas/swin2SR-lightweight-x2-64")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true", help="Process 1 sample and exit")
    args = parser.parse_args()

    # Load test csv
    with open(args.test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Total test samples to process: {len(rows)}")

    print(f"Loading Swin2SR model '{args.sr_model}'...")
    sr_processor = AutoImageProcessor.from_pretrained(args.sr_model)
    sr_model = Swin2SRForImageSuperResolution.from_pretrained(args.sr_model).to(args.device)
    sr_model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    videos_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    report_rows = []
    start_time = time.time()

    for idx, row in enumerate(rows):
        sample_name = row["sample_id"]
        n_future = int(row["target_N"])
        orig_w = int(row["frame_width"])
        orig_h = int(row["frame_height"])

        video_name = f"{sample_name}.avi"
        video_path = os.path.join(videos_dir, video_name)
        sample_dir = os.path.join(args.output_dir, sample_name)
        
        # Check if SFT prediction frames exist for this sample
        sft_sample_dir = os.path.join(args.input_dir, sample_name)
        frame_files = sorted(glob.glob(os.path.join(sft_sample_dir, "*.png")), 
                             key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        if len(frame_files) == 0:
            print(f"[{idx+1}/{len(rows)}] Skipping {sample_name}: No raw frames found in {sft_sample_dir}")
            continue

        if os.path.exists(video_path) and os.path.exists(sample_dir) and len(glob.glob(os.path.join(sample_dir, "*.png"))) == len(frame_files):
            print(f"[{idx+1}/{len(rows)}] Skipping {sample_name} (already exists)")
            report_rows.append({
                "sample_id": sample_name,
                "video": video_path,
                "status": "ok",
                "future_frames": n_future
            })
            continue

        print(f"\n--- [{idx+1}/{len(rows)}] Upscaling: {sample_name} ---")
        print(f"Target future frames: {n_future} ({orig_h}x{orig_w})")
        sys.stdout.flush()

        input_images = []
        for fpath in frame_files:
            img = Image.open(fpath).convert("RGB")
            input_images.append(img)

        # Swin2SR upscaling
        t0 = time.time()
        try:
            upscaled_predictions = upscale_with_swin2sr_batch(
                input_images, orig_h, orig_w, sr_processor, sr_model, args.device, args.batch_size
            )
            print(f"Batch Swin2SR upscaled {len(upscaled_predictions)} frames in {time.time()-t0:.2f}s")
            
            # Initialize OpenCV VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = 10.0
            writer = cv2.VideoWriter(video_path, fourcc, fps, (orig_w, orig_h))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open VideoWriter for {video_path}")
            
            os.makedirs(sample_dir, exist_ok=True)
            for f_idx, frame_pil in enumerate(upscaled_predictions):
                frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                writer.write(frame_cv2)
                
                png_path = os.path.join(sample_dir, f"{f_idx}.png")
                cv2.imwrite(png_path, frame_cv2)
            
            writer.release()
            print(f"Saved upscaled video to {video_path} and extracted PNGs to {sample_dir}")
            sys.stdout.flush()

            report_rows.append({
                "sample_id": sample_name,
                "video": video_path,
                "status": "ok",
                "future_frames": len(upscaled_predictions)
            })

        except Exception as e:
            print(f"Upscaling failed for {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            report_rows.append({
                "sample_id": sample_name,
                "video": "",
                "status": "failed",
                "future_frames": 0
            })

        # Free GPU memory cache
        torch.cuda.empty_cache()

        if args.dry_run:
            print("Dry-run sample completed successfully!")
            break

    # Write report.csv
    report_file = os.path.join(args.output_dir, "report.csv")
    with open(report_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "video", "status", "future_frames"])
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Report written to {report_file}")

    elapsed_time = time.time() - start_time
    print(f"\nBatch upscaling completed in {elapsed_time:.2f} seconds!")

if __name__ == "__main__":
    main()
