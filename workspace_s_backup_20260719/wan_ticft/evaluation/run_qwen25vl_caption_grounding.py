"""Audit caption-grounded traffic actors with Qwen2.5-VL on BDD histories."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb


PROMPT = """You are analyzing four chronological observations from one traffic video.
The supplied caption describes the event and may contain incomplete, future, or noisy information.

Tasks:
1. Infer ego-camera motion from the visual sequence.
2. Identify the primary pedestrian or vehicle referred to by the caption, but do not invent it if it is not visible in the LAST frame.
3. List at most 5 relevant visible traffic actors in the LAST frame. Prioritize the caption actor and nearby moving actors. Distinguish truly moving actors from parked vehicles and static objects using all four observations.
4. Give an observed motion label based only on pixels, and a separate future motion prior that may also use the caption.

Return exactly one JSON object and no markdown. Bounding boxes use pixel coordinates [x1,y1,x2,y2] in the supplied LAST frame; its dimensions are stated after the caption.
Schema:
{
  "ego_motion": "stationary|forward|turn_left|turn_right|uncertain",
  "caption_actor_visible": true,
  "caption_actor_id": "actor_1|null",
  "actors": [
    {
      "id": "actor_1",
      "class": "pedestrian|car|truck|bus|motorcycle|bicycle|other",
      "bbox_last": [0,0,1000,1000],
      "caption_match": true,
      "observed_motion": "stationary|moving_left|moving_right|approaching|receding|crossing|uncertain",
      "future_motion_prior": "stationary|moving_left|moving_right|approaching|receding|crossing|uncertain",
      "confidence": 0.0
    }
  ],
  "scene_note": "at most 8 words"
}

Caption:
"""


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("response contains no JSON object")
        cleaned = cleaned[start:]
    try:
        result, _ = json.JSONDecoder().raw_decode(cleaned)
        salvaged = False
    except json.JSONDecodeError:
        # Generation may finish the actor array and then hit max_new_tokens in
        # the optional scene note. Recover only a structurally complete array.
        key = cleaned.find('"actors"')
        start = cleaned.find("[", key)
        if key < 0 or start < 0:
            raise
        depth, in_string, escaped, end = 0, False, False, -1
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end < 0:
            raise
        result = json.loads(cleaned[: end + 1] + "\n}")
        salvaged = True
    if not isinstance(result, dict) or not isinstance(result.get("actors"), list):
        raise ValueError("response does not match actor schema")
    result["_salvaged"] = salvaged
    return result


def normalize_box(raw, width: int, height: int, coordinate_size=None):
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    maximum = max(abs(value) for value in values)
    if maximum <= 1.5:
        x1, y1, x2, y2 = values[0] * width, values[1] * height, values[2] * width, values[3] * height
    elif (
        coordinate_size
        and 0 <= values[0] <= coordinate_size[0] * 1.05
        and 0 <= values[2] <= coordinate_size[0] * 1.05
        and 0 <= values[1] <= coordinate_size[1] * 1.05
        and 0 <= values[3] <= coordinate_size[1] * 1.05
    ):
        x1, y1, x2, y2 = (
            values[0] * width / coordinate_size[0],
            values[1] * height / coordinate_size[1],
            values[2] * width / coordinate_size[0],
            values[3] * height / coordinate_size[1],
        )
    elif maximum <= 1000.5:
        x1, y1, x2, y2 = values[0] * width / 1000, values[1] * height / 1000, values[2] * width / 1000, values[3] * height / 1000
    else:
        x1, y1, x2, y2 = values
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return x1, y1, x2, y2


def actor_color(actor: dict):
    motion = str(actor.get("observed_motion", "uncertain"))
    if motion == "stationary":
        return (70, 220, 255)
    if motion == "uncertain":
        return (255, 210, 60)
    return (255, 70, 70)


def draw_overlay(cv2, image, result: dict, coordinate_size=None):
    output = image.copy()
    height, width = output.shape[:2]
    for actor in result.get("actors", [])[:8]:
        box = normalize_box(actor.get("bbox_last"), width, height, coordinate_size)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        color = actor_color(actor)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        caption_mark = "*" if actor.get("caption_match") else ""
        try:
            confidence = float(actor.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        label = f"{actor.get('id','?')}{caption_mark} {actor.get('class','?')} {actor.get('observed_motion','?')} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(0, y1 - text_height - 10)
        cv2.rectangle(output, (x1, top), (min(width - 1, x1 + text_width + 8), y1), color, -1)
        cv2.putText(output, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    header = f"ego={result.get('ego_motion','?')} caption_actor={result.get('caption_actor_id','?')}"
    cv2.rectangle(output, (0, 0), (width, 32), (0, 0, 0), -1)
    cv2.putText(output, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def make_sheet(cv2, frames, overlay, sample_id: str):
    tiles = []
    for index, frame in enumerate([*frames, overlay]):
        tile = cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA)
        label = f"history {index}" if index < len(frames) else "Qwen grounding on last frame"
        cv2.rectangle(tile, (0, 0), (384, 28), (0, 0, 0), -1)
        cv2.putText(tile, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    del sample_id
    return cv2.hconcat(tiles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--sample_id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=700)
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--reuse_responses", action="store_true")
    args = parser.parse_args()

    from PIL import Image

    cv2, _ = import_cv2_np()
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"]
    if args.sample_id:
        wanted = set(args.sample_id)
        rows = [row for row in rows if row.get("sample_id") in wanted]
        order = {sample_id: index for index, sample_id in enumerate(args.sample_id)}
        rows.sort(key=lambda row: order[row["sample_id"]])
    rows = rows[: args.limit]
    if len(rows) != args.limit:
        raise RuntimeError(f"expected {args.limit} BDD rows, found {len(rows)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_responses:
        torch = processor = model = device = None
    else:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            args.model,
            min_pixels=256 * 28 * 28,
            max_pixels=args.max_pixels,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()
        device = next(model.parameters()).device
    report = []
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        condition_paths = (row.get("frames") or {}).get("condition") or []
        frames = [frame for path in condition_paths if (frame := read_rgb(cv2, Path(path))) is not None]
        if len(frames) != 4:
            raise RuntimeError(f"{sample_id}: expected 4 condition frames, found {len(frames)}")
        sample_dir = args.out_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        pil_images = []
        for index, frame in enumerate(frames):
            height, width = frame.shape[:2]
            scale = min(1.0, 768.0 / max(height, width))
            if scale < 1:
                frame = cv2.resize(frame, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
            path = sample_dir / f"history_{index}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            pil_images.append(Image.open(path).convert("RGB"))

        response_path = sample_dir / "response.txt"
        if args.reuse_responses:
            if not response_path.is_file():
                raise FileNotFoundError(response_path)
            response = response_path.read_text(encoding="utf-8")
        else:
            content = []
            for index in range(4):
                content.append({"type": "text", "text": f"FRAME_{index} (chronological):"})
                content.append({"type": "image"})
            content.append(
                {
                    "type": "text",
                    "text": PROMPT + str(row.get("prompt", "")) + f"\nLAST_FRAME_SIZE={pil_images[-1].size[0]}x{pil_images[-1].size[1]}",
                }
            )
            messages = [{"role": "user", "content": content}]
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat_text], images=pil_images, padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
            response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            response_path.write_text(response, encoding="utf-8")
        try:
            result = extract_json(response)
            parse_status = "salvaged" if result.pop("_salvaged", False) else "ok"
            (sample_dir / "response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            overlay = draw_overlay(cv2, frames[-1], result, pil_images[-1].size)
            actor_count = len(result.get("actors", []))
        except (ValueError, json.JSONDecodeError) as exc:
            result = {}
            parse_status = f"bad: {exc}"
            overlay = frames[-1]
            actor_count = 0
        overlay_path = sample_dir / "grounding_overlay.jpg"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        sheet = make_sheet(cv2, frames, overlay, sample_id)
        sheet_path = sample_dir / "grounding_sheet.jpg"
        cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])
        report.append(
            {
                "sample_id": sample_id,
                "status": parse_status,
                "actors": actor_count,
                "ego_motion": result.get("ego_motion", ""),
                "caption_actor_visible": result.get("caption_actor_visible", ""),
                "caption_actor_id": result.get("caption_actor_id", ""),
                "overlay": str(overlay_path),
                "sheet": str(sheet_path),
            }
        )
        print(f"[{position}/{len(rows)}] {sample_id} | {parse_status} actors={actor_count}", flush=True)
        if not args.reuse_responses:
            del inputs, generated, trimmed
            torch.cuda.empty_cache()

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0])
        writer.writeheader()
        writer.writerows(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
