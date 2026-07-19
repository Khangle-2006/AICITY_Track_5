"""Use Qwen2.5-VL to write per-sample LTX prompts from traffic history frames."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb


PLANNER_PROMPT = """You are writing a SHORT prompt for LTX-Video 13B.
Input images are chronological history frames from the same traffic video.

Goal: make the video generator continue the exact observed scene, not redraw it.
The prompt must help preserve geometry and timing, not maximize cinematic beauty.

Return exactly one JSON object, no markdown:
{
  "camera_motion": "short phrase",
  "moving_objects": ["short phrase", "..."],
  "static_locks": ["short phrase", "..."],
  "risk": "main thing the generator must not change",
  "ltx_prompt": "80 to 110 words, concise, concrete, no bullet list"
}

Rules for ltx_prompt:
- Put geometry locks first.
- Mention only visible or caption-supported objects.
- Do not invent future events.
- Keep it under 110 words.
- Do not use vague phrases like beautiful, cinematic, high quality, masterpiece.
- Include: exact continuation, same camera, no new objects, no removed objects.

Challenge caption/context:
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
    result, _ = json.JSONDecoder().raw_decode(cleaned)
    if not isinstance(result, dict):
        raise ValueError("response is not a JSON object")
    prompt = str(result.get("ltx_prompt", "")).strip()
    if not prompt:
        raise ValueError("response missing ltx_prompt")
    result["ltx_prompt"] = compress_prompt(prompt, 110)
    return result


def compress_prompt(prompt: str, max_words: int) -> str:
    words = prompt.replace("\n", " ").split()
    return " ".join(words[:max_words])


def fallback_prompt(row: dict) -> str:
    phase = str(row.get("phase", ""))
    if phase in {"0", "1"}:
        motion = "Use very small forward motion."
    elif phase == "2":
        motion = "Continue the observed motion smoothly."
    else:
        motion = "Continue later-phase motion smoothly without changing layout."
    return " ".join(
        [
            "Exact traffic-video continuation from the final observed frame.",
            "Keep the same camera, focal length, horizon, road geometry, lanes, curbs, sidewalks, buildings, trees, parked cars, traffic lights, shadows, colors, and object identities.",
            "No new objects, no removed objects, no restyling, no zoom, no camera rotation.",
            motion,
        ]
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "sample_id",
        "status",
        "prompt_words",
        "camera_motion",
        "moving_objects",
        "static_locks",
        "risk",
        "ltx_prompt",
        "response_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def select_rows(rows: list[dict], sample_ids: list[str], limit: int) -> list[dict]:
    rows = [row for row in rows if row.get("view_type") == "bdd"]
    if sample_ids:
        wanted = set(sample_ids)
        order = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        rows = [row for row in rows if row.get("sample_id") in wanted]
        rows.sort(key=lambda row: order[row["sample_id"]])
    rows = rows[:limit]
    if len(rows) != limit:
        raise RuntimeError(f"expected {limit} BDD rows, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--sample_id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=420)
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--reuse_responses", action="store_true")
    args = parser.parse_args()

    from PIL import Image

    cv2, _ = import_cv2_np()
    rows = select_rows(read_jsonl(args.manifest), args.sample_id, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_responses:
        torch = processor = model = device = None
    else:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            args.model,
            min_pixels=128 * 28 * 28,
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

    planned_rows = []
    report = []
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        sample_dir = args.out_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        condition_paths = (row.get("frames") or {}).get("condition") or []
        frames = [frame for path in condition_paths if (frame := read_rgb(cv2, Path(path))) is not None]
        if len(frames) != len(condition_paths) or not frames:
            raise RuntimeError(f"{sample_id}: missing condition frames")
        pil_images = []
        for index, frame in enumerate(frames):
            height, width = frame.shape[:2]
            scale = min(1.0, 768.0 / max(height, width))
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (int(round(width * scale)), int(round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            image_path = sample_dir / f"history_{index}.jpg"
            cv2.imwrite(str(image_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            pil_images.append(Image.open(image_path).convert("RGB"))

        response_path = sample_dir / "planner_response.txt"
        status = "ok"
        if args.reuse_responses:
            response = response_path.read_text(encoding="utf-8")
        else:
            content = []
            for index in range(len(pil_images)):
                content.append({"type": "text", "text": f"FRAME_{index} chronological:"})
                content.append({"type": "image"})
            content.append({"type": "text", "text": PLANNER_PROMPT + str(row.get("prompt", ""))})
            messages = [{"role": "user", "content": content}]
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat_text], images=pil_images, padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
            response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            response_path.write_text(response, encoding="utf-8")
            del inputs, generated, trimmed
            torch.cuda.empty_cache()

        try:
            plan = extract_json(response)
        except Exception as exc:
            status = f"fallback: {exc}"
            plan = {
                "camera_motion": "",
                "moving_objects": [],
                "static_locks": [],
                "risk": "",
                "ltx_prompt": fallback_prompt(row),
            }
        plan_path = sample_dir / "planner.json"
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        planned = dict(row)
        planned["prompt"] = plan["ltx_prompt"]
        planned["qwen_prompt_plan"] = plan
        planned_rows.append(planned)
        report.append(
            {
                "sample_id": sample_id,
                "status": status,
                "prompt_words": len(str(plan["ltx_prompt"]).split()),
                "camera_motion": plan.get("camera_motion", ""),
                "moving_objects": json.dumps(plan.get("moving_objects", []), ensure_ascii=False),
                "static_locks": json.dumps(plan.get("static_locks", []), ensure_ascii=False),
                "risk": plan.get("risk", ""),
                "ltx_prompt": plan["ltx_prompt"],
                "response_path": str(response_path),
            }
        )
        print(f"[{position}/{len(rows)}] {sample_id} | {status} words={len(str(plan['ltx_prompt']).split())}", flush=True)

    write_jsonl(args.out_dir / "planned_manifest.jsonl", planned_rows)
    write_report(args.out_dir / "planner_report.csv", report)
    print(f"manifest={args.out_dir / 'planned_manifest.jsonl'}")
    print(f"report={args.out_dir / 'planner_report.csv'}")


if __name__ == "__main__":
    main()
