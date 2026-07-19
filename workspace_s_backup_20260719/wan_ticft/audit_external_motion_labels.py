from pathlib import Path
import json
import re
from collections import Counter

cap_root = Path("/data/Track_5/WTS_dataset/WTS_dataset/external/caption")
vid_root = Path("/data/Track_5/WTS_dataset/WTS_dataset/external/video")

speed_re = re.compile(r"(\d+(?:\.\d+)?)\s*km/?h", re.I)
ped_terms = ("cross", "walking", "moving", "stand", "standing", "still", "stop", "stopped", "deceler")
veh_terms = ("constant speed", "standstill", "stopped", "moving", "accelerat", "deceler", "turn")

rows = []
for cap_path in sorted(cap_root.rglob("*_caption.json")):
    split = cap_path.parent.name
    try:
        obj = json.load(open(cap_path, encoding="utf-8"))
    except Exception:
        continue
    video_name = obj.get("video_name") or cap_path.name.replace("_caption.json", ".mp4")
    video_path = vid_root / split / video_name
    phases = obj.get("event_phase") or []
    text = " ".join(
        str(phase.get("caption_pedestrian", "")) + " " + str(phase.get("caption_vehicle", ""))
        for phase in phases
    ).lower()
    speeds = [float(x) for x in speed_re.findall(text)]
    rows.append((split, video_path.exists(), video_name, speeds, text))

print("rows", len(rows), "videos_exist", sum(1 for r in rows if r[1]))
print("split", Counter(r[0] for r in rows))
print("with_speed", sum(1 for r in rows if r[3]))
print("zero_speed", sum(1 for r in rows if r[3] and max(r[3]) == 0))
print("moving_speed", sum(1 for r in rows if r[3] and max(r[3]) > 0))
for term in ped_terms:
    print("ped_term", term, sum(1 for r in rows if term in r[4]))
for term in veh_terms:
    print("veh_term", term, sum(1 for r in rows if term in r[4]))
print("examples")
for row in rows[:8]:
    print(row[0], row[2], "speeds", row[3][:4], "exists", row[1])

