from pathlib import Path
import json

root = Path("/data/Track_5/WTS_dataset/WTS_dataset")

def count_ext(path, suffixes):
    counts = {s: 0 for s in suffixes}
    examples = {s: [] for s in suffixes}
    if not path.exists():
        return counts, examples
    for p in path.rglob("*"):
        if p.is_file():
            s = p.suffix.lower()
            if s in counts:
                counts[s] += 1
                if len(examples[s]) < 5:
                    examples[s].append(str(p))
    return counts, examples

for sub in ["video/train", "video/val", "external", "caption"]:
    path = root / sub
    print(f"\n== {sub} ==")
    print("exists", path.exists())
    if path.exists():
        dirs = [p for p in path.iterdir() if p.is_dir()]
        files = [p for p in path.iterdir() if p.is_file()]
        print("top_dirs", len(dirs), [p.name for p in dirs[:10]])
        print("top_files", len(files), [p.name for p in files[:10]])
        counts, examples = count_ext(path, [".mp4", ".json", ".txt", ".csv", ".png", ".jpg"])
        print("counts", counts)
        for suffix, items in examples.items():
            if items:
                print("examples", suffix, items)

for p in list((root / "external").rglob("*.json"))[:5]:
    print(f"\nJSON {p}")
    try:
        obj = json.load(open(p, encoding="utf-8"))
        print(type(obj), list(obj)[:8] if isinstance(obj, dict) else len(obj))
        print(str(obj)[:1200])
    except Exception as exc:
        print("ERR", exc)

