from pathlib import Path

roots = [Path("/data/Track_5"), Path("/workspace/experiments/wan_ticft")]
terms = ("bdd", "pc", "train", "val", "wts")

for root in roots:
    print(f"\n== {root} ==")
    if not root.exists():
        print("missing")
        continue
    shown = 0
    for path in root.rglob("*"):
        name = str(path).lower()
        if any(term in name for term in terms):
            print(path)
            shown += 1
            if shown >= 120:
                break

