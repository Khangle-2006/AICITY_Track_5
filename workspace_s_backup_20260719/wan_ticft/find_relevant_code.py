from pathlib import Path

roots = [Path("/workspace/wan_ticft"), Path("/workspace/experiments/wan_ticft")]
terms = ("masked_pool", "flow_conf", "train5000", "external", "bdd_pc", "retriev", "raft", "compute_flow", "refiner", "motion")

for root in roots:
    print(f"\n== {root} ==")
    shown = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = str(path).lower()
        if any(term in lower for term in terms):
            print(path)
            shown += 1
            if shown >= 180:
                break

print("\n== content hits ==")
shown = 0
for root in [Path("/workspace/wan_ticft")]:
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            continue
        hits = [term for term in terms if term in text]
        if hits:
            print(path, hits[:8])
            shown += 1
            if shown >= 120:
                raise SystemExit

