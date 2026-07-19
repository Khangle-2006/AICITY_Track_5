from __future__ import annotations

import argparse
import csv
import html
import time
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append(
                        {
                            "step": float(row["global_step"]),
                            "loss": float(row["loss"]),
                            "lr": float(row["learning_rate"]),
                            "step_time": float(row["step_time"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return rows
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    acc = 0.0
    q: list[float] = []
    for value in values:
        q.append(value)
        acc += value
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out


def write_html(path: Path, image_path: Path, title: str, interval: float) -> None:
    rel_image = html.escape(image_path.name)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<meta charset='utf-8'>",
                f"<meta http-equiv='refresh' content='{max(1, int(interval))}'>",
                f"<title>{html.escape(title)}</title>",
                "<style>body{margin:24px;font-family:Arial,sans-serif;background:#111;color:#eee}"
                "img{max-width:100%;height:auto;border:1px solid #333}</style>",
                f"<h2>{html.escape(title)}</h2>",
                f"<img src='{rel_image}?t={time.time():.0f}' alt='loss curve'>",
            ]
        ),
        encoding="utf-8",
    )


def plot_svg(rows: list[dict[str, float]], out_svg: Path, title: str, ma_window: int) -> None:
    width, height = 1120, 640
    pad_l, pad_r, pad_t, pad_b = 78, 28, 64, 74
    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    smooth = moving_average(losses, max(1, ma_window))
    if not steps:
        return

    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(losses + smooth), max(losses + smooth)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    def xy(step: float, loss: float) -> tuple[float, float]:
        x = pad_l + (step - x_min) / (x_max - x_min) * (width - pad_l - pad_r)
        y = pad_t + (y_max - loss) / (y_max - y_min) * (height - pad_t - pad_b)
        return x, y

    raw_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(s, v) for s, v in zip(steps, losses)))
    smooth_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(s, v) for s, v in zip(steps, smooth)))
    latest = losses[-1]
    best = min(losses)
    grid = []
    labels = []
    for i in range(6):
        y = pad_t + i / 5 * (height - pad_t - pad_b)
        value = y_max - i / 5 * (y_max - y_min)
        grid.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width-pad_r}' y2='{y:.1f}' stroke='#333'/>")
        labels.append(f"<text x='{pad_l-10}' y='{y+4:.1f}' text-anchor='end' fill='#aaa' font-size='13'>{value:.3f}</text>")
    for i in range(6):
        x = pad_l + i / 5 * (width - pad_l - pad_r)
        value = x_min + i / 5 * (x_max - x_min)
        grid.append(f"<line x1='{x:.1f}' y1='{pad_t}' x2='{x:.1f}' y2='{height-pad_b}' stroke='#292929'/>")
        labels.append(f"<text x='{x:.1f}' y='{height-pad_b+28}' text-anchor='middle' fill='#aaa' font-size='13'>{value:.0f}</text>")

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text(
        "\n".join(
            [
                f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
                "<rect width='100%' height='100%' fill='#111'/>",
                f"<text x='{pad_l}' y='34' fill='#eee' font-size='22' font-family='Arial'>{html.escape(title)}</text>",
                *grid,
                f"<polyline points='{raw_points}' fill='none' stroke='#6aa9ff' stroke-opacity='0.38' stroke-width='2'/>",
                f"<polyline points='{smooth_points}' fill='none' stroke='#ffcc66' stroke-width='4'/>",
                "<line x1='{0}' y1='{1}' x2='{2}' y2='{1}' stroke='#777'/>".format(pad_l, height - pad_b, width - pad_r),
                "<line x1='{0}' y1='{1}' x2='{0}' y2='{2}' stroke='#777'/>".format(pad_l, pad_t, height - pad_b),
                *labels,
                f"<text x='{width-pad_r-8}' y='{pad_t+26}' text-anchor='end' fill='#eee' font-size='16' font-family='Arial'>latest={latest:.4f}  best={best:.4f}  steps={int(steps[-1])}</text>",
                f"<text x='{width-pad_r-8}' y='{pad_t+52}' text-anchor='end' fill='#ffcc66' font-size='14' font-family='Arial'>moving avg {ma_window}</text>",
                "<text x='{0}' y='{1}' text-anchor='middle' fill='#bbb' font-size='15' font-family='Arial'>training step</text>".format((pad_l + width - pad_r) / 2, height - 22),
                "<text x='22' y='{0}' text-anchor='middle' fill='#bbb' font-size='15' font-family='Arial' transform='rotate(-90 22,{0})'>loss</text>".format((pad_t + height - pad_b) / 2),
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def plot(rows: list[dict[str, float]], out_png: Path, title: str, ma_window: int) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        out_svg = out_png.with_suffix(".svg")
        plot_svg(rows, out_svg, title, ma_window)
        return out_svg

    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    smooth = moving_average(losses, max(1, ma_window))

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=140)
    ax.plot(steps, losses, color="#6aa9ff", alpha=0.35, linewidth=1.0, label="loss")
    ax.plot(steps, smooth, color="#ffcc66", linewidth=2.0, label=f"moving avg {ma_window}")
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    if losses:
        latest = losses[-1]
        best = min(losses)
        ax.text(
            0.99,
            0.98,
            f"latest={latest:.4f}\nbest={best:.4f}\nsteps={int(steps[-1])}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#ddd"},
        )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out_png", type=Path, default=None)
    parser.add_argument("--out_html", type=Path, default=None)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--ma_window", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    out_png = args.out_png or args.csv.with_name("loss_curve_live.png")
    out_html = args.out_html or args.csv.with_name("loss_curve_live.html")
    title = f"Loss curve: {args.csv.parent.name}"

    while True:
        rows = read_rows(args.csv)
        if rows:
            image_path = plot(rows, out_png, title, args.ma_window)
            write_html(out_html, image_path, title, args.interval)
            latest = rows[-1]
            print(
                f"step={int(latest['step'])} loss={latest['loss']:.5f} "
                f"lr={latest['lr']:.2e} image={image_path}",
                flush=True,
            )
        else:
            print(f"waiting for {args.csv}", flush=True)

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
