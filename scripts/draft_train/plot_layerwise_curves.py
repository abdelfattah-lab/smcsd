#!/usr/bin/env python3
"""Plot layerwise MLP training curves from train_log.jsonl.

This intentionally uses only the Python standard library and writes SVG, so it
works in minimal training environments without matplotlib.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def read_log(path: Path) -> Dict[int, List[Tuple[int, float]]]:
    curves: Dict[int, List[Tuple[int, float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["step"])
            for layer_s, loss in row.get("loss_by_layer", {}).items():
                curves.setdefault(int(layer_s), []).append((step, float(loss)))
    return curves


def moving_average(points: List[Tuple[int, float]], window: int) -> List[Tuple[int, float]]:
    if window <= 1 or len(points) <= 2:
        return points
    out: List[Tuple[int, float]] = []
    vals: List[float] = []
    for step, value in points:
        vals.append(value)
        if len(vals) > window:
            vals.pop(0)
        out.append((step, sum(vals) / len(vals)))
    return out


def nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 5, 10), key=lambda x: abs(x * mag - raw)) * mag
    start = math.floor(lo / step) * step
    ticks = []
    x = start
    while x <= hi + 0.5 * step:
        if x >= lo - 0.5 * step:
            ticks.append(x)
        x += step
    return ticks[: n + 2]


def make_svg(
    *,
    title: str,
    run_name: str,
    curves: Dict[int, List[Tuple[int, float]]],
    output: Path,
    width: int,
    height: int,
    smooth: int,
) -> None:
    curves = {layer: moving_average(points, smooth) for layer, points in curves.items()}
    all_steps = [step for points in curves.values() for step, _ in points]
    all_losses = [loss for points in curves.values() for _, loss in points]
    if not all_steps or not all_losses:
        raise ValueError("no points to plot")

    x_min, x_max = min(all_steps), max(all_steps)
    y_min, y_max = min(all_losses), max(all_losses)
    y_pad = max((y_max - y_min) * 0.08, 1e-6)
    y_min = max(0.0, y_min - y_pad)
    y_max = y_max + y_pad

    margin_l, margin_r, margin_t, margin_b = 68, 190, 52, 54
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def sx(x: float) -> float:
        if x_max == x_min:
            return margin_l + plot_w / 2
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        if y_max == y_min:
            return margin_t + plot_h / 2
        return margin_t + (y_max - y) / (y_max - y_min) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;font-size:13px;fill:#222}",
        ".title{font-size:18px;font-weight:700}.sub{font-size:12px;fill:#555}",
        ".axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.curve{fill:none;stroke-width:2.2}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text class="title" x="{margin_l}" y="26">{html.escape(title)}</text>',
        f'<text class="sub" x="{margin_l}" y="43">{html.escape(run_name)}; smoothing window={smooth}</text>',
    ]

    for tick in nice_ticks(y_min, y_max):
        y = sy(tick)
        lines.append(f'<line class="grid" x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text x="{margin_l - 10}" y="{y + 4:.1f}" text-anchor="end">{tick:.2g}</text>')

    for tick in nice_ticks(float(x_min), float(x_max)):
        x = sx(tick)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_t}" x2="{x:.1f}" y2="{margin_t + plot_h}"/>')
        lines.append(f'<text x="{x:.1f}" y="{margin_t + plot_h + 22}" text-anchor="middle">{int(tick)}</text>')

    lines.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}"/>')
    lines.append(f'<text x="{margin_l + plot_w / 2}" y="{height - 13}" text-anchor="middle">training step</text>')
    lines.append(f'<text transform="translate(18 {margin_t + plot_h / 2}) rotate(-90)" text-anchor="middle">KL loss</text>')

    for i, layer in enumerate(sorted(curves)):
        color = COLORS[i % len(COLORS)]
        pts = " ".join(f"{sx(step):.1f},{sy(loss):.1f}" for step, loss in curves[layer])
        lines.append(f'<polyline class="curve" stroke="{color}" points="{pts}"/>')
        lx = margin_l + plot_w + 24
        ly = margin_t + 22 + i * 22
        lines.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 24}" y2="{ly - 4}" stroke="{color}" stroke-width="3"/>')
        last = curves[layer][-1][1]
        lines.append(f'<text x="{lx + 32}" y="{ly}">layer {layer}: {last:.3f}</text>')

    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run directories or NAME=DIR entries containing train_log.jsonl")
    parser.add_argument("--output-dir", default="outputs/layerwise_plots")
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--width", type=int, default=1050)
    parser.add_argument("--height", type=int, default=620)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    for item in args.runs:
        if "=" in item:
            name, path_s = item.split("=", 1)
            run_dir = Path(path_s)
        else:
            run_dir = Path(item)
            name = run_dir.name
        log_path = run_dir / "train_log.jsonl"
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        curves = read_log(log_path)
        out = out_dir / f"{name}_learning_curves.svg"
        make_svg(
            title="Layerwise MLP KL Training Curves",
            run_name=name,
            curves=curves,
            output=out,
            width=args.width,
            height=args.height,
            smooth=args.smooth,
        )
        print(out)


if __name__ == "__main__":
    main()
