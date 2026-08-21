"""MATH500 accuracy vs compute-per-question, SMC vs TTS baselines.

Data: 2026-08-20/21 runs on 8xB200 (seed 0, 100 questions, temp 0.7,
non-thinking). Cost = wall-clock x GPUs used / questions, batch-throughput
runs. SMC config: Qwen3.5-4B draft, gamma=8, tau=0 (no resampling),
particle-majority answer. GSI uses 3 engines (4B step proposer, 27B reward,
target); Verifier-BoN judge cost estimated at +0.4 GPU.s/q over SC@8.

Usage: python figures/plot_math500_frontier.py  (writes PNG+PDF next to it)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "AR@1":                {"color": "#8A8578", "marker": "o", "fc": "none"},
    "Self-consistency":    {"color": "#3E6C8E", "marker": "s"},
    "Verifier-BoN@8":      {"color": "#7A6C9E", "marker": "D"},
    "GSI@8":               {"color": "#5E8C61", "marker": "^"},
    "SMC (ours)":          {"color": "#C8354F", "marker": "o", "big": True},
}

# method, config-label (annotated next to point), GPU.s/question, accuracy
PANELS = [
    ("Qwen3.8-27B target (1 GPU)", "math500_frontier_27b", [
        ("AR@1", None, 0.27, 57),
        ("Self-consistency", "n=8", 1.72, 70),
        ("Verifier-BoN@8", None, 2.12, 69),
        ("GSI@8", None, 15.5, 65),
        ("SMC (ours)", "N=8", 1.56, 72),
        ("SMC (ours)", "N=32", 6.23, 79),
    ]),
    ("Qwen3.5-397B-A17B target (FP8, 4 GPUs)", "math500_frontier_397b", [
        ("AR@1", None, 0.79, 65),
        ("Self-consistency", "n=8", 5.23, 70),
        ("Self-consistency", "n=64", 39.5, 77),
        ("Verifier-BoN@8", None, 5.63, 70),
        ("GSI@8", None, 30.7, 68),
        ("SMC (ours)", "N=8", 5.76, 77),
        ("SMC (ours)", "N=32", 14.9, 83),
    ]),
]

here = os.path.dirname(os.path.abspath(__file__))
for title, fname, pts in PANELS:
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    seen = set()
    for method, cfg, x, y in pts:
        st = STYLE[method]
        ax.scatter(
            x, y,
            s=110 if st.get("big") else 70,
            marker=st["marker"],
            facecolors=st.get("fc", st["color"]),
            edgecolors=st["color"],
            linewidths=1.6,
            label=method if method not in seen else None,
            zorder=3,
        )
        seen.add(method)
        if cfg:
            ax.annotate(
                cfg, (x, y),
                textcoords="offset points", xytext=(7, 5),
                fontsize=8.5, color=st["color"],
            )
    ax.set_xscale("log")
    ax.set_xlim(0.2, 60)
    ax.set_ylim(54, 86)
    ax.set_xticks([0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.3", "1", "3", "10", "30"])
    ax.set_xlabel("GPU-seconds per question")
    ax.set_ylabel("MATH500 accuracy (%)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(here, f"{fname}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"wrote {fname}.png/.pdf")
