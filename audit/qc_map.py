# -*- coding: utf-8 -*-
"""Render per-chip QC maps: true label strip vs model P(bad), in acquisition order.

Usage: python3 qc_map.py --preds perfield_preds_384_s0.json --out qc_maps_384.png [--top 8]
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="perfield_preds_384_s0.json")
    ap.add_argument("--out", default="qc_maps_384.png")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    d = json.loads((OUT / args.preds).read_text())
    t = d["test"]
    by = collections.defaultdict(list)
    for p, y, s, i in zip(t["p"], t["y"], t["session"], t["idx"]):
        by[s].append((i, p, y))
    # rank sessions by "how much of the chip is bad" and by model error, pick a mix
    stats = []
    for s, v in by.items():
        v.sort()
        y = np.array([x[2] for x in v]); p = np.array([x[1] for x in v])
        stats.append((s, len(v), float((y == 0).mean()), float(((p > 0.5) == (y == 0)).mean())))
    stats.sort(key=lambda x: -x[2])
    picks = [x[0] for x in stats[:args.top // 2]] + [x[0] for x in stats[len(stats) // 2 - args.top // 4:len(stats) // 2 + args.top // 4]]
    picks = list(dict.fromkeys(picks))[:args.top]

    n = len(picks)
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.6 * n + 1.2), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, picks):
        v = sorted(by[s])
        idx = [x[0] for x in v]; y = np.array([x[2] for x in v]); p = np.array([x[1] for x in v])
        x = np.arange(len(v))
        ax.imshow((y == 0)[None, :], aspect="auto", cmap="Reds", vmin=0, vmax=1,
                  extent=[0, len(v), 1.15, 1.45], alpha=0.85)
        ax.plot(x, p, color="#1f77b4", lw=1.6, label="model P(bad)")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_ylim(-0.02, 1.5)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("P(bad)", fontsize=8)
        acc = ((p > 0.5) == (y == 0)).mean()
        ax.set_title(f"chip {s}  n={len(v)}  true bad {100*(y==0).mean():.0f}%  field acc {acc:.2f}",
                     fontsize=9, loc="left")
        if ax is axes[-1]:
            ax.set_xlabel("field index (acquisition order) — red strip = true 'bad' label", fontsize=8)
    fig.suptitle("OOC chip QC maps (session-disjoint test): labels are contiguous blocks, model tracks them",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / args.out, dpi=130)
    print("saved", OUT / args.out)
    print("field acc on these chips:", {s: round(float(((np.array([x[1] for x in sorted(by[s])]) > 0.5) ==
          (np.array([x[2] for x in sorted(by[s])]) == 0)).mean()), 3) for s in picks})


if __name__ == "__main__":
    main()
