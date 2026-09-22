#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Are consecutive fields near-duplicates? (report §4.2)

Clustered labels would be trivial if the dataset contained many near-identical frames.
This script measures, for a sample of sessions:
  * correlation between consecutive fields vs random pairs (downscaled grayscale)
  * a greedy "view clustering": a new view starts when the correlation with the previous
    field drops below --thr  ->  the number of distinct views

Writes results/redundancy.json
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
from PIL import Image

from structure_probe import load

OUT = Path(__file__).resolve().parent.parent / "results"


def vec(path, size=64):
    a = np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    return a.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.95, help="correlation above which two fields are one view")
    ap.add_argument("--n-largest", type=int, default=6)
    ap.add_argument("--n-smallest", type=int, default=6)
    args = ap.parse_args()

    recs, _ = load()
    by = collections.defaultdict(list)
    for k, r in recs.items():
        by[r["session"]].append(r)
    sizes = sorted(((len(v), s) for s, v in by.items()), reverse=True)
    picks = [s for _, s in sizes[:args.n_largest]] + [s for _, s in sizes[-args.n_smallest:]]

    rng = np.random.default_rng(0)
    out = {}
    for s in picks:
        v = sorted(by[s], key=lambda r: r["idx"])
        vs = [vec(r["path"]) for r in v]
        cons = [float(np.mean(vs[i] * vs[i + 1])) for i in range(len(vs) - 1)]
        pairs = min(400, max(1, len(vs) * (len(vs) - 1) // 2))
        ii = rng.integers(0, len(vs), size=(pairs, 2))
        rand = [float(np.mean(vs[i] * vs[j])) for i, j in ii if i != j]
        views = 1
        for i in range(1, len(vs)):
            if float(np.mean(vs[i] * vs[i - 1])) < args.thr:
                views += 1
        out[s] = dict(n=len(v), cons_mean=float(np.mean(cons)), cons_median=float(np.median(cons)),
                      cons_above_thr=float(np.mean(np.array(cons) > 0.99)),
                      rand_mean=float(np.mean(rand)), views=views)
        print(f"{s}: n={len(v):>3}  consecutive r={np.mean(cons):.3f} (median {np.median(cons):.3f})  "
              f"random r={np.mean(rand):.3f}  views={views}")

    tot_n = sum(o["n"] for o in out.values())
    tot_v = sum(o["views"] for o in out.values())
    summary = dict(threshold=args.thr, sessions=len(out), images=tot_n, views=tot_v,
                   frac_distinct=tot_v / tot_n,
                   cons_mean=float(np.mean([o["cons_mean"] for o in out.values()])),
                   rand_mean=float(np.mean([o["rand_mean"] for o in out.values()])), per_session=out)
    (OUT / "redundancy.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\nTOTAL {tot_n} images -> {tot_v} distinct views ({tot_v/tot_n:.2f})")
    print("saved", OUT / "redundancy.json")


if __name__ == "__main__":
    main()
