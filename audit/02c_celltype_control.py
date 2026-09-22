#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cell-type control for the runs test (report §4.2).

A session can contain several cell types imaged in contiguous blocks, which would create
clustering for a trivial reason. This script repeats the runs test *inside* maximal
contiguous same-cell-type stretches and reports how many remain significant.

Writes results/celltype_control.json
"""
import collections, json
from pathlib import Path

from structure_probe import load, runs_test

OUT = Path(__file__).resolve().parent.parent / "results"


def main():
    recs, meta = load()
    joined = []
    for k, r in recs.items():
        cand = [f'{r["session"]}_{r["idx"]:02d}', f'{r["session"]}_{r["idx"]}']
        m = next((meta[c] for c in cand if c in meta), None)
        if m is None:
            continue
        joined.append(dict(r))
    by = collections.defaultdict(list)
    for r in joined:
        by[r["session"]].append(r)

    rows = []
    for s, v in by.items():
        v = sorted(v, key=lambda r: r["idx"])
        labs = [1 if r["cls"] == "good" else 0 for r in v]
        cells = [r["cell"] for r in v]
        start = 0
        for i in range(1, len(v) + 1):
            if i == len(v) or cells[i] != cells[start]:
                sub = labs[start:i]
                if len(sub) >= 8 and 0 < sum(sub) < len(sub):
                    rt = runs_test(sub)
                    if rt:
                        rows.append(dict(session=s, cell=cells[start], n=len(sub),
                                         good_frac=round(sum(sub) / len(sub), 3),
                                         runs=rt[0], expected=round(rt[1], 1),
                                         z=round(rt[2], 2), p=rt[3]))
                start = i

    n_sig = sum(1 for r in rows if r["p"] < 0.05)
    n_clust = sum(1 for r in rows if r["z"] < 0)
    out = dict(n_stretches=len(rows), n_significant=n_sig, n_clustered=n_clust,
               frac_clustered=n_clust / len(rows), rows=sorted(rows, key=lambda r: r["p"]))
    (OUT / "celltype_control.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"same-cell-type stretches tested: {len(rows)}")
    print(f"  p<0.05: {n_sig}   clustered (z<0): {n_clust} ({n_clust/len(rows):.0%})")
    print("  most clustered:")
    for r in out["rows"][:5]:
        print(f"    {r['session']} {r['cell']:<18} n={r['n']:>3} good={r['good_frac']:.2f} "
              f"runs={r['runs']:>3} (exp {r['expected']:>5}) z={r['z']:>6}")
    print("saved", OUT / "celltype_control.json")


if __name__ == "__main__":
    main()
