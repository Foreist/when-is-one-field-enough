# -*- coding: utf-8 -*-
"""Paired confidence intervals for the controlled leakage A/B (results/leakage_controlled.json).

Each seed draws its own test sessions and trains both arms on them, so seeds are the independent
unit; the interval is a paired t-interval over the per-seed differences (leaky - disjoint).

Writes results/leakage_ci.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / "results"


def main():
    d = json.loads((OUT / "leakage_controlled.json").read_text())
    runs = {(r["tag"], r["seed"]): r for r in d["runs"]}
    seeds = sorted({s for _, s in runs})
    res = dict(n_seeds=len(seeds))
    for key in ("acc", "auc"):
        diff = np.array([runs["leaky", s][key] - runs["disjoint", s][key] for s in seeds])
        m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
        h = stats.t.ppf(0.975, len(diff) - 1) * se
        res[key] = dict(mean=float(m), lo=float(m - h), hi=float(m + h),
                        per_seed=[float(x) for x in diff], n_positive=int((diff > 0).sum()),
                        p_two_sided=float(stats.ttest_1samp(diff, 0).pvalue))
        print(f"{key}: +{100*m:.1f} pp  95% CI [{100*(m-h):.1f}, {100*(m+h):.1f}]  "
              f"{int((diff > 0).sum())}/{len(diff)} seeds positive, range "
              f"{100*diff.min():.1f}..{100*diff.max():.1f}")
    (OUT / "leakage_ci.json").write_text(json.dumps(res, indent=1))
    print("saved", OUT / "leakage_ci.json")


if __name__ == "__main__":
    main()
