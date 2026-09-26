# -*- coding: utf-8 -*-
"""Quantify the block structure of OOC QC labels and what it means for sampling.

1) intraclass correlation (ICC) of the binary label within sessions / cell-type runs
2) lag-1 autocorrelation and run-length distribution
3) field-sampling simulation: agreement of a k-field random-sample majority with the session
   majority (saved as sampling_sim; not used in the report, which uses label_sufficiency.py)
"""
import collections, json, math
from pathlib import Path

import numpy as np

from structure_probe import load, runs_test

OUT = Path(__file__).resolve().parent.parent / "results"


def icc_binary(groups):
    """One-way ANOVA ICC(1) for a binary outcome. groups = list of arrays."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 1]
    k = len(groups)
    n = np.array([len(g) for g in groups])
    N = n.sum()
    if k < 2 or N <= k:
        return float("nan"), float("nan")
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    m0 = (N - (n ** 2).sum() / N) / (k - 1)
    denom = msb + (m0 - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else float("nan")
    return float(icc), float(1 + (m0 - 1) * icc)     # icc, design effect


def main():
    recs, meta = load()
    joined = []
    for k, r in recs.items():
        cand = [f'{r["session"]}_{r["idx"]:02d}', f'{r["session"]}_{r["idx"]}']
        m = next((meta[c] for c in cand if c in meta), None)
        if m is None:
            continue
        d = dict(r); d["meta_day"] = m.get("day")
        joined.append(d)
    by_sess = collections.defaultdict(list)
    for r in joined:
        by_sess[r["session"]].append(r)

    sess_groups, run_groups, lag1, runlens = [], [], [], []
    for s, v in by_sess.items():
        v = sorted(v, key=lambda r: r["idx"])
        labs = np.array([1 if r["cls"] == "good" else 0 for r in v], dtype=float)
        if len(labs) > 1:
            sess_groups.append(labs)
            x = labs - labs.mean()
            denom = (x ** 2).sum()
            if denom > 0:
                lag1.append(float((x[:-1] * x[1:]).sum() / denom))
            # run lengths
            cur = 1
            for i in range(1, len(labs)):
                if labs[i] == labs[i - 1]:
                    cur += 1
                else:
                    runlens.append(cur); cur = 1
            runlens.append(cur)
        cells = [r["cell"] for r in v]
        start = 0
        for i in range(1, len(v) + 1):
            if i == len(v) or cells[i] != cells[start]:
                sub = labs[start:i]
                if len(sub) > 1:
                    run_groups.append(sub)
                start = i

    icc_s, deff_s = icc_binary(sess_groups)
    icc_r, deff_r = icc_binary(run_groups)
    print(f"ICC session-level {icc_s:.3f}  design effect {deff_s:.2f}  "
          f"-> effective N = {len(joined)}/{deff_s:.1f} = {len(joined)/deff_s:.0f}")
    print(f"ICC cell-run-level {icc_r:.3f}  design effect {deff_r:.2f}  "
          f"-> effective N = {len(joined)}/{deff_r:.1f} = {len(joined)/deff_r:.0f}")
    print(f"lag-1 autocorrelation: mean {np.mean(lag1):.3f}  median {np.median(lag1):.3f}  "
          f"frac>0.3 {np.mean(np.array(lag1) > 0.3):.2f}  (n={len(lag1)} sessions)")
    rl = np.array(runlens)
    print(f"run length: mean {rl.mean():.1f}  median {np.median(rl):.0f}  p90 {np.percentile(rl, 90):.0f}  max {rl.max()}")
    # i.i.d. expectation of run length for the observed good fraction
    p = np.mean([1 if r["cls"] == "good" else 0 for r in joined])
    exp_run = 1 / (2 * p * (1 - p))
    print(f"i.i.d. expected run length {exp_run:.2f}  (observed {rl.mean():.2f})")

    # ---- sampling simulation ----
    rng = np.random.default_rng(7)
    sim = {}
    for k in [1, 2, 3, 5, 8, 12, 20]:
        agree, sens, spec = [], [], []
        for s, v in by_sess.items():
            v = sorted(v, key=lambda r: r["idx"])
            labs = np.array([1 if r["cls"] == "good" else 0 for r in v])
            if len(labs) < k:
                continue
            full_bad = 1 - int(labs.mean() >= 0.5)          # chip call: majority bad
            for _ in range(40):
                idx = rng.choice(len(labs), size=k, replace=False)
                samp = labs[idx]
                call_bad = 1 - int(samp.mean() >= 0.5)
                agree.append(int(call_bad == full_bad))
                if full_bad:
                    sens.append(int(call_bad == 1))
                else:
                    spec.append(int(call_bad == 0))
        sim[k] = dict(n=len(agree), agree=float(np.mean(agree)),
                      sens=float(np.mean(sens)) if sens else float("nan"),
                      spec=float(np.mean(spec)) if spec else float("nan"))
        # i.i.d. prediction for the same chip-level call
        print(f"k={k:>2}  agree {sim[k]['agree']:.3f}  sens {sim[k]['sens']:.3f}  spec {sim[k]['spec']:.3f}")

    out = dict(icc_session=icc_s, deff_session=deff_s, n_eff_session=len(joined) / deff_s,
               icc_cellrun=icc_r, deff_cellrun=deff_r, n_eff_cellrun=len(joined) / deff_r,
               lag1_mean=float(np.mean(lag1)), lag1_median=float(np.median(lag1)),
               frac_lag1_gt03=float(np.mean(np.array(lag1) > 0.3)),
               runlen_mean=float(rl.mean()), runlen_median=float(np.median(rl)),
               runlen_p90=float(np.percentile(rl, 90)), runlen_max=int(rl.max()),
               runlen_iid_expected=float(exp_run), good_frac=float(p),
               sampling_sim=sim, n_images=len(joined), n_sessions=len(by_sess))
    (OUT / "block_structure.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved", OUT / "block_structure.json")


if __name__ == "__main__":
    main()
