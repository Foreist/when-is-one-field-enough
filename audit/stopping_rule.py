# -*- coding: utf-8 -*-
"""Sequential stopping for chip-level QC under intra-chip label correlation.

Naive rule : Beta(1,1) posterior on the bad fraction assuming i.i.d. fields;
             stop when P(f>0.5) > 0.9 or P(f<0.5) > 0.9 or budget reached.
Corrected  : same, but with the empirical design-effect correction for random
             within-session sampling (n_eff = k / deff_emp), measured first.

Reported   : mean fields used, share of chips stopped early, and — crucially —
             the share of stops that are BOTH confident and WRONG (false confidence).
"""
import collections, json, math, random
from pathlib import Path

import numpy as np

from structure_probe import load

OUT = Path(__file__).resolve().parent.parent / "results"
KMAX = 20
TRIALS = 400


def sequences():
    recs, meta = load()
    by_sess = collections.defaultdict(list)
    for k, r in recs.items():
        by_sess[r["session"]].append(r)
    return {s: np.array([1 if r["cls"] == "good" else 0 for r in sorted(v, key=lambda r: r["idx"])])
            for s, v in by_sess.items()}


def measure_design_effect(seqs, ks=(4, 8, 12, 20), trials=400, seed=1):
    rng = random.Random(seed)
    out = {}
    for k in ks:
        ratios = []
        for s, seq in seqs.items():
            n = len(seq)
            if n < k:
                continue
            p = float((seq == 0).mean())
            if p in (0.0, 1.0):
                continue
            means = []
            for _ in range(trials):
                idx = rng.sample(range(n), k)
                means.append(float((seq[idx] == 0).mean()))
            var_emp = float(np.var(means))
            var_iid = p * (1 - p) / k
            if var_iid > 0:
                ratios.append(var_emp / var_iid)
        out[k] = dict(n_sessions=len(ratios), deff_median=float(np.median(ratios)),
                      deff_mean=float(np.mean(ratios)), deff_p90=float(np.percentile(ratios, 90)))
        print(f"  k={k:>2}  deff (empirical var / iid var): median {out[k]['deff_median']:.2f} "
              f"mean {out[k]['deff_mean']:.2f} p90 {out[k]['deff_p90']:.2f}  (n={len(ratios)})")
    return out


def log_beta_posterior(bad, good, deff):
    """P(f > 0.5) under Beta(1 + bad/deff, 1 + good/deff)."""
    from scipy.stats import beta as B
    a = 1 + bad / deff
    b = 1 + good / deff
    return float(1 - B.cdf(0.5, a, b))


def run_rule(seqs, deff, kmax=KMAX, conf=0.9, trials=TRIALS, seed=2):
    rng = random.Random(seed)
    used, stopped, false_conf, correct = [], [], [], []
    for s, seq in seqs.items():
        n = len(seq)
        if n < 4:
            continue
        true_bad = 1 if (seq == 0).mean() > 0.5 else 0
        for _ in range(trials):
            order = rng.sample(range(n), min(n, kmax))
            bad = good = 0
            for i, idx in enumerate(order, 1):
                if seq[idx] == 0:
                    bad += 1
                else:
                    good += 1
                p_bad = log_beta_posterior(bad, good, deff)
                if p_bad > conf or (1 - p_bad) > conf:
                    call = 1 if p_bad > 0.5 else 0
                    used.append(i); stopped.append(1)
                    correct.append(int(call == true_bad))
                    if (p_bad > conf or (1 - p_bad) > conf) and call != true_bad:
                        false_conf.append(1)
                    else:
                        false_conf.append(0)
                    break
            else:
                # budget exhausted: majority of the fields read; a tie is broken at random
                call = 1 if bad > good else 0 if bad < good else rng.randint(0, 1)
                used.append(len(order)); stopped.append(0)
                correct.append(int(call == true_bad)); false_conf.append(0)
    return dict(mean_fields=float(np.mean(used)), frac_early_stop=float(np.mean(stopped)),
                accuracy=float(np.mean(correct)), false_confident_rate=float(np.mean(false_conf)),
                n=len(used))


def main():
    seqs = sequences()
    print("sessions", len(seqs))
    print("empirical design effect of random k-field sampling within a session:")
    deff_tab = measure_design_effect(seqs)
    deff = float(np.median([v["deff_median"] for v in deff_tab.values()]))
    print(f"-> using deff = {deff:.2f} for the corrected rule")
    naive = run_rule(seqs, deff=1.0)
    corrected = run_rule(seqs, deff=deff)
    print("\nnaive  (i.i.d. posterior) :", json.dumps(naive, ensure_ascii=False))
    print("corrected (deff-adjusted) :", json.dumps(corrected, ensure_ascii=False))
    (OUT / "stopping_rule.json").write_text(json.dumps(
        dict(deff_table=deff_tab, deff_used=deff, naive=naive, corrected=corrected,
             kmax=KMAX, conf=0.9, trials=TRIALS), ensure_ascii=False, indent=1))
    print("saved", OUT / "stopping_rule.json")


if __name__ == "__main__":
    main()
