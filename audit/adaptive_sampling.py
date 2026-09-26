# -*- coding: utf-8 -*-
"""Fixed-budget sampling policies on the measured label sequences.

Policies (budget k fields per chip):
  random   : k distinct uniform fields
  window   : one contiguous window of length k (start uniform)
  adaptive : random start; if the last field was BAD, expand to an unsampled
             neighbour (prefer the side with more unsampled room); if GOOD, jump
             to a new uniform unsampled field.

Metrics vs the full-session label sequence:
  chip_acc   : majority call of the sample == majority of the session
  any_bad    : fraction of trials that find >=1 bad field  (sensitivity to badness)
  bad_recall : |sampled ∩ bad| / |bad|          (how much of the bad region is seen)
  bad_prec   : |sampled ∩ bad| / k              (hit rate)
  frac_err   : |mean(sample) - mean(session)|   (chip-level fraction estimate error)
"""
import collections, json, random
from pathlib import Path

import numpy as np

from structure_probe import load

OUT = Path(__file__).resolve().parent.parent / "results"
TRIALS = 300
KS = [2, 4, 8, 12, 20]


def sequences():
    recs, meta = load()
    by_sess = collections.defaultdict(list)
    for k, r in recs.items():
        by_sess[r["session"]].append(r)
    seqs = {}
    for s, v in by_sess.items():
        v = sorted(v, key=lambda r: r["idx"])
        seqs[s] = np.array([1 if r["cls"] == "good" else 0 for r in v])   # 1=good
    return seqs


def sample_random(n, k, rng):
    return rng.sample(range(n), k)


def sample_window(n, k, rng):
    if k >= n:
        return list(range(n))
    start = rng.randrange(0, n - k + 1)
    return list(range(start, start + k))


def sample_adaptive(seq, k, rng):
    n = len(seq)
    sampled = []
    taken = set()
    i = rng.randrange(n)
    sampled.append(i); taken.add(i)
    while len(sampled) < k:
        if seq[sampled[-1]] == 0:                     # last field BAD -> expand
            cands = [j for j in (sampled[-1] - 1, sampled[-1] + 1) if 0 <= j < n and j not in taken]
            if not cands:                              # both taken -> nearest unsampled
                dist = [(abs(p - sampled[-1]), p) for p in range(n) if p not in taken]
                if not dist:
                    break
                nxt = min(dist)[1]
            else:
                nxt = cands[0] if len(cands) == 1 else rng.choice(cands)
        else:                                          # GOOD -> jump elsewhere
            free = [p for p in range(n) if p not in taken]
            if not free:
                break
            nxt = rng.choice(free)
        sampled.append(nxt); taken.add(nxt)
    return sampled


def run(seqs, k, trials=TRIALS, seed=0):
    rng = random.Random(seed)
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, seq in seqs.items():
        n = len(seq)
        if n < k:
            continue
        bad_idx = set(np.where(seq == 0)[0].tolist())
        true_bad = 1 if len(bad_idx) * 2 > n else 0
        for pol, fn in (("random", lambda: sample_random(n, k, rng)),
                        ("window", lambda: sample_window(n, k, rng)),
                        ("adaptive", lambda: sample_adaptive(seq, k, rng))):
            for _ in range(trials):
                idx = fn()
                y = seq[idx]
                fb = 1 - y.mean()
                # even k can tie; break ties at random (calling a tie 'good' favours the policies
                # that tie most often, since 'good' is the majority class)
                est_bad = 1 if fb > 0.5 else 0 if fb < 0.5 else rng.randint(0, 1)
                hit = len(set(idx) & bad_idx)
                out[pol]["chip_acc"].append(int(est_bad == true_bad))
                out[pol]["any_bad"].append(int(hit > 0))
                out[pol]["bad_recall"].append(hit / len(bad_idx) if bad_idx else np.nan)
                out[pol]["bad_prec"].append(hit / len(idx))
                out[pol]["frac_err"].append(abs((1 - y.mean()) - (1 - seq.mean())))
    res = {}
    for pol, d in out.items():
        res[pol] = dict(chip_acc=float(np.mean(d["chip_acc"])),
                        any_bad=float(np.nanmean(d["any_bad"])),
                        bad_recall=float(np.nanmean(d["bad_recall"])),
                        bad_prec=float(np.mean(d["bad_prec"])),
                        frac_err=float(np.mean(d["frac_err"])), n=len(d["chip_acc"]))
    return res


def main():
    seqs = sequences()
    print("sessions", len(seqs), "fields", sum(len(v) for v in seqs.values()),
          "bad fraction", round(float(np.mean(np.concatenate(list(seqs.values())) == 0)), 3))
    allres = {}
    for k in KS:
        r = run(seqs, k)
        allres[k] = r
        print(f"\nk={k}  (n trials {r['random']['n']})")
        for pol in ("random", "window", "adaptive"):
            d = r[pol]
            print(f"  {pol:<9} chip_acc {d['chip_acc']:.3f}  any_bad {d['any_bad']:.3f}  "
                  f"bad_recall {d['bad_recall']:.3f}  bad_prec {d['bad_prec']:.3f}  frac_err {d['frac_err']:.3f}")
    (OUT / "adaptive_sampling.json").write_text(json.dumps(allres, ensure_ascii=False, indent=1))
    print("\nsaved", OUT / "adaptive_sampling.json")


if __name__ == "__main__":
    main()
