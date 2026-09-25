#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired bootstrap over chips for the model-in-the-loop policy comparison (report §4.3, Fig 4c).

Re-runs the exact trials of `03c_policy_model_in_loop.py` (same split, model, rng seed and
trial count), keeps each chip's mean accuracy per policy, and resamples *chips* with
replacement to put an interval on every paired difference. Chips are the unit because the
trials within a chip share one sequence of fields and are not independent. Six differences are
tested (3 pairs x 2 budgets), so each gets a plain 95% interval and a Bonferroni one (1 - 0.05/6).

Writes results/policy_bootstrap.json
"""
import argparse, collections, json, random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import TF, load_model                          # noqa: E402
from leakage_experiment import index_images                  # noqa: E402
from leakage_controlled import split_controlled               # noqa: E402

import importlib
P = importlib.import_module("03c_policy_model_in_loop")

OUT = Path(__file__).resolve().parent.parent / "results"
POL = ("random", "window", "adaptive")
PAIRS = (("random", "adaptive"), ("window", "adaptive"), ("random", "window"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 12])
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    recs = index_images()
    sp = split_controlled(recs, seed=1000, n_test_sessions=25)
    by = collections.defaultdict(list)
    for r in sp["disjoint"]["test"]:
        by[r["session"]].append(r)
    model = load_model(str(Path(__file__).resolve().parent.parent / "model" / "perfield_mnv3s_384_s0.pt"))

    chips = {}
    with torch.no_grad():
        for s, v in by.items():
            v = sorted(v, key=lambda r: int(r["name"].split("_")[1].split(".")[0]))
            probs = []
            for i in range(0, len(v), 32):
                batch = torch.stack([TF(Image.open(r["path"]).convert("RGB")) for r in v[i:i + 32]])
                probs += torch.softmax(model(batch), dim=1)[:, 0].tolist()
            chips[s] = dict(p=probs, bad=[1 if r["cls"] == "bad" else 0 for r in v])

    res = {}
    for k in args.ks:
        rng = random.Random(0)                       # same stream as 03c
        per_chip = {p: [] for p in POL}
        for s, c in chips.items():
            n = len(c["p"])
            if n < k:
                continue
            ref_bad = int(np.mean(c["bad"]) > 0.5)
            seq_bad = [pi > 0.5 for pi in c["p"]]
            for pol, fn in (("random", lambda: P.sample_random(n, k, rng)),
                            ("window", lambda: P.sample_window(n, k, rng)),
                            ("adaptive", lambda: P.sample_adaptive(seq_bad, k, rng))):
                hits = [int(int(np.mean([c["p"][i] for i in fn()]) > 0.5) == ref_bad)
                        for _ in range(args.trials)]
                per_chip[pol].append(np.mean(hits))
        a = {p: np.array(v) for p, v in per_chip.items()}
        m = len(a["random"])
        brng = np.random.default_rng(0)
        idx = brng.integers(0, m, size=(args.boot, m))
        out = {}
        alpha = 0.05 / (len(PAIRS) * len(args.ks))
        for x, y in PAIRS:
            diff = a[x] - a[y]
            d = diff[idx].mean(axis=1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            blo, bhi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[f"{x} - {y}"] = dict(mean=float(diff.mean()), lo=float(lo), hi=float(hi),
                                     significant=bool(lo > 0 or hi < 0),
                                     bonf_lo=float(blo), bonf_hi=float(bhi),
                                     bonf_significant=bool(blo > 0 or bhi < 0),
                                     chips_differing=int(np.sum(diff != 0)))
        res[f"k{k}"] = out
        res[f"n_chips_k{k}"] = m
        print(f"k={k} ({m} chips): " + "  ".join(f"{n} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] bonf [{v['bonf_lo']:+.3f}, {v['bonf_hi']:+.3f}]"
                                                 for n, v in out.items()))
    (OUT / "policy_bootstrap.json").write_text(json.dumps(res, indent=1))
    print("saved", OUT / "policy_bootstrap.json")


if __name__ == "__main__":
    main()
