#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the sampling-policy ranking survive with the model in the loop? (report §4.3)

`adaptive_sampling.py` compares policies on the *labels*, which is exactly the practice we
criticise elsewhere. This script repeats the comparison using the deployed model's per-field
probabilities on the 25 session-disjoint test chips, so the ranking is measured with the
model in the loop.

Policies (budget k, fields chosen from the chip's fields):
  random   : k distinct uniform fields
  window   : one contiguous window of k fields
  adaptive : random start; expand to an unsampled neighbour when the last field scored bad

Chip call  : mean of the sampled P(bad) > 0.5, compared with the chip's majority label.
Localisation: fraction of the chip's truly-bad fields that the sampled set contains.

Writes results/policy_model_in_loop.json
"""
import argparse, collections, json, random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import TF, load_model, spread_order          # noqa: E402
from leakage_experiment import index_images                  # noqa: E402
from leakage_controlled import split_controlled               # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"


def sample_random(n, k, rng):
    return rng.sample(range(n), k)


def sample_window(n, k, rng):
    if k >= n:
        return list(range(n))
    s = rng.randrange(0, n - k + 1)
    return list(range(s, s + k))


def sample_adaptive(seq_bad, k, rng):
    n = len(seq_bad); taken = set()
    i = rng.randrange(n); taken.add(i); out = [i]
    while len(out) < k:
        if seq_bad[out[-1]]:
            cands = [j for j in (out[-1] - 1, out[-1] + 1) if 0 <= j < n and j not in taken]
            if not cands:
                free = [p for p in range(n) if p not in taken]
                if not free:
                    break
                nxt = min(free, key=lambda p: abs(p - out[-1]))
            else:
                nxt = cands[0] if len(cands) == 1 else rng.choice(cands)
        else:
            free = [p for p in range(n) if p not in taken]
            if not free:
                break
            nxt = rng.choice(free)
        out.append(nxt); taken.add(nxt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 12])
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
        rng = random.Random(0)
        acc = {p: [] for p in ("random", "window", "adaptive")}
        rec = {p: [] for p in ("random", "window", "adaptive")}
        for s, c in chips.items():
            n = len(c["p"])
            if n < k:
                continue
            ref_bad = int(np.mean(c["bad"]) > 0.5)
            seq_bad = [pi > 0.5 for pi in c["p"]]
            for pol, fn in (("random", lambda: sample_random(n, k, rng)),
                            ("window", lambda: sample_window(n, k, rng)),
                            ("adaptive", lambda: sample_adaptive(seq_bad, k, rng))):
                for _ in range(args.trials):
                    idx = fn()
                    call_bad = int(np.mean([c["p"][i] for i in idx]) > 0.5)
                    acc[pol].append(int(call_bad == ref_bad))
                    if sum(c["bad"]):
                        rec[pol].append(sum(c["bad"][i] for i in idx) / sum(c["bad"]))
        res[k] = {p: dict(chip_acc=float(np.mean(acc[p])), bad_recall=float(np.mean(rec[p]))) for p in acc}
        print(f"k={k}: " + "  ".join(f"{p} acc {res[k][p]['chip_acc']:.3f} / recall {res[k][p]['bad_recall']:.3f}"
                                     for p in ("random", "window", "adaptive")))
    (OUT / "policy_model_in_loop.json").write_text(json.dumps(res, indent=1))
    print("saved", OUT / "policy_model_in_loop.json")


if __name__ == "__main__":
    main()
