#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness: the tool on the FULL held-out sessions.

The 25 test chips used everywhere else are the withheld half of each held-out session
(split_controlled(seed=1000, n_test_sessions=25) keeps the other half aside for the leakage
A/B design; it is never used for training). This script scores the same model on all fields of
those 25 sessions (still unseen: none of them is in train/val) and repeats the efficiency table,
with the reference re-defined as the full-session majority label.

Writes results/full_sessions.json
"""
import collections, json, math, re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import TF, beta_p_bad, load_model, spread_order      # noqa: E402
from leakage_experiment import index_images                          # noqa: E402
from leakage_controlled import split_controlled                       # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def main():
    recs = index_images()
    sp = split_controlled(recs, seed=1000, n_test_sessions=25)
    half = {r["name"] for r in sp["disjoint"]["test"]}
    T = {r["session"] for r in sp["disjoint"]["test"]}
    seen = {r["name"] for r in sp["disjoint"]["train"] + sp["disjoint"]["val"]}
    by = collections.defaultdict(list)
    for r in recs:
        if r["session"] in T:
            assert r["name"] not in seen
            by[r["session"]].append(r)
    model = load_model(str(Path(__file__).resolve().parent.parent / "model" / "perfield_mnv3s_384_s0.pt"))

    chips = {}
    with torch.no_grad():
        for s, v in by.items():
            v = sorted(v, key=lambda r: int(re.search(r"_(\d+)\.png$", r["name"]).group(1)))
            probs = []
            for i in range(0, len(v), 32):
                b = torch.stack([TF(Image.open(r["path"]).convert("RGB")) for r in v[i:i + 32]])
                probs += torch.softmax(model(b), dim=1)[:, 0].tolist()
            y = [1 if r["cls"] == "good" else 0 for r in v]
            yh = [yy for yy, r in zip(y, v) if r["name"] in half]
            chips[s] = dict(p=probs, ref=int(np.mean(y) < 0.5), ref_half=int(np.mean(yh) < 0.5))
    n = len(chips)
    nf = [len(c["p"]) for c in chips.values()]
    print(f"chips {n}  fields {sum(nf)}  mean {np.mean(nf):.1f}  range {min(nf)}-{max(nf)}")

    acc_all = float(np.mean([int(int(np.mean(c["p"]) > 0.5) == c["ref"]) for c in chips.values()]))
    cap = {}
    for k in [8, 12, 20]:
        a = [int(int(np.mean([c["p"][i] for i in spread_order(len(c["p"]), k)]) > 0.5) == c["ref"])
             for c in chips.values()]
        cap[k] = dict(acc=float(np.mean(a)), fields=float(np.mean([min(len(c["p"]), k) for c in chips.values()])))

    seq = {}
    for min_f in [8, 10, 12]:
        for mode in ("abstain", "force"):
            ok, used, abst, wrong_conf = [], [], 0, 0
            for c in chips.values():
                order = spread_order(len(c["p"]), 20)
                bad = good = 0; call = None; used_k = len(order); conf = False
                for i, idx in enumerate(order, 1):
                    bad += int(c["p"][idx] > 0.5); good += int(c["p"][idx] <= 0.5)
                    pb = beta_p_bad(bad, good)
                    if i >= min_f and (pb > 0.9 or 1 - pb > 0.9):
                        call = int(pb > 0.5); used_k = i; conf = True; break
                used.append(used_k)
                if call is None:
                    pb = beta_p_bad(bad, good)
                    if mode == "abstain" and 0.35 < pb < 0.65:
                        abst += 1; continue
                    call = int(pb > 0.5)
                ok.append(int(call == c["ref"]))
                wrong_conf += int(conf and call != c["ref"])
            key = f"{mode}_min{min_f}"
            seq[key] = dict(acc=float(np.mean(ok)), n_called=len(ok), abstained=abst,
                            fields=float(np.mean(used)), confident_wrong=wrong_conf,
                            wilson95=wilson(sum(ok), len(ok)))
            print(key, seq[key])

    out = dict(n_chips=n, total_fields=int(sum(nf)), mean_fields_per_chip=float(np.mean(nf)),
               fields_range=[int(min(nf)), int(max(nf))],
               reference_changes_vs_half=int(sum(c["ref"] != c["ref_half"] for c in chips.values())),
               all_fields_acc=acc_all, cap_k=cap, sequential=seq,
               note="same model and rule as the headline; chips = all fields of the 25 held-out "
                    "sessions (none used in training); reference = full-session majority label")
    print(json.dumps({k: out[k] for k in ("all_fields_acc", "cap_k", "reference_changes_vs_half")}))
    (OUT / "full_sessions.json").write_text(json.dumps(out, indent=1))
    print("saved", OUT / "full_sessions.json")


if __name__ == "__main__":
    main()
