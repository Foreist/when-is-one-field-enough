#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Field efficiency: how many fields does the chip decision need?

Baselines on the 25 session-disjoint test chips:
  all-fields   : average every field of the chip (the naive workflow; ~27 fields/chip here = half of each held-out session)
  fixed-k      : k evenly spread fields, mean of the scores
  sequential   : the shipped rule (spread fields, Beta posterior, stop at conf 0.9, min 8)

Reported: accuracy, abstention rate (inconclusive), fields used, and the budget at which the
sequential rule reaches the all-fields accuracy.

Writes results/efficiency.json
"""
import collections, json
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


def main():
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
                b = torch.stack([TF(Image.open(r["path"]).convert("RGB")) for r in v[i:i + 32]])
                probs += torch.softmax(model(b), dim=1)[:, 0].tolist()
            chips[s] = dict(p=probs, y=[1 if r["cls"] == "good" else 0 for r in v])
    n_chips = len(chips)
    total_fields = sum(len(c["p"]) for c in chips.values())
    print(f"chips {n_chips}  fields {total_fields}  mean fields/chip {total_fields/n_chips:.1f}")

    # ---- all fields ----
    acc_all = np.mean([int(int(np.mean(c["p"]) > 0.5) == int(np.mean(c["y"]) < 0.5))
                       for c in chips.values()])
    print(f"\nall-fields baseline: accuracy {acc_all:.3f} using {total_fields/n_chips:.1f} fields/chip")

    # ---- fixed k (spread) ----
    fixed = {}
    for k in [2, 4, 6, 8, 10, 12, 16, 20]:
        acc = []
        for c in chips.values():
            if len(c["p"]) < k:
                continue
            idx = spread_order(len(c["p"]), k)
            acc.append(int(int(np.mean([c["p"][i] for i in idx]) > 0.5) == int(np.mean(c["y"]) < 0.5)))
        fixed[k] = dict(acc=float(np.mean(acc)), n=len(acc), fields=k)
        print(f"  fixed k={k:>2}: accuracy {fixed[k]['acc']:.3f}  (chips {len(acc)})")

    # ---- cap of k spread fields on EVERY chip (all fields when the chip has fewer) ----
    cap = {}
    for k in [8, 12, 20]:
        acc = [int(int(np.mean([c["p"][i] for i in spread_order(len(c["p"]), k)]) > 0.5)
                   == int(np.mean(c["y"]) < 0.5)) for c in chips.values()]
        cap[k] = dict(acc=float(np.mean(acc)), n=len(acc),
                      fields=float(np.mean([min(len(c["p"]), k) for c in chips.values()])))
        print(f"  cap k={k:>2}: accuracy {cap[k]['acc']:.3f}  fields {cap[k]['fields']:.1f}")

    # ---- sequential (shipped rule), two modes ----
    seq = {}
    for min_f in [4, 6, 8, 12]:
        for mode in ("abstain", "force"):
            acc, abst, used = [], [], []
            for c in chips.values():
                order = spread_order(len(c["p"]), 20)
                bad = good = 0; call = None; used_k = len(order)
                for i, idx in enumerate(order, 1):
                    bad += int(c["p"][idx] > 0.5); good += int(c["p"][idx] <= 0.5)
                    pb = beta_p_bad(bad, good)
                    if i >= min_f and (pb > 0.9 or 1 - pb > 0.9):
                        call = int(pb > 0.5); used_k = i; break
                if call is None:
                    pb = beta_p_bad(bad, good)
                    if mode == "abstain" and 0.35 < pb < 0.65:
                        abst.append(1); used.append(used_k); continue
                    call = int(pb > 0.5)
                abst.append(0); used.append(used_k)
                acc.append(int(call == int(np.mean(c["y"]) < 0.5)))
            key = f"{mode}_min{min_f}"
            seq[key] = dict(mode=mode, min_fields=min_f,
                            acc=float(np.mean(acc)) if acc else float("nan"),
                            end_to_end_acc=float(np.sum(acc) / n_chips),
                            abstention=float(np.mean(abst)), fields=float(np.mean(used)),
                            n_called=len(acc))
            print(f"  sequential {mode:<7} min={min_f:>2}: acc {seq[key]['acc']:.3f} "
                  f"(called {len(acc)}/{n_chips})  abstain {seq[key]['abstention']:.2f}  "
                  f"fields {seq[key]['fields']:.1f}")

    out = dict(n_chips=n_chips, total_fields=total_fields, mean_fields_per_chip=total_fields / n_chips,
               all_fields_acc=float(acc_all), fixed_k=fixed, cap_k=cap, sequential=seq,
               note="mode=force calls every chip (no abstention); mode=abstain defers chips whose "
                    "posterior stays between 0.35 and 0.65, and acc is then conditional on the "
                    "chips that were called")
    (OUT / "efficiency.json").write_text(json.dumps(out, indent=1))
    print("\nsaved", OUT / "efficiency.json")


if __name__ == "__main__":
    main()
