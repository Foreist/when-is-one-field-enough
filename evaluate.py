#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Honest evaluation of the deployed tool on the SESSION-DISJOINT test chips.

Reproduces:
  1. per-field accuracy / AUC on 25 unseen chips
  2. chip-level decisions: all-field mean, a cap of k spread fields, and sequential stopping
     for each minimum-fields guard, including the FALSE-CONFIDENCE rate and the Wilson
     interval for the shipped setting
(split leakage is audit/leakage_experiment.py and audit/leakage_controlled.py)

Outputs results/tool_evaluation.json

Units are withheld half-session proxies, not physical-chip-validated; this script
does not claim an accuracy or imaging-time guarantee. Every test field is scored
for the per-field metrics / QC-style maps; sequential fields_used is rule consumption
(resource_basis), not measured time savings.
"""
import argparse, collections, json, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import mobilenet_v3_small

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "audit"))
from inference import TF, spread_order, sequential_decision, RESOURCE_BASIS  # noqa: E402


def load_test_chips(data_root=None):
    """Rebuild the session-disjoint split used in training (seed 1000, 25 test sessions).

    Honors `data_root` for this call only. Does not write OOC_DATA or leave
    leakage_experiment.DATA pointing at the supplied root.
    """
    import leakage_experiment as le
    from leakage_controlled import split_controlled
    recs_root = Path(data_root) if data_root is not None else Path(
        os.environ.get("OOC_DATA", str(HERE.parent / "data" / "OOC_image_dataset"))
    )
    prev = le.DATA
    try:
        le.DATA = recs_root
        recs = le.index_images()
    finally:
        le.DATA = prev
    sp = split_controlled(recs, seed=1000, n_test_sessions=25)
    by = collections.defaultdict(list)
    for r in sp["disjoint"]["test"]:
        by[r["session"]].append(r)
    import re as _re
    for s in by:
        by[s].sort(key=lambda r: int(_re.search(r"_(\d+)\.png$", r["name"]).group(1)))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get(
                        "OOC_DATA", str(HERE.parent / "data" / "OOC_image_dataset")),
                    help="image root (default: $OOC_DATA or ../data/OOC_image_dataset); "
                         "used for this run only, does not mutate OOC_DATA")
    ap.add_argument("--checkpoint", default=str(HERE / "model" / "perfield_mnv3s_384_s0.pt"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-fields", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.9)
    args = ap.parse_args()
    if args.max_fields < 2:
        raise SystemExit("max_fields must be >= 2")
    if not (0.5 < float(args.conf) < 1.0):
        raise SystemExit("conf must be in (0.5, 1)")

    by = load_test_chips(args.data_root)
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    chips = {}
    with torch.no_grad():
        for s, v in by.items():
            probs = []
            for i in range(0, len(v), args.batch):
                batch = torch.stack([TF(Image.open(r["path"]).convert("RGB")) for r in v[i:i + args.batch]])
                probs += torch.softmax(model(batch), dim=1)[:, 0].tolist()
            chips[s] = dict(probs=probs, y=[1 if r["cls"] == "good" else 0 for r in v])

    # ---- per-field metrics ----
    P = np.concatenate([np.array(c["probs"]) for c in chips.values()])
    Y = np.concatenate([np.array(c["y"]) for c in chips.values()])
    field_acc = float(((P > 0.5) == (Y == 0)).mean())
    yy = (Y == 0).astype(int)
    from scipy.stats import rankdata
    ranks = rankdata(P)                                  # average ranks: ties count half
    npos, nneg = int(yy.sum()), int((1 - yy).sum())
    auc = float((ranks[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

    # ---- chip-level: all fields, mean ----
    rows = []
    for s, c in chips.items():
        p = np.array(c["probs"]); y = np.array(c["y"])
        ref_bad = int((y == 0).mean() > 0.5)
        call_bad = int(p.mean() > 0.5)
        rows.append((s, ref_bad, call_bad))
    allfield_acc = float(np.mean([r[1] == r[2] for r in rows]))

    # ---- chip-level: sequential stopping, min-fields sweep ----
    # same metric definitions as before; sequential_decision is the shared rule
    seq = {}
    for min_f in [1, 3, 5, 8, 10, 12]:
        acc, wrong_conf, used, stopped = [], [], [], []
        outcomes = collections.Counter()
        for s, c in chips.items():
            probs = c["probs"]; y = np.array(c["y"])
            ref_bad = int((y == 0).mean() > 0.5)
            dec = sequential_decision(probs, thr_conf=args.conf, max_fields=args.max_fields,
                                      min_fields=min_f)
            used.append(dec["n_fields"])
            stopped.append(1 if dec["stopped"] else 0)
            if dec["call"] == "inconclusive":
                outcomes["inconclusive"] += 1
                continue
            call = int(dec["call"] == "fail")
            conf = max(dec["p_bad"], 1 - dec["p_bad"])
            acc.append(int(call == ref_bad))
            outcomes["fail" if call == 1 else "pass"] += 1
            if conf >= args.conf and call != ref_bad:
                wrong_conf.append(1)
            else:
                wrong_conf.append(0)
        n_conf = len(acc)
        seq[min_f] = dict(n_confident=n_conf,
                          chip_acc_among_confident=float(np.mean(acc)) if n_conf else float("nan"),
                          mean_fields=float(np.mean(used)),
                          frac_stopped=float(np.mean(stopped)),
                          false_confident_rate=float(np.mean(wrong_conf)) if n_conf else float("nan"),
                          outcomes=dict(outcomes), n_chips=len(chips))

    # ---- chip-level: cap of k spread fields (all fields if the chip has fewer) ----
    fixed = {}
    for k in [3, 5, 8, 12, 20]:
        acc = []
        for s, c in chips.items():
            probs = c["probs"]; y = np.array(c["y"])
            ref_bad = int((y == 0).mean() > 0.5)
            idx = spread_order(len(probs), k)
            acc.append(int(int(np.mean([probs[i] for i in idx]) > 0.5) == ref_bad))
        fixed[k] = float(np.mean(acc))

    # Wilson 95% interval for the shipped rule's accuracy on the chips it calls
    m8 = seq[8]
    n8, k8 = m8["n_confident"], round(m8["chip_acc_among_confident"] * m8["n_confident"])
    z = 1.959963984540054
    ctr = (k8 + z * z / 2) / (n8 + z * z)
    half = z * np.sqrt(k8 * (n8 - k8) / n8 + z * z / 4) / (n8 + z * z)
    wilson = [round(float(ctr - half), 3), round(float(ctr + half), 3)]

    n_scored = int(len(P))
    out = dict(
        data_root=Path(args.data_root).name, checkpoint=Path(args.checkpoint).name,
        n_chips=len(chips), n_fields=n_scored,
        n_fields_scored=n_scored,
        resource_basis=RESOURCE_BASIS,
        field_accuracy=field_acc, field_auc=auc,
        chip_all_fields_mean_acc=allfield_acc,
        chip_sequential_by_min_fields=seq,
        chip_fixed_k_spread=fixed,
        note="sequential stopping uses a Beta(1,1) posterior on the bad fraction with "
             "evenly spread fields; min_fields guards against stopping on a lucky good region",
        chip_acc_among_confident_wilson95=wilson,
        note_ci=f"Wilson interval over {n8} confident calls ({len(chips) - n8} of {len(chips)} chips were inconclusive)",
    )
    (HERE / "results" / "tool_evaluation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("\nsaved results/tool_evaluation.json")


if __name__ == "__main__":
    main()
