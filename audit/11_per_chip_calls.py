#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-chip decisions of the shipped rule on the 25 unseen test chips (REPORT Table 7, §6.2(a)).

For every chip, in imaging order: the shipped call (spread fields, min 8), the call when reading
the first fields instead (min 1, the bug in §6.2(a)), per-field accuracy and the mean P(bad) over
the first five and the last twenty fields; and, on chips with at least 20 fields, the mean-of-20
call from the first 20 fields versus 20 spread fields (§4.3).

Writes results/per_chip_calls.json
"""
import collections, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import TF, beta_p_bad, load_model, sequential_decision, spread_order   # noqa: E402
from leakage_experiment import index_images                              # noqa: E402
from leakage_controlled import split_controlled                           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def first_k(probs, conf=0.9, max_fields=20):
    bad = good = 0
    for i, p in enumerate(probs[:max_fields], 1):
        bad += p > 0.5
        good += p <= 0.5
        pb = beta_p_bad(bad, good)
        if pb > conf or 1 - pb > conf:
            return dict(n_fields=i, call="fail" if pb > 0.5 else "pass", confidence=max(pb, 1 - pb))
    return dict(n_fields=min(len(probs), max_fields), call="fail" if pb > 0.5 else "pass",
                confidence=max(pb, 1 - pb))


def main():
    sp = split_controlled(index_images(), seed=1000, n_test_sessions=25)
    by = collections.defaultdict(list)
    for r in sp["disjoint"]["test"]:
        by[r["session"]].append(r)
    model = load_model(str(ROOT / "model" / "perfield_mnv3s_384_s0.pt"))
    out = {}
    with torch.no_grad():
        for s, v in sorted(by.items()):
            v = sorted(v, key=lambda r: int(r["name"].split("_")[1].split(".")[0]))
            p = []
            for i in range(0, len(v), 32):
                x = torch.stack([TF(Image.open(r["path"]).convert("RGB")) for r in v[i:i + 32]])
                p += torch.softmax(model(x), dim=1)[:, 0].tolist()
            y_bad = [r["cls"] != "good" for r in v]
            d = sequential_decision(p)
            call = d["call"]
            if not d["stopped"] and 0.35 < d["p_bad"] < 0.65:
                call = "inconclusive"
            truth = "fail" if np.mean(y_bad) > 0.5 else "pass"
            out[s] = dict(n_fields=len(v), bad_share=float(np.mean(y_bad)),
                          field_acc=float(np.mean([(q > 0.5) == b for q, b in zip(p, y_bad)])),
                          truth=truth, call=call, confidence=max(d["p_bad"], 1 - d["p_bad"]),
                          fields_used=d["n_fields"], correct=call == truth if call != "inconclusive" else None,
                          first_k=first_k(p), mean_p_bad_first5=float(np.mean(p[:5])),
                          mean_p_bad_last20=float(np.mean(p[-20:])),
                          # fixed budget of 20, averaged: first 20 fields vs 20 spread fields
                          k20_first=(int(np.mean(p[:20]) > 0.5) == int(truth == "fail")) if len(p) >= 20 else None,
                          k20_spread=(int(np.mean([p[i] for i in spread_order(len(p), 20)]) > 0.5)
                                      == int(truth == "fail")) if len(p) >= 20 else None)
            print(s, call, truth, round(out[s]["confidence"], 3), flush=True)
    k20 = [c for c in out.values() if c["k20_first"] is not None]
    wrong = sorted(s for s, c in out.items() if c["correct"] is False)
    (ROOT / "results" / "per_chip_calls.json").write_text(json.dumps(dict(
        k20=dict(n_chips=len(k20), first_acc=float(np.mean([c["k20_first"] for c in k20])),
                 spread_acc=float(np.mean([c["k20_spread"] for c in k20]))),
        wrong=wrong, inconclusive=sorted(s for s, c in out.items() if c["call"] == "inconclusive"),
        per_session=out), indent=1))
    print("wrong:", wrong)


if __name__ == "__main__":
    main()
