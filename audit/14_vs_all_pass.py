# -*- coding: utf-8 -*-
"""Is the chip rule better than calling every chip *pass*? Paired exact McNemar test.

Both the rule (spread fields, conf 0.9, max 20, every chip called) and the constant call are scored
on the same chips, so the comparison is paired: only chips where exactly one of them is right count.
  validation: 68 halves from 34 sessions; descriptive discordances only (clustered)
  Do not attach the iid McNemar p to repeated session halves.
  test:       the 25 held-out chips (perfield_preds_384_s0.json)
Writes results/vs_all_pass.json
"""
import collections, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
OUT = HERE.parent / "results"
from inference import spread_order, beta_p_bad                    # noqa: E402
import importlib.util                                             # noqa: E402
_s = importlib.util.spec_from_file_location("icv", HERE / "09_inner_cv_minfields.py")
icv = importlib.util.module_from_spec(_s); _s.loader.exec_module(icv)


def call(probs, min_f):
    bad = good = 0
    for i, idx in enumerate(spread_order(len(probs), 20), 1):
        bad += probs[idx] > 0.5; good += probs[idx] <= 0.5
        pb = beta_p_bad(bad, good)
        if i >= min_f and max(pb, 1 - pb) > 0.9:
            break
    return int(beta_p_bad(bad, good) > 0.5)


def compare(chips, min_f):
    sessions = [c.get("session") for c in chips]
    clustered = any(s is None for s in sessions) or len(set(sessions)) != len(sessions)
    ref = np.array([int((np.array(c["y"]) == 0).mean() > 0.5) for c in chips])
    rule = np.array([call(c["probs"], min_f) for c in chips]) == ref
    allpass = ref == 0
    b, d = int((rule & ~allpass).sum()), int((~rule & allpass).sum())
    return dict(n_chips=len(chips), n_sessions=len(set(sessions)) if all(s is not None for s in sessions) else None,
                rule_acc=float(rule.mean()), all_pass_acc=float(allpass.mean()),
                only_rule_right=b, only_all_pass_right=d,
                mcnemar_p=None if clustered else (float(binomtest(b, b + d).pvalue) if b + d else 1.0),
                inference_status="descriptive_only" if clustered else "exact_paired_test",
                inference_reason="Repeated or unknown session IDs: independent discordant pairs are not established" if clustered else
                                 "One proxy per distinct session; inference assumes independence between sessions")


def main():
    val = icv.chips_from(json.loads((OUT / "inner_cv_oof.json").read_text())["oof"])
    pr = json.loads((OUT / "perfield_preds_384_s0.json").read_text())["test"]
    by = collections.defaultdict(list)
    for p, y, s, i in zip(pr["p"], pr["y"], pr["session"], pr["idx"]):
        by[s].append((i, p, y))
    test = [dict(session=s, probs=[p for _, p, _ in sorted(v)], y=[y for _, _, y in sorted(v)])
            for s, v in by.items()]
    res = {f"{name}_min{m}": compare(ch, m) for name, ch in (("validation", val), ("test", test))
           for m in (1, 8)}
    for k, v in res.items():
        p_text = "not tested (session-clustered)" if v['mcnemar_p'] is None else f"{v['mcnemar_p']:.3f}"
        print(f"{k:<16} rule {v['rule_acc']:.3f}  all-pass {v['all_pass_acc']:.3f}  "
              f"discordant {v['only_rule_right']}/{v['only_all_pass_right']}  p {p_text}")
    (OUT / "vs_all_pass.json").write_text(json.dumps(res, indent=1))
    print("saved", OUT / "vs_all_pass.json")


if __name__ == "__main__":
    main()
