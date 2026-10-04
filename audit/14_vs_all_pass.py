# -*- coding: utf-8 -*-
"""Is the chip rule better than calling every chip *pass*? Paired exact McNemar test.

Both the rule (spread fields, conf 0.9, max 20, every chip called) and the constant call are scored
on the same chips, so the comparison is paired: only chips where exactly one of them is right count.
  validation: 68 halves from 34 sessions; descriptive discordances only (clustered)
  Do not attach the iid McNemar p to repeated session halves.
  test:       the 25 held-out chips (perfield_preds_384_s0.json)
Writes results/vs_all_pass.json
"""
import collections, json, math, sys
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


def chips_from_prediction_arrays(pr):
    """Validate and group the saved per-field prediction arrays."""
    names = ("p", "y", "session", "idx")
    if not isinstance(pr, dict):
        raise ValueError("saved predictions must be an object of arrays")
    missing = [name for name in names if name not in pr]
    if missing:
        raise ValueError(f"missing prediction arrays: {', '.join(missing)}")
    for name in names:
        if not isinstance(pr[name], list):
            raise ValueError(f"{name} must be a prediction array")
    lengths = {name: len(pr[name]) for name in names}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"prediction arrays must have equal lengths: {lengths}")
    if not lengths["p"]:
        raise ValueError("prediction arrays must not be empty")
    by = collections.defaultdict(list)
    seen = set()
    for pos, (p, y, session, idx) in enumerate(zip(*(pr[name] for name in names))):
        if (isinstance(p, bool) or not isinstance(p, (int, float))
                or not 0 <= p <= 1 or not math.isfinite(p)):
            raise ValueError(f"p[{pos}] must be a finite number in [0, 1]")
        if isinstance(y, bool) or y not in (0, 1):
            raise ValueError(f"y[{pos}] must be 0 or 1")
        if not isinstance(session, str) or not session.strip():
            raise ValueError(f"session[{pos}] must be a non-empty string")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
            raise ValueError(f"idx[{pos}] must be a non-negative integer")
        key = (session, idx)
        if key in seen:
            raise ValueError(f"duplicate field index {idx} in session {session!r}")
        seen.add(key)
        by[session].append((idx, float(p), int(y)))
    return [dict(session=s, probs=[p for _, p, _ in sorted(v)],
                 y=[y for _, _, y in sorted(v)]) for s, v in by.items()]


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
    test = chips_from_prediction_arrays(pr)
    res = {f"{name}_min{m}": compare(ch, m) for name, ch in (("validation", val), ("test", test))
           for m in (1, 8)}
    for k, v in res.items():
        p_text = "not tested (session-clustered)" if v['mcnemar_p'] is None else f"{v['mcnemar_p']:.3f}"
        print(f"{k:<16} rule {v['rule_acc']:.3f}  all-pass {v['all_pass_acc']:.3f}  "
              f"discordant {v['only_rule_right']}/{v['only_all_pass_right']}  p {p_text}")
    (OUT / "vs_all_pass.json").write_text(json.dumps(res, indent=1, allow_nan=False))
    print("saved", OUT / "vs_all_pass.json")


if __name__ == "__main__":
    main()
