"""
Like-for-like version of the label-only vs model-in-the-loop comparison (REPORT §6.7).

`stopping_rule.py` simulates a naive rule on ground-truth labels (random field order, no minimum,
all 59 sessions).  That differs from the deployed tool in three ways besides the model.  Here the
DEPLOYED rule (spread order, conf 0.9, max 20, inconclusive band 0.35-0.65) is run on the SAME 25
held-out chips twice: once with ground-truth field labels as the per-field call, once with the
shipped model's calls.  Both with min_fields = 8 (shipped) and min_fields = 1 (no minimum).

Output: results/label_vs_model.json
"""
import collections, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import spread_order, beta_p_bad      # noqa: E402

OUT = ROOT / "results"


def run(chips, source, min_f, conf=0.9, max_fields=20):
    acc, used, wrong_conf, inconclusive = [], [], [], 0
    for s in sorted(chips):
        v = chips[s]
        y = np.array([x[2] for x in v]); ref_bad = int((y == 0).mean() > 0.5)
        score = [x[1] for x in v] if source == "model" else [1.0 - x[2] for x in v]   # P(bad)
        bad = good = 0; call = None
        for i, idx in enumerate(spread_order(len(score), max_fields), 1):
            if score[idx] > 0.5: bad += 1
            else: good += 1
            pb = beta_p_bad(bad, good)
            if i >= min_f and max(pb, 1 - pb) > conf:
                call = int(pb > 0.5); c = max(pb, 1 - pb); used.append(i)
                break
        if call is None:
            pb = beta_p_bad(bad, good); c = max(pb, 1 - pb); used.append(i)
            if 0.35 < pb < 0.65:
                inconclusive += 1
                continue
            call = int(pb > 0.5)
        acc.append(int(call == ref_bad))
        wrong_conf.append(int(c >= conf and call != ref_bad))
    return dict(mean_fields=float(np.mean(used)), n_called=len(acc), n_correct=int(sum(acc)),
                accuracy=float(np.mean(acc)), n_false_confident=int(sum(wrong_conf)),
                n_inconclusive=inconclusive, n_chips=len(chips))


def main():
    t = json.load(open(OUT / "perfield_preds_384_s0.json"))["test"]
    chips = collections.defaultdict(list)
    for p, y, s, i in zip(t["p"], t["y"], t["session"], t["idx"]):
        chips[s].append((i, p, y))
    for s in chips:
        chips[s].sort()
    res = {}
    for min_f in (8, 1):
        for src in ("labels", "model"):
            r = run(chips, src, min_f)
            res[f"min{min_f}_{src}"] = r
            print(f"min_fields={min_f} {src:<6}: {r['mean_fields']:.1f} fields, "
                  f"{r['n_correct']}/{r['n_called']} = {r['accuracy']:.3f}, "
                  f"false-confident {r['n_false_confident']}, inconclusive {r['n_inconclusive']}")
    (OUT / "label_vs_model.json").write_text(json.dumps(res, indent=1))
    print("saved", OUT / "label_vs_model.json")


if __name__ == "__main__":
    main()
