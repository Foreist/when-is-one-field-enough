"""
Does smoothing per-field scores along the acquisition order help the chip call?  (REPORT §4.6)

Uses the shipped session-disjoint predictions (results/perfield_preds_384_s0.json, 25 held-out chips).
For each chip, P(bad) is put in acquisition order and optionally replaced by a centred moving
average of width w (edge-padded).  k fields are then drawn at random (the SAME draws for every w),
their scores averaged, and the chip called bad if the mean exceeds 0.5.  Chips with fewer than k
fields are skipped.  Reference = majority label of the chip.

Output: results/smoothing_test.json
"""
import collections, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
WIDTHS = (1, 3, 5)          # 1 = no smoothing
TRIALS = 200


def smooth(pb, w):
    if w == 1:
        return pb
    return np.convolve(np.pad(pb, (w // 2, w // 2), mode="edge"), np.ones(w) / w, mode="valid")


def main():
    t = json.load(open(OUT / "perfield_preds_384_s0.json"))["test"]
    chips = collections.defaultdict(list)
    for p, y, s, i in zip(t["p"], t["y"], t["session"], t["idx"]):
        chips[s].append((i, p, y))           # p = P(bad), y = 1 good
    res = {}
    for k in (4, 6, 8, 12):
        rng = np.random.default_rng(k)
        acc = {w: [] for w in WIDTHS}
        n_chips = 0
        for s in sorted(chips):
            v = sorted(chips[s])
            if len(v) < k:
                continue
            n_chips += 1
            pb = np.array([x[1] for x in v]); ref = int(np.mean([x[2] for x in v]) < 0.5)
            sm = {w: smooth(pb, w) for w in WIDTHS}
            for _ in range(TRIALS):
                idx = rng.choice(len(pb), k, replace=False)
                for w in WIDTHS:
                    acc[w].append(int(int(sm[w][idx].mean() > 0.5) == ref))
        res[str(k)] = dict(n_chips=n_chips, **{f"w{w}": float(np.mean(acc[w])) for w in WIDTHS})
        print(f"k={k:>2} ({n_chips} chips): " + "  ".join(f"w={w} {res[str(k)][f'w{w}']:.3f}" for w in WIDTHS))
    (OUT / "smoothing_test.json").write_text(json.dumps(res, indent=1))
    print("saved", OUT / "smoothing_test.json")


if __name__ == "__main__":
    main()
