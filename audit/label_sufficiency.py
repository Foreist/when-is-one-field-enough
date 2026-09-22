# -*- coding: utf-8 -*-
"""How many fields (images) does an expert-style call need to be stable?

Label-only analysis (no model):
 - per-session good/bad composition
 - agreement of an m-image majority vote with the full-session majority, for m = 1..10
 - agreement of a single image with the full-session majority (= how field-dependent the call is)
Stratified by cell line and by day bucket.
"""
import collections, json, random, re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "OOC_image_dataset"
OUT = Path(__file__).resolve().parent.parent / "results"
PAT = re.compile(r"^(\d{6})_(\d+)\.png$")


def load():
    recs = []
    for p in DATA.rglob("*.png"):
        parts = p.relative_to(DATA).parts
        if len(parts) != 5:
            continue
        m = PAT.match(parts[4])
        if not m:
            continue
        recs.append(dict(cls=parts[1], cell=parts[2], day=parts[3], session=m.group(1)))
    return recs


def main():
    recs = load()
    by_sess = collections.defaultdict(list)
    for r in recs:
        by_sess[r["session"]].append(r)

    comp = {}
    minority = []
    for s, v in sorted(by_sess.items()):
        c = collections.Counter(r["cls"] for r in v)
        n = len(v)
        minority.append(min(c.values()) / n)
        comp[s] = dict(n=n, good=c.get("good", 0), bad=c.get("bad", 0),
                       good_frac=round(c.get("good", 0) / n, 3))
    minority = np.array(minority)
    mixed = [s for s, v in comp.items() if v["good"] and v["bad"]]

    # sufficiency curve: m images -> majority vote vs full-session majority
    rng = random.Random(20260922)
    curve = {}
    for m in range(1, 11):
        agree = []
        for s, v in by_sess.items():
            labels = [1 if r["cls"] == "good" else 0 for r in v]
            full_maj = int(round(np.mean(labels))) if len(labels) else 0
            if len(labels) < m:
                continue
            trials = min(50, 200 // max(1, m))
            for _ in range(trials):
                sample = rng.sample(labels, m)
                s = sum(sample) * 2
                if s > m:
                    maj = 1
                elif s < m:
                    maj = 0
                else:                       # tie -> random, never consult the answer
                    maj = rng.randint(0, 1)
                agree.append(int(maj == full_maj))
        curve[m] = dict(n=len(agree), agree=float(np.mean(agree)) if agree else float("nan"))

    # per-cell and per-day single-image agreement with session majority
    def strat(key):
        out = collections.defaultdict(lambda: [0, 0])
        for s, v in by_sess.items():
            labels = [1 if r["cls"] == "good" else 0 for r in v]
            full_maj = int(round(np.mean(labels)))
            for r, y in zip(v, labels):
                k = r[key]
                out[k][0] += int(y == full_maj)
                out[k][1] += 1
        return {k: dict(n=n, agree=round(a / n, 3)) for k, (a, n) in sorted(out.items())}

    res = dict(
        n_images=len(recs), n_sessions=len(by_sess),
        sessions_mixed=len(mixed),
        minority_share_mean=float(minority.mean()),
        minority_share_mixed_mean=float(np.mean([min(v["good"], v["bad"]) / v["n"]
                                                 for s, v in comp.items() if s in set(mixed)])),
        session_composition=comp,
        sufficiency_curve=curve,
        single_image_agreement_by_cell=strat("cell"),
        single_image_agreement_by_day=strat("day"),
    )
    (OUT / "label_sufficiency.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"images {res['n_images']}  sessions {res['n_sessions']}  mixed {res['sessions_mixed']}")
    print(f"minority share: all {res['minority_share_mean']:.3f}  mixed-only {res['minority_share_mixed_mean']:.3f}")
    print("m-image majority vs full-session majority:")
    for m, v in curve.items():
        print(f"  m={m:>2}  agree {v['agree']:.3f}  (n={v['n']})")
    print("single-image agreement by cell:", json.dumps(res["single_image_agreement_by_cell"], ensure_ascii=False))
    print("single-image agreement by day :", json.dumps(res["single_image_agreement_by_day"], ensure_ascii=False))
    print("saved", OUT / "label_sufficiency.json")


if __name__ == "__main__":
    main()
