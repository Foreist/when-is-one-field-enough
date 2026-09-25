# -*- coding: utf-8 -*-
"""Select the stopping-rule settings WITHOUT touching the 25 test chips.

The shipped minimum-fields guard (8) was read off a sweep on the test chips (evaluate.py).
This script re-selects it on the 34 NON-test sessions only:

  1. session-grouped 5-fold CV over the non-test sessions of split_controlled(seed 1000, 25);
     each fold trains the shipped recipe (MobileNetV3-small, 384 px, 25 epochs, AdamW 3e-4,
     wd 0.02, cosine, label smoothing 0.05) on the other folds and predicts its own sessions;
  2. every non-test session is cut into two random halves, each half one "chip"
     (the test chips are also half-sessions), giving 68 validation chips;
  3. the selection criterion is fixed before looking at the result:
     highest force-mode chip accuracy (every chip called), ties -> fewer mean fields,
     over min_fields in {1, 3, 5, 8, 10, 12} at conf 0.9, max 20;
  4. the same rule reading the FIRST fields instead of spread ones is reported alongside
     (the spread order was motivated by a test chip, so it is re-checked here);
  5. the same criterion picks the plain cap-of-k baseline (k in {8, 12, 20}), so the rule and the
     baseline it is compared with are both chosen without the test chips.

    python3 audit/09_inner_cv_minfields.py            # train folds + select (GPU, ~1 h)
    python3 audit/09_inner_cv_minfields.py --select   # reuse saved OOF predictions

Writes results/inner_cv_oof.json and results/inner_cv_minfields.json
"""
import argparse, collections, json, random, re, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
OUT = HERE.parent / "results"
MIN_GRID = [1, 3, 5, 8, 10, 12]
CAP_GRID = [8, 12, 20]


def folds_of(recs, k=5, seed=0):
    by = collections.defaultdict(list)
    for r in recs:
        by[r["session"]].append(r)
    sess = sorted(by)
    random.Random(seed).shuffle(sess)
    sess.sort(key=lambda s: -len(by[s]))              # greedy balance by field count
    load, groups = [0] * k, [[] for _ in range(k)]
    for s in sess:
        i = int(np.argmin(load)); groups[i].append(s); load[i] += len(by[s])
    return by, groups


def train_folds(args):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader
    from leakage_experiment import OOC, index_images, PAT
    from leakage_controlled import split_controlled
    from perfield_model import build, make_loaders, predict
    recs = index_images()
    sp = split_controlled(recs, seed=1000, n_test_sessions=25)["disjoint"]
    test_sess = {r["session"] for r in sp["test"]}
    nonT = [r for r in recs if r["session"] not in test_sess]
    by, groups = folds_of(nonT, k=args.folds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oof = {}
    for f, held in enumerate(groups):
        tr = [r for s in by if s not in held for r in by[s]]
        ho = [r for s in held for r in by[s]]
        loaders = make_loaders(dict(train=tr, test=ho), args.size, args.batch, args.workers, 0)
        torch.manual_seed(0)
        model = build("small").to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.02)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=3e-4 / 30)
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)
        t0 = time.time()
        for ep in range(args.epochs):
            model.train()
            for x, y in loaders["train"]:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                crit(model(x), y).backward(); opt.step()
            sched.step()
        p, y = predict(model, loaders["test"], device)
        acc = float(((p > 0.5) == (y == 0)).mean())
        print(f"[fold {f}] train {len(tr)} held {len(ho)} ({len(held)} sessions) "
              f"field acc {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        for r, pi, yi in zip(ho, p, y):
            oof.setdefault(r["session"], []).append(
                (int(PAT.match(r["name"]).group(2)), round(float(pi), 5), int(yi)))
    for s in oof:
        oof[s].sort()
    (OUT / "inner_cv_oof.json").write_text(json.dumps(dict(
        folds=[sorted(g) for g in groups], args=vars(args), oof=oof)))
    return oof


def chips_from(oof, seed=0):
    rng = random.Random(seed)
    chips = []
    for s in sorted(oof):
        v = oof[s][:]
        rng.shuffle(v)
        cut = len(v) // 2
        for part in (v[:cut], v[cut:]):
            part = sorted(part)                         # acquisition order within the chip
            if len(part) >= 2:
                chips.append(dict(session=s, probs=[p for _, p, _ in part], y=[y for _, _, y in part]))
    return chips


def select(oof):
    from inference import spread_order, beta_p_bad
    chips = chips_from(oof)
    res = run_rule(chips, spread_order, beta_p_bad, "spread")
    first = run_rule(chips, lambda n, k: list(range(min(n, k))), beta_p_bad, "first-k")
    return finish(chips, oof, res, first, spread_order)


def run_rule(chips, order_fn, beta_p_bad, tag):
    res = {}
    for min_f in MIN_GRID:
        force, called, wrongconf, used = [], [], [], []
        for c in chips:
            probs, y = c["probs"], np.array(c["y"])
            ref = int((y == 0).mean() > 0.5)
            order = order_fn(len(probs), 20)
            bad = good = 0; stop = None
            for i, idx in enumerate(order, 1):
                bad += probs[idx] > 0.5; good += probs[idx] <= 0.5
                pb = beta_p_bad(bad, good)
                if i >= min_f and max(pb, 1 - pb) > 0.9:
                    stop = i; break
            used.append(stop or len(order))
            pb = beta_p_bad(bad, good); call = int(pb > 0.5)
            force.append(call == ref)
            if stop is not None or not (0.35 < pb < 0.65):
                called.append(call == ref)
                wrongconf.append(max(pb, 1 - pb) >= 0.9 and call != ref)
        res[min_f] = dict(force_acc=float(np.mean(force)), abstain_acc=float(np.mean(called)),
                          n_called=len(called), false_confident=int(np.sum(wrongconf)),
                          mean_fields=float(np.mean(used)))
        print(f"[{tag}] min {min_f:>2}: force {res[min_f]['force_acc']:.3f}  abstain "
              f"{res[min_f]['abstain_acc']:.3f} ({len(called)}/{len(chips)})  "
              f"wrong-conf {res[min_f]['false_confident']}  fields {res[min_f]['mean_fields']:.1f}")
    return res


def finish(chips, oof, res, first, spread_order):
    best = max(MIN_GRID, key=lambda m: (round(res[m]["force_acc"], 6), -res[m]["mean_fields"]))
    # the baseline the report compares against: a plain cap of k spread fields, mean P(bad) > 0.5
    cap = {}
    for k in CAP_GRID:
        acc, used = [], []
        for c in chips:
            idx = spread_order(len(c["probs"]), k)
            ref = int((np.array(c["y"]) == 0).mean() > 0.5)
            acc.append(int(np.mean([c["probs"][i] for i in idx]) > 0.5) == ref); used.append(len(idx))
        cap[k] = dict(force_acc=float(np.mean(acc)), mean_fields=float(np.mean(used)))
        print(f"cap {k:>2}: acc {cap[k]['force_acc']:.3f}  fields {cap[k]['mean_fields']:.1f}")
    all_acc = float(np.mean([int(np.mean(c["probs"]) > 0.5) == int((np.array(c["y"]) == 0).mean() > 0.5)
                             for c in chips]))
    all_fields = float(np.mean([len(c["probs"]) for c in chips]))
    print(f"all fields: acc {all_acc:.3f}  fields {all_fields:.1f}")
    best_cap = max(CAP_GRID, key=lambda k: (round(cap[k]["force_acc"], 6), -cap[k]["mean_fields"]))
    out = dict(n_chips=len(chips), n_sessions=len(oof), criterion=(
        "max force-mode chip accuracy, ties -> fewer mean fields; conf 0.9, max 20"),
        by_min_fields=res, selected_min_fields=best, shipped_min_fields=8,
        by_cap=cap, selected_cap=best_cap, all_fields=dict(force_acc=all_acc, mean_fields=all_fields), first_k_by_min_fields=first)
    (OUT / "inner_cv_minfields.json").write_text(json.dumps(out, indent=1))
    print("selected min_fields =", best, " selected cap =", best_cap)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true", help="skip training, reuse inner_cv_oof.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.select:
        oof = json.loads((OUT / "inner_cv_oof.json").read_text())["oof"]
    else:
        oof = train_folds(args)
    select(oof)


if __name__ == "__main__":
    main()
