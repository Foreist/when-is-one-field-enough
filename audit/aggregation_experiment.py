# -*- coding: utf-8 -*-
"""Does the aggregation rule matter more than the per-field model?

Train one per-field model on a session-disjoint split, then compare chip-level
aggregation rules at a fixed imaging budget k (random fields per chip):
  mean      : mean of per-field P(bad) > 0.5
  max       : any field P(bad) > 0.5
  top2      : mean of the 2 highest field scores > 0.5
  conf      : any field P(bad) > 0.8
Chip reference = majority label over ALL fields of the session.
"""
import argparse, collections, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from leakage_experiment import (OOC, index_images, make_loaders, evaluate)
from leakage_controlled import split_controlled
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

OUT = Path(__file__).resolve().parent.parent / "results"


def train_predict(recs, args, seed, device):
    sp = split_controlled(recs, seed=1000 + seed, n_test_sessions=args.n_test_sessions)
    split = sp["disjoint"]                     # session-disjoint training
    loaders = make_loaders(split, args.size, args.batch, args.workers, seed)
    torch.manual_seed(seed)
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train()
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y); loss.backward(); opt.step()
    # per-field predictions on val (threshold pick) and test
    def predict(loader):
        model.eval(); probs, ys = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                q = torch.softmax(model(x), dim=1)[:, 0]      # P(bad) = class 0
                probs.append(q.cpu().numpy()); ys.append(y.numpy())
        return np.concatenate(probs), np.concatenate(ys)
    pv, yv = predict(loaders["val"])
    pt, yt = predict(loaders["test"])
    # pick the field threshold on val that maximises balanced accuracy (P(bad))
    thr, best = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        pred_bad = pv > t
        tpr = pred_bad[yv == 0].mean() if (yv == 0).any() else 0.0
        tnr = (~pred_bad[yv == 1]).mean() if (yv == 1).any() else 0.0
        if 0.5 * (tpr + tnr) > best:
            best, thr = 0.5 * (tpr + tnr), float(t)
    acc = float(((pt > thr) == (yt == 0)).mean())
    print(f"  [seed {seed}] field thr {thr:.2f} (val bal-acc {best:.3f})  test field acc {acc:.4f}  n={len(yt)}", flush=True)
    return dict(p=pt.tolist(), y=yt.tolist(), sessions=[r["session"] for r in split["test"]],
                field_acc=acc, thr=thr, meta=sp["meta"])


def simulate(pred, k, rules=("mean", "max", "top2"), trials=60, seed=0):
    """All rules use the SAME field threshold `pred['thr']`; scores are P(bad)."""
    rng = random.Random(seed)
    thr = pred["thr"]
    by_sess = collections.defaultdict(list)
    for pi, yi, s in zip(pred["p"], pred["y"], pred["sessions"]):
        by_sess[s].append((pi, yi))
    res = {r: dict(acc=[], sens=[], spec=[]) for r in rules}
    for s, v in by_sess.items():
        if len(v) < k:
            continue
        y = np.array([t[1] for t in v])              # y=1 good, 0 bad
        ref_bad = 1 if y.mean() < 0.5 else 0         # chip reference = majority bad
        for _ in range(trials):
            idx = rng.sample(range(len(v)), k)
            sc = np.array([v[i][0] for i in idx])    # P(bad) per sampled field
            calls = {
                "mean": int(sc.mean() > thr),
                "max": int(sc.max() > thr),
                "top2": int(np.sort(sc)[-min(2, k):].mean() > thr),
            }
            for r, c in calls.items():
                res[r]["acc"].append(int(c == ref_bad))
                if ref_bad:
                    res[r]["sens"].append(int(c == 1))
                else:
                    res[r]["spec"].append(int(c == 0))
    out = {}
    for r in rules:
        out[r] = dict(acc=float(np.mean(res[r]["acc"])),
                      sens=float(np.mean(res[r]["sens"])) if res[r]["sens"] else float("nan"),
                      spec=float(np.mean(res[r]["spec"])) if res[r]["spec"] else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-test-sessions", type=int, default=25)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = index_images()
    all_sim = {}
    preds = []
    for seed in range(args.seeds):
        pred = train_predict(recs, args, seed, device)
        preds.append({k: pred[k] for k in ("field_acc", "thr", "meta")})   # thr: chosen on validation
        for k in [1, 3, 5, 8, 12]:
            sim = simulate(pred, k, seed=seed)
            all_sim.setdefault(k, []).append(sim)
            print(f"    k={k}: " + "  ".join(f"{r} acc{sim[r]['acc']:.3f}/sens{sim[r]['sens']:.3f}"
                                             f"/spec{sim[r]['spec']:.3f}" for r in ("mean", "max", "top2")),
                  flush=True)
    summary = {}
    for k, sims in all_sim.items():
        summary[str(k)] = {r: dict(acc=float(np.mean([s[r]["acc"] for s in sims])),
                                   sens=float(np.mean([s[r]["sens"] for s in sims])),
                                   spec=float(np.mean([s[r]["spec"] for s in sims]))
                                   ) for r in sims[0]}
    (OUT / "aggregation_results.json").write_text(json.dumps(
        dict(summary=summary, runs=preds, args=vars(args)), ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("saved", OUT / "aggregation_results.json")


if __name__ == "__main__":
    main()
