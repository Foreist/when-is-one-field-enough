#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leave-one-cell-line-out: does the per-field model work on a cell line it has never seen?

The session-disjoint split used elsewhere keeps all six cell lines in training. A reviewer can
ask what happens on a *new* cell line. For each cell line X:
  test  = all fields of cell line X
  train = fields of the other five cell lines, excluding any session that also contains X
          (otherwise the same chip would appear on both sides)
Model: the deployed configuration (MobileNetV3-small, 384 px, 25 epochs).

Writes results/lolo_cellline.json
"""
import argparse, collections, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from leakage_experiment import OOC, index_images, PAT, make_loaders  # noqa: E402
from perfield_model import build, predict, metrics                     # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
CELLS = ["cell_type_A549", "cell_type_CACO", "cell_type_HPMEC",
         "cell_type_HUVEC", "cell_type_NHBE", "cell_type_HSAEC"]


def fold(recs, cell):
    test = [r for r in recs if r["cell"] == cell]
    bad_sessions = {r["session"] for r in test}
    train = [r for r in recs if r["cell"] != cell and r["session"] not in bad_sessions]
    return dict(train=train, val=train[-200:] if len(train) > 400 else train, test=test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    recs = index_images()
    res = {}
    for cell in CELLS:
        sp = fold(recs, cell)
        if len(sp["test"]) < 20 or len(sp["train"]) < 300:
            res[cell] = dict(skipped=True, n_test=len(sp["test"]), n_train=len(sp["train"]))
            print(f"{cell}: skipped (train {len(sp['train'])}, test {len(sp['test'])})", flush=True)
            continue
        loaders = make_loaders(sp, args.size, args.batch, args.workers, args.seed)
        torch.manual_seed(args.seed)
        model = build("small").to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.02)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)
        t0 = time.time()
        for ep in range(args.epochs):
            model.train()
            for x, y in loaders["train"]:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y); loss.backward(); opt.step()
            sched.step()
        p, y = predict(model, loaders["test"], device)
        m = metrics(p, y)
        res[cell] = dict(n_train=len(sp["train"]), n_test=len(sp["test"]),
                         n_sessions_test=len({r["session"] for r in sp["test"]}),
                         acc=m["acc"], bal_acc=m["bal_acc"], auc=m["auc"],
                         secs=round(time.time() - t0, 1))
        print(f"{cell:<20} test n={len(sp['test']):>4} ({res[cell]['n_sessions_test']} sessions)  "
              f"acc {m['acc']:.3f}  bal {m['bal_acc']:.3f}  auc {m['auc']:.3f}  ({res[cell]['secs']:.0f}s)",
              flush=True)
    done = [v for v in res.values() if not v.get("skipped")]
    summary = dict(per_cell=res,
                   mean_acc=float(np.mean([v["acc"] for v in done])),
                   mean_auc=float(np.mean([v["auc"] for v in done])),
                   note="test = all fields of the held-out cell line; sessions containing it are "
                        "excluded from training")
    (OUT / "lolo_cellline.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\nmean acc {summary['mean_acc']:.3f}  mean auc {summary['mean_auc']:.3f}")
    print("saved", OUT / "lolo_cellline.json")


if __name__ == "__main__":
    main()
