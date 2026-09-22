# -*- coding: utf-8 -*-
"""Train a stronger per-field QC model on the session-disjoint protocol.

- split: leakage_controlled.split_controlled(disjoint arm) -> no session spans train/test
- model: MobileNetV3-small, 384px (or --size), cosine LR, label smoothing
- saves: checkpoint, per-field test predictions (with session + index) for the demo,
         and a metrics JSON (field acc/AUC/balanced acc, per-session acc).
"""
import argparse, collections, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

from leakage_experiment import OOC, index_images, PAT
from leakage_controlled import split_controlled

OUT = Path(__file__).resolve().parent


def make_loaders(split, size, bs, workers, seed):
    tr = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ev = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    g = torch.Generator().manual_seed(seed)
    return {k: DataLoader(OOC(v, tr if k == "train" else ev), batch_size=bs,
                          shuffle=(k == "train"), num_workers=workers, pin_memory=True,
                          generator=g, drop_last=(k == "train"))
            for k, v in split.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ps, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        p = torch.softmax(model(x), dim=1)[:, 0]      # P(bad)
        ps.append(p.cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def metrics(p, y, thr=0.5):
    pred_bad = p > thr
    acc = float((pred_bad == (y == 0)).mean())
    tpr = float(pred_bad[y == 0].mean()) if (y == 0).any() else float("nan")
    tnr = float((~pred_bad[y == 1]).mean()) if (y == 1).any() else float("nan")
    # AUC (score = P(bad), positive class = bad)
    yy = (y == 0).astype(int)
    order = np.argsort(p); ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = int(yy.sum()), int((1 - yy).sum())
    auc = float((ranks[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")
    return dict(acc=acc, bal_acc=float(np.nanmean([tpr, tnr])), auc=auc,
                sens=tpr, spec=float(tnr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test-sessions", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = index_images()
    sp = split_controlled(recs, seed=1000 + args.seed, n_test_sessions=args.n_test_sessions)
    split = sp["disjoint"]
    print("split meta:", sp["meta"], flush=True)
    loaders = make_loaders(split, args.size, args.batch, args.workers, args.seed)

    torch.manual_seed(args.seed)
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 30)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = -1.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; n = 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(y); n += len(y)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            pv, yv = predict(model, loaders["val"], device)
            mv = metrics(pv, yv)
            print(f"  ep{ep+1:>2} loss {tot/max(1,n):.4f} val acc {mv['acc']:.4f} "
                  f"bal {mv['bal_acc']:.4f} auc {mv['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            best = max(best, mv["bal_acc"])
    pt, yt = predict(model, loaders["test"], device)
    mt = metrics(pt, yt)
    sessions = [r["session"] for r in split["test"]]
    idxs = [int(PAT.match(r["name"]).group(2)) for r in split["test"]]
    print(f"  TEST field acc {mt['acc']:.4f} bal {mt['bal_acc']:.4f} auc {mt['auc']:.4f} n={len(yt)}", flush=True)
    per_sess = {}
    for s in sorted(set(sessions)):
        m = np.array([x == s for x in sessions])
        ps, ys = pt[m], yt[m]
        per_sess[s] = dict(n=int(m.sum()), bad_frac=float((ys == 0).mean()),
                           acc=float(((ps > 0.5) == (ys == 0)).mean()))
    torch.save(model.state_dict(), OUT / f"perfield_mnv3s_{args.size}_s{args.seed}.pt")
    (OUT / f"perfield_preds_{args.size}_s{args.seed}.json").write_text(json.dumps(dict(
        size=args.size, epochs=args.epochs, seed=args.seed, meta=sp["meta"],
        test=dict(p=[round(float(v), 5) for v in pt], y=[int(v) for v in yt],
                  session=sessions, idx=idxs),
        metrics_test=mt, per_session=per_sess, args=vars(args)), ensure_ascii=False))
    (OUT / f"perfield_metrics_{args.size}_s{args.seed}.json").write_text(json.dumps(
        dict(metrics_test=mt, per_session=per_sess, args=vars(args), meta=sp["meta"]),
        ensure_ascii=False, indent=1))
    print("saved checkpoint + predictions")


if __name__ == "__main__":
    main()
