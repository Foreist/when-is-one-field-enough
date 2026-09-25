# -*- coding: utf-8 -*-
"""Controlled leakage test for the OOC Image Dataset.

Holds the TEST IMAGES and the TRAINING SET SIZE fixed; the only difference between
the two arms is whether the training set contains other images from the same
sessions as the test images.

  test pool      : sessions T (default 20 sessions). Half of each T-session's images
                   are withheld as the test set; the other half are "available".
  arm DISJOINT   : train = N images drawn only from non-T sessions
  arm LEAKY      : train = (N - k) images from non-T sessions + k images from T sessions
                   (k = number of available T images)  -> same total N
                   (N = non-T pool minus the 200 validation images, identical in both arms)
  both arms are evaluated on the identical withheld test images.

Difference in test accuracy = inflation caused by session leakage.
"""
import argparse, collections, json, random, re, time
from pathlib import Path

import numpy as np
import torch

from leakage_experiment import (OOC, evaluate, index_images, make_loaders, PAT,
                                build_cellaware_session_split)
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn as nn

OUT = Path(__file__).resolve().parent.parent / "results"


def split_controlled(recs, seed, n_test_sessions=20, test_frac=0.5):
    rng = random.Random(seed)
    by_sess = collections.defaultdict(list)
    for r in recs:
        by_sess[r["session"]].append(r)
    sessions = sorted(by_sess)
    rng.shuffle(sessions)
    T = sessions[:n_test_sessions]
    nonT = sessions[n_test_sessions:]
    test_imgs, avail_T = [], []
    for s in T:
        v = by_sess[s][:]
        rng.shuffle(v)
        cut = int(len(v) * test_frac)
        test_imgs.extend(v[:cut])
        avail_T.extend(v[cut:])
    pool = [r for s in nonT for r in by_sess[s]]
    rng.shuffle(pool)
    N = len(pool)
    k = len(avail_T)
    if k >= N:
        k = N // 4
        avail_T = avail_T[:k]
    disjoint = pool[:N]                      # all non-T
    # a small val split from non-T (fixed across arms)
    val = disjoint[-200:]
    # leaky train: swap k non-T images for the k available T images -> same size N - 200
    leaky = pool[:N - 200 - k] + avail_T
    assert len(leaky) == len(disjoint) - 200
    return dict(disjoint=dict(train=disjoint[:-200], val=val, test=test_imgs),
                leaky=dict(train=leaky, val=val, test=test_imgs),
                meta=dict(n_test_sessions=len(T), test_images=len(test_imgs),
                          available_T=len(avail_T), pool=len(pool), train_size=len(disjoint) - 200))


def run_arm(split, args, seed, device, tag):
    torch.manual_seed(seed)
    loaders = make_loaders(split, args.size, args.batch, args.workers, seed)
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; n = 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(y); n += len(y)
        if ep == args.epochs - 1:
            va = evaluate(model, loaders["val"], device)
            print(f"  [{tag} s{seed}] last ep loss {tot/max(1,n):.4f} val_acc {va[0]:.4f}", flush=True)
    sess = [r["session"] for r in split["test"]]
    te = evaluate(model, loaders["test"], device, sessions=sess)
    print(f"  [{tag} s{seed}] TEST acc {te[0]:.4f} auc {te[1]:.4f}", flush=True)
    return dict(tag=tag, seed=seed, acc=te[0], auc=te[1], per_session=te[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-test-sessions", type=int, default=20)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = index_images()
    results = []
    for seed in range(args.seeds):
        sp = split_controlled(recs, seed=1000 + seed, n_test_sessions=args.n_test_sessions)
        print(f"[seed {seed}] meta:", sp["meta"], flush=True)
        results.append(run_arm(sp["disjoint"], args, seed, device, "disjoint"))
        results.append(run_arm(sp["leaky"], args, seed, device, "leaky"))
    a = [r for r in results if r["tag"] == "disjoint"]
    b = [r for r in results if r["tag"] == "leaky"]
    summary = dict(
        n_images=len(recs), n_test_sessions=args.n_test_sessions, args=vars(args),
        disjoint=dict(acc=float(np.mean([r["acc"] for r in a])),
                      acc_sd=float(np.std([r["acc"] for r in a])),
                      auc=float(np.mean([r["auc"] for r in a]))),
        leaky=dict(acc=float(np.mean([r["acc"] for r in b])),
                   acc_sd=float(np.std([r["acc"] for r in b])),
                   auc=float(np.mean([r["auc"] for r in b]))),
        inflation_acc=float(np.mean([r["acc"] for r in b]) - np.mean([r["acc"] for r in a])),
        inflation_auc=float(np.mean([r["auc"] for r in b]) - np.mean([r["auc"] for r in a])),
        runs=results)
    (OUT / "leakage_controlled.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, ensure_ascii=False, indent=1))
    print("saved", OUT / "leakage_controlled.json")


if __name__ == "__main__":
    main()
