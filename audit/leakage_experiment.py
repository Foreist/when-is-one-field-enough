# -*- coding: utf-8 -*-
"""OOC Image Dataset — does the published (image-level) split leak sessions?

A: published split as shipped in the zip  (train/val/test folders)
B: session-grouped split rebuilt from all images (no session spans two splits)

Same model, same budget, 3 seeds. Report test acc/AUC gap.

Usage: python3 leakage_experiment.py [--epochs 6] [--seeds 3] [--size 224] [--limit N]
"""
import argparse, collections, json, os, random, re, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]          # repo/../.. = competition folder
import os
DATA = Path(os.environ.get("OOC_DATA", ROOT / "data" / "OOC_image_dataset"))
OUT = Path(__file__).resolve().parent
PAT = re.compile(r"^(\d{6})_(\d+)\.png$")


def index_images():
    """-> list of dict(path, split, cls, cell, day, session)"""
    recs = []
    for p in DATA.rglob("*.png"):
        parts = p.relative_to(DATA).parts          # split/cls/cell/day/file
        if len(parts) != 5:
            continue
        m = PAT.match(parts[4])
        if not m:
            continue
        recs.append(dict(path=str(p), split=parts[0], cls=parts[1], cell=parts[2],
                         day=parts[3], session=m.group(1), name=parts[4]))
    return recs


class OOC(Dataset):
    def __init__(self, recs, tf):
        self.recs = recs
        self.tf = tf

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(r["path"]).convert("RGB")
        y = 1 if r["cls"] == "good" else 0
        return self.tf(img), y


def build_session_split(recs, seed, ratios=(0.70, 0.10, 0.20)):
    """Assign whole sessions to train/val/test so image counts follow ratios."""
    by_sess = collections.defaultdict(list)
    for r in recs:
        by_sess[r["session"]].append(r)
    sess = sorted(by_sess)
    rng = random.Random(seed)
    rng.shuffle(sess)
    total = len(recs)
    targets = [ratios[0] * total, ratios[1] * total, ratios[2] * total]
    out = {"train": [], "val": [], "test": []}
    counts = [0, 0, 0]
    for s in sess:
        # put the session into whichever split is furthest below target
        deficits = [t - c for t, c in zip(targets, counts)]
        k = int(np.argmax(deficits))
        out[["train", "val", "test"][k]].extend(by_sess[s])
        counts[k] += len(by_sess[s])
    return out


def build_cellaware_session_split(recs, seed, test_ratio=0.20, val_ratio=0.10):
    """Sessions are disjoint across splits AND every cell type appears in test and train.

    Greedy: rarest cell types first pick one (smallest) session for test, then fill test
    to the target ratio, then hold out val sessions from what is left.
    """
    by_sess = collections.defaultdict(list)
    for r in recs:
        by_sess[r["session"]].append(r)
    sess_of_cell = collections.defaultdict(set)
    for s, v in by_sess.items():
        for c in {r["cell"] for r in v}:
            sess_of_cell[c].add(s)
    rng = random.Random(seed)
    total = len(recs)
    test_target = test_ratio * total

    test_sessions = set()
    for c in sorted(sess_of_cell, key=lambda c: len(sess_of_cell[c])):
        if any(c in {r["cell"] for r in by_sess[s]} for s in test_sessions):
            continue
        cands = sorted(sess_of_cell[c], key=lambda s: len(by_sess[s]))
        rng.shuffle(cands[:2])
        test_sessions.add(cands[0])
    # fill test
    rest = sorted((s for s in by_sess if s not in test_sessions),
                  key=lambda s: -len(by_sess[s]))
    n_test = sum(len(by_sess[s]) for s in test_sessions)
    for s in rest:
        if n_test >= test_target:
            break
        test_sessions.add(s)
        n_test += len(by_sess[s])
    # val: hold out small sessions (with cell coverage if easy)
    rest = [s for s in by_sess if s not in test_sessions]
    rest.sort(key=lambda s: len(by_sess[s]))
    val_sessions = set(rest[:max(2, int(len(rest) * 0.10))])
    train_sessions = [s for s in by_sess if s not in test_sessions and s not in val_sessions]
    out = {"train": [], "val": [], "test": []}
    for s in train_sessions:
        out["train"].extend(by_sess[s])
    for s in val_sessions:
        out["val"].extend(by_sess[s])
    for s in test_sessions:
        out["test"].extend(by_sess[s])
    return out


def make_loaders(split, size, bs, workers, seed):
    tf_tr = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_ev = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    g = torch.Generator().manual_seed(seed)
    return {k: DataLoader(OOC(v, tf_tr if k == "train" else tf_ev),
                          batch_size=bs, shuffle=(k == "train"), num_workers=workers,
                          pin_memory=True, generator=g)
            for k, v in split.items()}


@torch.no_grad()
def evaluate(model, loader, device, sessions=None):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logit = model(x)
        p = torch.softmax(logit, dim=1)[:, 1]
        ps.append(p.cpu().numpy()); ys.append(y.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    acc = float(((p > 0.5) == y).mean())
    # AUC
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    auc = float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")
    per_sess = {}
    if sessions is not None:
        for s in sorted(set(sessions)):
            m = np.array([x == s for x in sessions])
            if m.sum():
                per_sess[s] = dict(n=int(m.sum()), acc=float(((p[m] > 0.5) == y[m]).mean()))
    return acc, auc, per_sess


def comp(rs):
    return dict(n=len(rs), sessions=len({r["session"] for r in rs}),
                cells=dict(collections.Counter(r["cell"] for r in rs)),
                good=sum(1 for r in rs if r["cls"] == "good"))


def train_eval(split, args, seed, device, tag):
    torch.manual_seed(seed)
    loaders = make_loaders(split, args.size, args.batch, args.workers, seed)
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train()
        t0 = time.time(); tot = 0.0; n = 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(y); n += len(y)
        va = evaluate(model, loaders["val"], device)
        print(f"  [{tag} s{seed}] ep{ep+1} loss {tot/max(1,n):.4f} val_acc {va[0]:.4f} "
              f"val_auc {va[1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    te_sessions = [r["session"] for r in split["test"]]
    te = evaluate(model, loaders["test"], device, sessions=te_sessions)
    print(f"  [{tag} s{seed}] TEST acc {te[0]:.4f} auc {te[1]:.4f}", flush=True)
    return dict(tag=tag, seed=seed, test_acc=te[0], test_auc=te[1],
                n_train=len(split["train"]), n_val=len(split["val"]), n_test=len(split["test"]),
                test_cells=comp(split["test"])["cells"], test_sessions=comp(split["test"])["sessions"],
                per_session_acc=te[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="use only N images (debug)")
    ap.add_argument("--variants", type=str, default="published,cellaware")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)
    recs = index_images()
    print("images:", len(recs), flush=True)
    if args.limit:
        rng = random.Random(0); rng.shuffle(recs); recs = recs[:args.limit]

    pub = {k: [r for r in recs if r["split"] == k] for k in ("train", "val", "test")}
    print("published split sizes:", {k: len(v) for k, v in pub.items()}, flush=True)
    sess = {k: len({r["session"] for r in v}) for k, v in pub.items()}
    print("published sessions:", sess, flush=True)

    # label heterogeneity within session
    by_sess = collections.defaultdict(list)
    for r in recs:
        by_sess[r["session"]].append(r)
    mixed = sum(1 for v in by_sess.values() if len({r["cls"] for r in v}) > 1)
    frac_minor = []
    for v in by_sess.values():
        c = collections.Counter(r["cls"] for r in v)
        frac_minor.append(min(c.values()) / len(v))
    print(f"sessions: {len(by_sess)}  mixed: {mixed}  "
          f"minority-label share mean {np.mean(frac_minor):.3f}", flush=True)

    print("published composition:", {k: comp(v) for k, v in pub.items()}, flush=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    results = []
    for seed in range(args.seeds):
        if "published" in variants:
            results.append(train_eval(pub, args, seed, device, "published"))
        if "cellaware" in variants:
            grp = build_cellaware_session_split(recs, seed=100 + seed)
            print(f"  [cellaware s{seed}] composition:", {k: comp(v) for k, v in grp.items()}, flush=True)
            results.append(train_eval(grp, args, seed, device, "cellaware-grouped"))
        if "random" in variants:
            grp = build_session_split(recs, seed=100 + seed)
            results.append(train_eval(grp, args, seed, device, "random-grouped"))

    def agg(tag):
        r = [x for x in results if x["tag"] == tag]
        if not r:
            return {}
        return dict(acc_mean=float(np.mean([x["test_acc"] for x in r])),
                    acc_sd=float(np.std([x["test_acc"] for x in r])),
                    auc_mean=float(np.mean([x["test_auc"] for x in r])))

    a, b = agg("published"), agg("cellaware-grouped")
    summary = dict(n_images=len(recs),
                   published=a, cellaware_grouped=b, random_grouped=agg("random-grouped"),
                   gap_acc=(a.get("acc_mean", float("nan")) - b.get("acc_mean", float("nan"))),
                   gap_auc=(a.get("auc_mean", float("nan")) - b.get("auc_mean", float("nan"))),
                   sessions_total=len(by_sess), sessions_mixed=mixed,
                   minority_share_mean=float(np.mean(frac_minor)),
                   args=vars(args), runs=results)
    (OUT / "leakage_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, ensure_ascii=False, indent=1))
    print("saved", OUT / "leakage_results.json")


if __name__ == "__main__":
    main()
