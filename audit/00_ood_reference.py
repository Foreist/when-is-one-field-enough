#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the out-of-distribution reference statistics shipped in model/ (REPORT §6.5).

On the 1,495 training fields of the shipped split (split_controlled seed 1000, 25 test sessions):
  image statistics : chip-free per-field [mean, s.d., mean absolute gradient, dark fraction],
                     z-scored; Mahalanobis distance distribution -> model/train_image_stats.json
  feature space    : 576-d pooled MobileNetV3 feature of the shipped checkpoint; Mahalanobis
                     distance (covariance + 1e-3 ridge) distribution -> model/train_feature_stats.json

`--check` recomputes both and compares with the shipped files instead of overwriting them.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from inference import TF, image_stats, load_model          # noqa: E402
from leakage_experiment import index_images                # noqa: E402
from leakage_controlled import split_controlled            # noqa: E402


def maha(z, inv):
    return np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", z, inv, z)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    tr = split_controlled(index_images(), seed=1000, n_test_sessions=25)["disjoint"]["train"]
    model = load_model(str(ROOT / "model" / "perfield_mnv3s_384_s0.pt"))
    st, F = [], []
    with torch.no_grad():
        for i in range(0, len(tr), 32):
            ims = [Image.open(r["path"]).convert("RGB") for r in tr[i:i + 32]]
            st += [image_stats(im) for im in ims]
            x = torch.stack([TF(im) for im in ims])
            F.append(model.avgpool(model.features(x)).flatten(1).numpy())
    st = np.array(st); F = np.concatenate(F)

    mu, sd = st.mean(0), st.std(0)
    z = (st - mu) / sd
    inv = np.linalg.inv(np.cov(z, rowvar=False))
    d = maha(z, inv)
    img = dict(features=["mean", "std", "lap", "dark"], mu=mu.tolist(), sd=sd.tolist(), cov_inv=inv.tolist(),
               dist_p50=float(np.percentile(d, 50)), dist_p99=float(np.percentile(d, 99)),
               dist_max=float(d.max()), n_train=len(tr))

    fmu = F.mean(0)
    # 576 dimensions from 1,495 fields: a small ridge keeps the covariance invertible
    finv = np.linalg.pinv(np.cov(F - fmu, rowvar=False) + 1e-3 * np.eye(F.shape[1]))
    fd = maha(F - fmu, finv)
    feat = dict(mu=fmu.tolist(), cov_inv=finv.tolist(), thr_p95=float(np.percentile(fd, 95)),
                thr_p99=float(np.percentile(fd, 99)), train_p50=float(np.percentile(fd, 50)),
                train_max=float(fd.max()))

    if args.check:
        for name, new in (("train_image_stats.json", img), ("train_feature_stats.json", feat)):
            old = json.loads((ROOT / "model" / name).read_text())
            for k in new:
                if k in ("cov_inv", "mu", "sd", "features"):
                    continue
                print(f"{name:<26} {k:<10} shipped {old.get(k)}  recomputed {new[k]}")
        return
    (ROOT / "model" / "train_image_stats.json").write_text(json.dumps(img))
    (ROOT / "model" / "train_feature_stats.json").write_text(json.dumps(feat))
    print("wrote model/train_image_stats.json and model/train_feature_stats.json")


if __name__ == "__main__":
    main()
