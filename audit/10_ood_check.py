#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do the shipped out-of-distribution references flag the 25 unseen test chips?

Two detectors, both with reference statistics shipped in model/:
  image statistics : Mahalanobis distance of the chip-mean [brightness, contrast, sharpness,
                     dark fraction] to the training fields (train_image_stats.json; this is the
                     check inference.py runs), alarm above the training 99th percentile;
  feature space    : Mahalanobis distance of the 576-d pooled MobileNetV3 feature to the training
                     fields (train_feature_stats.json), alarm above the training 99th percentile.
                     Scored per chip two ways: the chip-mean feature, and the median field distance.

Reported: how many of the 25 chips each detector flags, and whether it flags session 230405
(the 100%-bad chip called pass with high confidence).

Writes results/ood_check.json
"""
import collections, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import TF, image_stats, load_model, ood_distance       # noqa: E402
from leakage_experiment import index_images                          # noqa: E402
from leakage_controlled import split_controlled                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"


def maha(x, mu, inv):
    d = x - mu
    return float(np.sqrt(max(0.0, float(d @ inv @ d))))


def main():
    recs = index_images()
    sp = split_controlled(recs, seed=1000, n_test_sessions=25)
    by = collections.defaultdict(list)
    for r in sp["disjoint"]["test"]:
        by[r["session"]].append(r)
    model = load_model(str(ROOT / "model" / "perfield_mnv3s_384_s0.pt"))
    img_ref = json.loads((ROOT / "model" / "train_image_stats.json").read_text())
    fs = json.loads((ROOT / "model" / "train_feature_stats.json").read_text())
    mu, inv = np.array(fs["mu"]), np.array(fs["cov_inv"])

    def feats(x):
        return model.avgpool(model.features(x)).flatten(1)

    per = {}
    with torch.no_grad():
        for s, v in sorted(by.items()):
            F, st, pb = [], [], []
            for i in range(0, len(v), 32):
                ims = [Image.open(r["path"]).convert("RGB") for r in v[i:i + 32]]
                x = torch.stack([TF(im) for im in ims])
                F.append(feats(x).numpy())
                pb += torch.softmax(model(x), dim=1)[:, 0].tolist()
                st += [image_stats(im) for im in ims]
            F = np.concatenate(F)
            d_field = [maha(f, mu, inv) for f in F]
            per[s] = dict(
                n_fields=len(v), bad_share=float(np.mean([r["cls"] != "good" for r in v])),
                mean_p_bad=float(np.mean(pb)),
                image_stats_distance=ood_distance(np.mean(st, axis=0), img_ref),
                feature_distance_chip_mean=maha(F.mean(0), mu, inv),
                feature_distance_median_field=float(np.median(d_field)),
                feature_frac_fields_above_p99=float(np.mean(np.array(d_field) > fs["thr_p99"])))
            print(s, {k: round(x, 3) for k, x in per[s].items()}, flush=True)

    flag = dict(
        image_stats=[s for s in per if per[s]["image_stats_distance"] > img_ref["dist_p99"]],
        feature_chip_mean=[s for s in per if per[s]["feature_distance_chip_mean"] > fs["thr_p99"]],
        feature_median_field=[s for s in per if per[s]["feature_distance_median_field"] > fs["thr_p99"]])
    summary = {k: dict(n_flagged=len(v), flags_230405="230405" in v, sessions=sorted(v))
               for k, v in flag.items()}
    for k, v in summary.items():
        print(f"{k:>22}: {v['n_flagged']}/{len(per)} flagged, 230405 flagged: {v['flags_230405']}")
    (OUT / "ood_check.json").write_text(json.dumps(dict(
        n_chips=len(per), image_stats_p99=img_ref["dist_p99"], feature_p99=fs["thr_p99"],
        summary=summary, per_session=per), indent=1))


if __name__ == "__main__":
    main()
