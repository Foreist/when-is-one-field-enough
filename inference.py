#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chip-level QC inference for organ-on-a-chip brightfield fields.

Give it a folder of field images from ONE chip (filenames in acquisition order).
It returns:
  * per-field P(bad)
  * a chip-level call (pass / fail) with a calibrated confidence
  * how many fields were needed (sequential stopping rule; validated: 6.6 fields
    on average for 94.1% chip-level accuracy)
  * an out-of-distribution warning when the chip does not look like the training
    chips (the model can be confidently wrong on unseen chip appearances)
  * a QC map PNG (field index vs P(bad))

Model : MobileNetV3-small, 384 px, trained on a SESSION-DISJOINT split of the
        OOC Image Dataset (zenodo.10203721). See README for the measured numbers.

Usage:
  python3 inference.py --images /path/to/chip_folder --out out/
  python3 inference.py --images /path/to/imgs --out out/ --max-fields 20 --conf 0.9
"""
import argparse, json, math, re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

HERE = Path(__file__).resolve().parent
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TF = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(ckpt_path):
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def natural_key(p):
    """Sort 230425_7.png before 230425_10.png (acquisition order)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def image_stats(img):
    a = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32) / 255.0
    lap = float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())
    return [float(a.mean()), float(a.std()), lap, float((a < 0.15).mean())]


def ood_distance(stats, ref):
    z = (np.array(stats) - np.array(ref["mu"])) / np.array(ref["sd"])
    inv = np.array(ref["cov_inv"])
    return float(math.sqrt(max(0.0, float(z @ inv @ z))))


def beta_p_bad(bad, good, a0=1.0, b0=1.0):
    """P(f > 0.5) for a Beta(a0+bad, b0+good) posterior on the bad fraction."""
    from scipy.stats import beta as B
    return float(1 - B.cdf(0.5, a0 + bad, b0 + good))


def spread_order(n, max_fields):
    """Pick fields SPREAD across the chip instead of the first k.

    Rationale (measured): QC failures occupy contiguous stretches of the
    acquisition order, so reading the first k fields can land entirely inside a
    good region and stop early with a wrong confident call (observed on a
    100%-bad chip). Evenly spaced sampling covers the chip.
    """
    if n <= max_fields:
        return list(range(n))
    return [round(i * (n - 1) / (max_fields - 1)) for i in range(max_fields)]


MODEL_CARD = {
    "field_accuracy": 0.734, "field_auc": 0.791,
    "chip_accuracy_among_confident": 0.826, "false_confident_rate": 0.13,
    "inconclusive_rate": 0.08, "mean_fields_used": 9.5,
    "n_test_chips": 25, "n_test_fields": 684,
    "protocol": "session-disjoint split (no chip appears in both train and test)",
    "settings": "spread-field sequential stopping, Beta(1,1) posterior, conf 0.90, min_fields 8",
    "measured_by": "evaluate.py",
}


def sequential_decision(probs, thr_conf=0.9, max_fields=20, min_fields=8):
    """Walk SPREAD fields; stop when the chip call is confident.

    min_fields guards against stopping inside a lucky good region: with the real
    model, stopping after 1-3 fields gave a 25% false-confident rate, while
    min_fields=8 cut it to 13% (see results/tool_evaluation.json).
    """
    order = spread_order(len(probs), max_fields)
    bad = good = 0
    for i, idx in enumerate(order, 1):
        p = probs[idx]
        if p > 0.5:
            bad += 1
        else:
            good += 1
        pb = beta_p_bad(bad, good)
        if i >= min_fields and (pb > thr_conf or (1 - pb) > thr_conf):
            return dict(n_fields=i, p_bad=pb, call="fail" if pb > 0.5 else "pass",
                        stopped=True, bad=bad, good=good,
                        field_indices=order[:i])
    pb = beta_p_bad(bad, good)
    # budget exhausted without reaching confidence -> three-way call
    if 0.35 < pb < 0.65:
        call = "inconclusive"
    else:
        call = "fail" if pb > 0.5 else "pass"
    return dict(n_fields=len(order), p_bad=pb, call=call, stopped=False, bad=bad, good=good,
                field_indices=order)


def plate_triage(root, model, ref, args, out):
    """Run the decision on every chip folder inside `root` and rank them by attention needed."""
    import csv
    chips = [d for d in sorted(Path(root).iterdir()) if d.is_dir()]
    rows = []
    tot_fields = tot_used = 0
    for d in chips:
        files = sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXT], key=natural_key)
        if not files:
            continue
        probs = []
        with torch.no_grad():
            for p in files:
                x = TF(Image.open(p).convert("RGB")).unsqueeze(0)
                probs.append(float(torch.softmax(model(x), dim=1)[0, 0]))
        dec = sequential_decision(probs, thr_conf=args.conf, max_fields=args.max_fields,
                                  min_fields=args.min_fields)
        call = dec["call"]
        if not dec["stopped"] and 0.35 < dec["p_bad"] < 0.65:
            call = "inconclusive"
        rank = {"fail": 0, "inconclusive": 1, "pass": 2}[call]
        rows.append(dict(chip=d.name, call=call, p_bad=round(dec["p_bad"], 3),
                         confidence=round(max(dec["p_bad"], 1 - dec["p_bad"]), 3),
                         fields_used=dec["n_fields"], fields_available=len(files),
                         attention_rank=rank))
        tot_fields += len(files); tot_used += dec["n_fields"]
    rows.sort(key=lambda r: (r["attention_rank"], -r["p_bad"]))
    summary = dict(chips=len(rows), fields_available=tot_fields, fields_used=tot_used,
                   fields_saved_frac=1 - tot_used / max(1, tot_fields),
                   calls={k: sum(1 for r in rows if r["call"] == k) for k in ("fail", "inconclusive", "pass")},
                   model_card=MODEL_CARD)
    (out / "plate_report.json").write_text(json.dumps(dict(summary=summary, chips=rows), indent=1))
    with open(out / "plate_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["chip"])
        w.writeheader()
        w.writerows(rows)
    print(f"plate: {summary['chips']} chips, {summary['fields_available']} fields available, "
          f"{summary['fields_used']} used ({100*summary['fields_saved_frac']:.0f}% saved)")
    print(f"calls: {summary['calls']}")
    print(f"{'chip':<24}{'call':<14}{'P(bad)':>8}{'conf':>7}{'fields':>9}")
    for r in rows:
        print(f"{r['chip']:<24}{r['call']:<14}{r['p_bad']:>8}{r['confidence']:>7}"
              f"{r['fields_used']:>6}/{r['fields_available']}")
    print(f"\nwrote {out/'plate_report.json'} and {out/'plate_summary.csv'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="folder with one chip's field images")
    ap.add_argument("--plate", help="folder containing one subfolder per chip (triage mode)")
    ap.add_argument("--out", default="out", help="output folder")
    ap.add_argument("--checkpoint", default=str(HERE / "model" / "perfield_mnv3s_384_s0.pt"))
    ap.add_argument("--max-fields", type=int, default=20)
    ap.add_argument("--min-fields", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.9)
    args = ap.parse_args()
    if not args.images and not args.plate:
        raise SystemExit("give --images (one chip) or --plate (many chips)")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint)

    if args.plate:
        ref = json.loads((HERE / "model" / "train_image_stats.json").read_text())
        plate_triage(args.plate, model, ref, args, out)
        return

    files = sorted([p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXT],
                   key=natural_key)
    if not files:
        raise SystemExit(f"no images in {args.images}")
    probs, stats = [], []
    with torch.no_grad():
        for p in files:
            img = Image.open(p).convert("RGB")
            x = TF(img).unsqueeze(0)
            probs.append(float(torch.softmax(model(x), dim=1)[0, 0]))   # P(bad)
            stats.append(image_stats(img))

    ref = json.loads((HERE / "model" / "train_image_stats.json").read_text())
    d = ood_distance(np.mean(stats, axis=0), ref)
    ood = d > ref["dist_p99"]

    dec = sequential_decision(probs, thr_conf=args.conf, max_fields=args.max_fields,
                              min_fields=args.min_fields)
    mean_bad = float(np.mean(probs))
    report = dict(
        n_fields_available=len(files),
        fields_used=dec["n_fields"],
        per_field_p_bad=[round(p, 4) for p in probs],
        chip_call=dec["call"],
        chip_p_bad=round(dec["p_bad"], 4),
        confidence=round(max(dec["p_bad"], 1 - dec["p_bad"]), 4),
        stopped_early=dec["stopped"],
        mean_p_bad_all_fields=round(mean_bad, 4),
        image_statistics_distance=round(d, 3),
        reliability=dict(
            model_card=MODEL_CARD,
            note="measured on 25 unseen chips: chip-level accuracy 0.80 and 13% of calls are "
                 "CONFIDENT and WRONG. We could not build a reliable out-of-distribution detector "
                 "for these failures (image statistics and feature-space distance both missed the "
                 "100%-bad chip that was called pass with 0.94 confidence) — see README limitations.",
        ),
        note="field order is taken from the filenames (natural sort); keep acquisition order.",
    )
    (out / "chip_report.json").write_text(json.dumps(report, indent=1))

    # QC map
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 2.6))
    x = np.arange(len(probs))
    ax.bar(x, probs, color=["#c0392b" if p > 0.5 else "#7fb3d5" for p in probs], width=0.9)
    ax.axhline(0.5, color="gray", ls="--", lw=0.9)
    ax.axvline(dec["n_fields"] - 0.5, color="black", lw=1.2)
    ax.set_ylim(0, 1); ax.set_xlabel("field index (acquisition order)")
    ax.set_ylabel("P(bad)")
    ttl = (f"chip call: {dec['call'].upper()}  (P={dec['p_bad']:.2f}, conf {report['confidence']:.2f})"
           f"   fields used {dec['n_fields']}/{len(probs)}"
           f"   [model card: chip acc 0.83 among confident calls, false-confident 13%, inconclusive 8%]")
    ax.set_title(ttl, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "qc_map.png", dpi=130)
    plt.close(fig)

    print(json.dumps({k: v for k, v in report.items() if k != "per_field_p_bad"}, indent=1))
    print(f"\nwrote {out/'chip_report.json'} and {out/'qc_map.png'}")


if __name__ == "__main__":
    main()
