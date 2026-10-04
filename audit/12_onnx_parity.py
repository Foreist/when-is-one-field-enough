#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the browser model (ONNX, on the Hugging Face demo) give the same P(bad) as the PyTorch checkpoint?

Runs both on the 44 bundled example fields and reports the largest absolute difference in the
softmax output. Needs onnxruntime and huggingface_hub (listed in requirements.txt; only this
check uses them).

Writes results/onnx_parity.json
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inference import TF, load_model   # noqa: E402

SPACE = "taewoong23/ooc-chip-qc-demo"
REVISION = "d5d27e9134a4d4cc6dcb39ef1250e3058bb8e92a"
REMOTE_SHA256 = "bedae8a18f3662e0dd90f1d334c939cd50fef79e73ab3ded6cb429c701700c79"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, help="check a locally exported model instead of downloading")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "onnx_parity.json")
    args = ap.parse_args()
    model = load_model(str(ROOT / "model" / "perfield_mnv3s_384_s0.pt"))
    path = args.onnx or Path(hf_hub_download(SPACE, "ooc_model.onnx", repo_type="space", revision=REVISION))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not args.onnx and digest != REMOTE_SHA256:
        raise ValueError("pinned browser model SHA-256 mismatch")
    sess = ort.InferenceSession(str(path))
    name = sess.get_inputs()[0].name
    files = sorted((ROOT / "demo" / "examples").glob("*/*.png"))
    diffs = []
    for f in files:
        x = TF(Image.open(f).convert("RGB"))[None]
        with torch.no_grad():
            a = np.asarray(torch.softmax(model(x), dim=1).numpy())
        if a.shape != (1, 2) or not np.isfinite(a).all():
            raise ValueError(f"PyTorch produced invalid probabilities for {f.name}")
        o = np.asarray(sess.run(None, {name: x.numpy()})[0])
        if o.shape != a.shape or not np.isfinite(o).all():
            raise ValueError(f"ONNX produced invalid output for {f.name}")
        if not np.allclose(o.sum(-1), 1, atol=1e-4):          # logits -> softmax
            o = np.exp(o - o.max(-1, keepdims=True)); o /= o.sum(-1, keepdims=True)
        if not np.isfinite(o).all():
            raise ValueError(f"ONNX produced invalid probabilities for {f.name}")
        diff = float(np.abs(a - o).max())
        if not np.isfinite(diff):
            raise ValueError(f"non-finite parity difference for {f.name}")
        diffs.append(diff)
    out = dict(space=SPACE, revision=REVISION if not args.onnx else None, onnx_sha256=digest,
               n_fields=len(files), max_abs_diff=max(diffs), mean_abs_diff=float(np.mean(diffs)),
               scope="identical Python-preprocessed tensors; not browser input-path validation")
    if max(diffs) > 1e-4:
        raise ValueError(f"ONNX parity exceeded tolerance: {max(diffs)}")
    print(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
