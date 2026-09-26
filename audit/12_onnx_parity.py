#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the browser model (ONNX, on the Hugging Face demo) give the same P(bad) as the PyTorch checkpoint?

Runs both on the 44 bundled example fields and reports the largest absolute difference in the
softmax output. Needs onnxruntime and huggingface_hub (not in requirements.txt: only this check
uses them).

Writes results/onnx_parity.json
"""
import json, sys
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


def main():
    model = load_model(str(ROOT / "model" / "perfield_mnv3s_384_s0.pt"))
    sess = ort.InferenceSession(hf_hub_download(SPACE, "ooc_model.onnx", repo_type="space"))
    name = sess.get_inputs()[0].name
    files = sorted((ROOT / "demo" / "examples").glob("*/*.png"))
    diffs = []
    for f in files:
        x = TF(Image.open(f).convert("RGB"))[None]
        with torch.no_grad():
            a = torch.softmax(model(x), dim=1).numpy()
        o = sess.run(None, {name: x.numpy()})[0]
        if not np.allclose(o.sum(-1), 1, atol=1e-4):          # logits -> softmax
            o = np.exp(o - o.max(-1, keepdims=True)); o /= o.sum(-1, keepdims=True)
        diffs.append(float(np.abs(a - o).max()))
    out = dict(space=SPACE, n_fields=len(files), max_abs_diff=max(diffs), mean_abs_diff=float(np.mean(diffs)))
    print(out)
    (ROOT / "results" / "onnx_parity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
