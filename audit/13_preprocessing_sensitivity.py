#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much does the per-field P(bad) depend on how the image is resized? (REPORT §6.8)

The model was trained on 2,056 x 1,542 fields resized to 384 px with PIL bilinear + antialiasing
(torchvision Resize on PIL images). Two deployments differ from that:
  * the bundled demo examples are stored at 1,024 x 768 (to keep the repository small);
  * the browser demo resizes with the canvas (drawImage), not PIL.
On 200 random test fields (seed 0) this script compares the shipped preprocessing with other
resamplers, and on the 44 bundled fields it compares the original with the stored copy, including
the chip calls.

Writes results/preprocessing_sensitivity.json
"""
import json, random, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from inference import TF, load_model, natural_key, sequential_decision   # noqa: E402
from leakage_experiment import index_images                               # noqa: E402
from leakage_controlled import split_controlled                           # noqa: E402

NORM = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def main():
    model = load_model(str(ROOT / "model" / "perfield_mnv3s_384_s0.pt"))
    p = lambda x: float(torch.softmax(model(x[None]), dim=1)[0, 0])
    recs = index_images()
    test = split_controlled(recs, seed=1000, n_test_sessions=25)["disjoint"]["test"]
    random.Random(0).shuffle(test)
    alt = {
        "bilinear_no_antialias": lambda im: transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
            transforms.functional.resize(transforms.functional.to_tensor(im), [384, 384], antialias=False)),
        "nearest": lambda im: NORM(im.resize((384, 384), Image.NEAREST)),
        "bicubic": lambda im: NORM(im.resize((384, 384), Image.BICUBIC)),
        "box": lambda im: NORM(im.resize((384, 384), Image.BOX)),
    }
    base, other = [], {k: [] for k in alt}
    with torch.no_grad():
        for r in test[:200]:
            im = Image.open(r["path"]).convert("RGB")
            base.append(p(TF(im)))
            for k, f in alt.items():
                other[k].append(p(f(im)))
    b = np.array(base)
    resamplers = {k: dict(max_abs_diff=float(np.abs(np.array(v) - b).max()),
                          mean_abs_diff=float(np.abs(np.array(v) - b).mean()),
                          flips=int(((np.array(v) > 0.5) != (b > 0.5)).sum()), n=len(b))
                  for k, v in other.items()}

    by_name = {r["name"]: r for r in recs}
    examples = {}
    with torch.no_grad():
        for d in sorted((ROOT / "demo" / "examples").iterdir()):
            fs = sorted(d.glob("*.png"), key=natural_key)
            if not fs:
                continue
            po = [p(TF(Image.open(by_name[f.name.split("_", 1)[1]]["path"]).convert("RGB"))) for f in fs]
            pe = [p(TF(Image.open(f).convert("RGB"))) for f in fs]
            a, e = sequential_decision(po), sequential_decision(pe)
            examples[d.name] = dict(
                n=len(fs), max_abs_diff=float(np.abs(np.array(po) - np.array(pe)).max()),
                flips=int(((np.array(po) > 0.5) != (np.array(pe) > 0.5)).sum()),
                original=dict(call=a["call"], p_bad=a["p_bad"], fields=a["n_fields"]),
                stored=dict(call=e["call"], p_bad=e["p_bad"], fields=e["n_fields"]))
    sizes = sorted({Image.open(f).size for f in (ROOT / "demo" / "examples").glob("*/*.png")})
    out = dict(resamplers_vs_shipped=resamplers, bundled_examples=examples, bundled_example_size=list(sizes[0]))
    print(json.dumps(out, indent=1))
    (ROOT / "results" / "preprocessing_sensitivity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
