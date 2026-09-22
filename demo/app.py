#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gradio demo: chip-level QC from a handful of brightfield fields.

    python3 demo/app.py            # then open the printed local URL

Pick a bundled example chip or upload your own field images (one chip per run).
"""
import json, sys
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from inference import (MODEL_CARD, TF, beta_p_bad, load_model,  # noqa: E402
                       spread_order)

MODEL = load_model(str(ROOT / "model" / "perfield_mnv3s_384_s0.pt"))
EXAMPLES = sorted([p for p in (HERE / "examples").glob("*") if p.is_dir()])


def run_chip(files, max_fields=20, min_fields=8, conf=0.9):
    paths = sorted([Path(f) for f in files], key=lambda p: p.name)
    if not paths:
        return None, "upload at least one image (one chip per run)", ""
    probs = []
    with torch.no_grad():
        for p in paths:
            x = TF(Image.open(p).convert("RGB")).unsqueeze(0)
            probs.append(float(torch.softmax(MODEL(x), dim=1)[0, 0]))
    order = spread_order(len(probs), max_fields)
    bad = good = 0
    call, used, stopped = None, len(order), False
    for i, idx in enumerate(order, 1):
        bad += int(probs[idx] > 0.5)
        good += int(probs[idx] <= 0.5)
        pb = beta_p_bad(bad, good)
        if i >= min_fields and (pb > conf or (1 - pb) > conf):
            call, used, stopped, p_final = ("fail" if pb > 0.5 else "pass"), i, True, pb
            break
    if call is None:
        p_final = beta_p_bad(bad, good)
        call = "inconclusive" if 0.35 < p_final < 0.65 else ("fail" if p_final > 0.5 else "pass")

    fig, ax = plt.subplots(figsize=(11, 2.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.18)
    x = np.arange(len(probs))
    ax.bar(x, probs, color=["#c0392b" if p > 0.5 else "#7fb3d5" for p in probs], width=0.9)
    ax.axhline(0.5, color="gray", ls="--", lw=0.9)
    ax.axvline(used - 0.5, color="black", lw=1.4)
    tx, ha = used + 0.3, "left"
    if tx > len(probs) - 4:                      # keep the label inside the axes
        tx, ha = len(probs) - 0.5, "right"
    ax.text(tx, 1.04, f"used {used} of {len(probs)} fields", fontsize=8, ha=ha, va="top")
    ax.set_ylim(0, 1.12); ax.set_xlabel("field index (acquisition order)", fontsize=9)
    ax.set_ylabel("P(bad)", fontsize=9)
    ax.tick_params(labelsize=8)

    badge = {"pass": "PASS", "fail": "FAIL", "inconclusive": "INCONCLUSIVE"}[call]
    txt = (f"## chip call: **{badge}**\n"
           f"- posterior P(chip is bad) = **{p_final:.2f}**  (confidence {max(p_final, 1-p_final):.2f})\n"
           f"- fields used: **{used} / {len(probs)}**{'  (early stop)' if stopped else '  (budget exhausted)'}\n"
           f"- model card: field acc {MODEL_CARD['field_accuracy']}, "
           f"chip acc among confident calls {MODEL_CARD['chip_accuracy_among_confident']}, "
           f"false-confident {MODEL_CARD['false_confident_rate']}")
    report = json.dumps(dict(call=call, p_bad=p_final, fields_used=used,
                             n_fields=len(probs), per_field=[round(p, 4) for p in probs],
                             model_card=MODEL_CARD), indent=1)
    return fig, txt, report


with gr.Blocks(title="Organ-on-a-chip QC") as demo:
    gr.Markdown(
        "# Organ-on-a-chip QC — chip-level decision from a few fields\n"
        "Feed in the brightfield fields of **one chip** (filenames in acquisition order). "
        "The tool reports a chip call, a posterior confidence, how many fields it needed, and a "
        "per-field QC map.\n\n"
        "**It is a research prototype**: 13% of confident calls are wrong on unseen chips, and we "
        "could not build a reliable OOD detector — see the repository README."
    )
    with gr.Row():
        with gr.Column(scale=1):
            src = gr.Radio([p.name for p in EXAMPLES] + ["upload your own"],
                           value=EXAMPLES[0].name if EXAMPLES else "upload your own", label="chip")
            up = gr.File(file_count="multiple", file_types=["image"], label="field images (one chip)")
            run = gr.Button("run QC", variant="primary")
            gr.Markdown("Examples bundled with attribution from the OOC Image Dataset "
                        "(zenodo.10203721). Full dataset not redistributed.")
        with gr.Column(scale=2):
            out_txt = gr.Markdown()
            out_plot = gr.Plot()
            out_json = gr.Code(label="chip_report.json", language="json")

    def _files(src_name, uploaded):
        if src_name != "upload your own" and uploaded is None:
            d = HERE / "examples" / src_name
            return [str(p) for p in sorted(d.glob("*.png"))]
        return uploaded or []

    run.click(lambda s, u: run_chip(_files(s, u)), [src, up], [out_plot, out_txt, out_json])
    demo.load(lambda s, u: run_chip(_files(s, u)),
              [src, up], [out_plot, out_txt, out_json])

if __name__ == "__main__":
    import os
    demo.launch(server_name=os.environ.get("DEMO_HOST", "127.0.0.1"),
                server_port=int(os.environ.get("DEMO_PORT", "7861")),
                share=bool(os.environ.get("GRADIO_SHARE")))
