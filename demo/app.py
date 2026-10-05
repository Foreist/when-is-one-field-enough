#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gradio demo: chip-level QC from a handful of brightfield fields.

    python3 demo/app.py            # then open the printed local URL

Pick a bundled example chip or upload your own field images (one chip per run).
Measured numbers are on withheld half-session proxies, not physical-chip-validated;
there is no accuracy or imaging-time guarantee.
"""
import collections, json, sys
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
from inference import (MODEL_CARD, RESOURCE_BASIS, TF, load_model,  # noqa: E402
                       last_consumed_index, mark_qc_map, natural_key,
                       sequential_decision, short_chip_warning, validate_rule_args)

MODEL = load_model(str(ROOT / "model" / "perfield_mnv3s_384_s0.pt"))
EXAMPLES = sorted([p for p in (HERE / "examples").glob("*") if p.is_dir()])


def run_chip(files, max_fields=20, min_fields=8, conf=0.9):
    paths = sorted([Path(f) for f in files], key=natural_key)
    if not paths:
        return None, "upload at least one image (one chip per run)", ""
    try:
        max_fields, min_fields, conf = validate_rule_args(max_fields, min_fields, conf)
    except ValueError as e:
        return None, str(e), ""
    probs = []
    try:
        with torch.no_grad():
            for p in paths:
                x = TF(Image.open(p).convert("RGB")).unsqueeze(0)
                probs.append(float(torch.softmax(MODEL(x), dim=1)[0, 0]))
        n_scored = len(probs)
        dec = sequential_decision(probs, thr_conf=conf, max_fields=max_fields, min_fields=min_fields)
    except (OSError, ValueError, RuntimeError) as e:
        return None, (f"**QC failed ({type(e).__name__}). No chip result.** "
                      "Check the images/model and retry."), ""
    call, used, stopped, p_final = dec["call"], dec["n_fields"], dec["stopped"], dec["p_bad"]
    warn = short_chip_warning(n_scored, min_fields)

    fig, ax = plt.subplots(figsize=(11, 2.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.18)
    mark_qc_map(ax, probs, dec["field_indices"], vline_lw=1.4)
    last = last_consumed_index(dec["field_indices"])
    tx, ha = (last if last is not None else used) + 0.3, "left"
    if tx > n_scored - 4:                      # keep the label inside the axes
        tx, ha = n_scored - 0.5, "right"
    ax.text(tx, 1.04, f"rule used {used} of {n_scored} scored", fontsize=8, ha=ha, va="top")
    ax.set_ylim(0, 1.12); ax.set_xlabel("field index (acquisition order)", fontsize=9)
    ax.set_ylabel("P(bad)", fontsize=9)
    ax.tick_params(labelsize=8)

    badge = {"pass": "PASS", "fail": "FAIL", "inconclusive": "INCONCLUSIVE"}[call]
    txt = (f"## chip call: **{badge}**\n"
           f"- posterior P(chip is bad) = **{p_final:.2f}**  (confidence {max(p_final, 1-p_final):.2f})\n"
           f"- fields the rule consumed: **{used} / {n_scored} scored**"
           f"{'  (early stop)' if stopped else '  (budget exhausted)'} — policy consumption, not measured time\n"
           f"- model card: field acc {MODEL_CARD['field_accuracy']}, "
           f"chip acc among called chips {MODEL_CARD['chip_accuracy_among_confident']}, "
           f"false-confident {MODEL_CARD['false_confident_rate']} "
           f"(min_fields 8 was tuned on the test chips; untuned min 1: 0.708 on 24 calls, 0.680 with every chip called; "
           f"half-session proxies, not physical-chip-validated; no accuracy/time guarantee)")
    if warn:
        txt += f"\n\n**warning:** {warn}"
    report = json.dumps(dict(call=call, p_bad=p_final, fields_used=used,
                             n_fields=n_scored, n_fields_scored=n_scored,
                             field_indices=list(dec["field_indices"]),
                             resource_basis=RESOURCE_BASIS, warning=warn,
                             per_field=[round(p, 4) for p in probs],
                             model_card=MODEL_CARD), indent=1)
    return fig, txt, report


def run_plate(files):
    """files: list of paths from a directory upload; one subfolder per chip."""
    if not files:
        return "upload a folder that contains one subfolder per chip", None
    groups = collections.defaultdict(list)
    for f in files:
        p = Path(f)
        groups[p.parent.name].append(str(p))
    rows = []
    tot_avail = tot_used = 0
    warns = []
    for chip, paths in sorted(groups.items()):
        probs = []
        try:
            with torch.no_grad():
                for p in sorted(paths, key=lambda q: natural_key(Path(q))):
                    x = TF(Image.open(p).convert("RGB")).unsqueeze(0)
                    probs.append(float(torch.softmax(MODEL(x), dim=1)[0, 0]))
            n_scored = len(probs)
            dec = sequential_decision(probs)              # shipped settings: conf 0.9, min 8, max 20
        except (OSError, ValueError, RuntimeError) as e:
            return (f"**QC failed ({type(e).__name__}). No plate result.** "
                    "Check every chip's images/model and retry."), None
        call, used, p_final = dec["call"], dec["n_fields"], dec["p_bad"]
        tot_avail += n_scored; tot_used += used
        warn = short_chip_warning(n_scored, 8)
        if warn:
            warns.append(f"{chip}: {warn}")
        rows.append([chip, call, round(p_final, 3), round(max(p_final, 1 - p_final), 3),
                     f"{used}/{n_scored}", warn or ""])
    order_rank = {"fail": 0, "inconclusive": 1, "pass": 2}
    rows.sort(key=lambda r: (order_rank[r[1]], -r[2]))
    txt = (f"**{len(rows)} chips · {tot_used}/{tot_avail} fields used by the rule "
           f"({tot_avail} scored; {100*(1-tot_used/max(1,tot_avail)):.0f}% policy consumption, "
           f"not measured time)** — sorted by attention needed. "
           f"Half-session proxies; no accuracy/time guarantee.")
    if warns:
        txt += "\n\n" + "\n".join(f"- {w}" for w in warns)
    return txt, rows


with gr.Blocks(title="Organ-on-a-chip QC") as demo:
    gr.Markdown(
        "# Organ-on-a-chip QC — chip-level decision from a few fields\n"
        "Feed in the brightfield fields of **one chip** (filenames in acquisition order). "
        "The tool reports a chip call, a posterior confidence, how many fields the rule consumed, and a "
        "per-field QC map.\n\n"
        "**It is a research prototype**: 4 of 23 calls are wrong on unseen half-session proxies (3 of them confidently), "
        "not physical-chip-validated, and we could not build a reliable OOD detector — see the repository README. "
        "No accuracy or imaging-time guarantee. The app scores every uploaded field for the map; "
        "`fields_used` is sequential-rule consumption, not measured time."
    )
    with gr.Tab("single chip"):
        with gr.Row():
            with gr.Column(scale=1):
                default_chip = ("good_chip" if (HERE / "examples" / "good_chip").is_dir()
                                else (EXAMPLES[0].name if EXAMPLES else "upload your own"))
                src = gr.Radio([p.name for p in EXAMPLES] + ["upload your own"],
                               value=default_chip, label="chip")
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

    with gr.Tab("plate triage"):
        gr.Markdown("Upload a folder that contains **one subfolder per chip** (each subfolder holds "
                    "that chip's field images). Chips are ranked by how much attention they need. "
                    "All fields are scored; the field count is rule consumption, not time saved.")
        plate_up = gr.File(file_count="directory", file_types=["image"], label="plate folder")
        plate_btn = gr.Button("triage plate", variant="primary")
        plate_txt = gr.Markdown()
        plate_tbl = gr.Dataframe(headers=["chip", "call", "P(bad)", "confidence", "fields used", "warning"],
                                 interactive=False)
        plate_btn.click(run_plate, [plate_up], [plate_txt, plate_tbl])

if __name__ == "__main__":
    import os
    demo.launch(server_name=os.environ.get("DEMO_HOST", "127.0.0.1"),
                server_port=int(os.environ.get("DEMO_PORT", "7861")),
                share=bool(os.environ.get("GRADIO_SHARE")))
