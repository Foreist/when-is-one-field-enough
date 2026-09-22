#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate every figure in the report from results/*.json (reproducible).

    python3 figures.py        ->  figures/*.png
"""
import collections, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE / "results"
F = HERE / "figures"
F.mkdir(exist_ok=True)


def load(name):
    return json.loads((R / name).read_text())


def fig_dataset():
    d = load("label_sufficiency.json")
    comp = d["session_composition"]
    sizes = np.array([v["n"] for v in comp.values()])
    badf = np.array([v["bad"] / v["n"] for v in comp.values()])
    order = np.argsort(-badf)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].hist(sizes, bins=20, color="#4a6fa5")
    ax[0].set_xlabel("fields per session"); ax[0].set_ylabel("sessions")
    ax[0].set_title(f"(a) 59 sessions, {sizes.sum()} fields  (median {int(np.median(sizes))})", fontsize=9, loc="left")
    ax[1].bar(np.arange(len(badf)), badf[order], color="#c0392b", width=1.0)
    ax[1].set_xlabel("session (sorted)"); ax[1].set_ylabel("share of fields labelled bad")
    ax[1].axhline(np.mean(badf), color="k", ls="--", lw=1, label=f"mean {np.mean(badf):.2f}")
    ax[1].legend(fontsize=8)
    ax[1].set_title("(b) expert 'bad' share varies from 0% to 100% per session", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(F / "fig1_dataset.png", dpi=150); plt.close(fig)


def fig_leakage():
    c = load("leakage_controlled.json")
    l = load("leakage_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    x = np.arange(2)
    ax[0].bar(x - 0.18, [c["disjoint"]["acc"], c["leaky"]["acc"]], width=0.34,
              label="accuracy", color=["#7fb3d5", "#c0392b"])
    ax[0].bar(x + 0.18, [c["disjoint"]["auc"], c["leaky"]["auc"]], width=0.34,
              label="AUC", color=["#a9cce3", "#e6b0aa"])
    for i, (a, b) in enumerate(zip([c["disjoint"]["acc"], c["disjoint"]["auc"]],
                                   [c["leaky"]["acc"], c["leaky"]["auc"]])):
        ax[0].text(i - 0.18, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax[0].text(i + 0.18, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["session-disjoint\n(no leakage)", "same sessions\nin train (leaky)"])
    ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8, loc="lower right")
    ax[0].set_title(f"(a) controlled A/B: +{100*c['inflation_acc']:.1f} pp accuracy, "
                    f"+{100*c['inflation_auc']:.1f} pp AUC", fontsize=9, loc="left")
    ax[0].set_ylabel("test score (same test images, same training size)")
    tags = ["published\nsplit", "session-\ngrouped"]
    vals = [l["published"]["acc_mean"], l["cellaware_grouped"]["acc_mean"]]
    errs = [l["published"]["acc_sd"], l["cellaware_grouped"]["acc_sd"]]
    ax[1].bar(tags, vals, yerr=errs, color=["#c0392b", "#4a6fa5"], capsize=4)
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("accuracy")
    ax[1].set_title("(b) as shipped vs session-grouped (all 6 cell lines kept)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(F / "fig2_leakage.png", dpi=150); plt.close(fig)


def fig_label_structure():
    b = load("block_structure.json")
    s = load("structure_probe.json")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    z = [r["z"] for r in s["A"]]
    ax[0].hist(z, bins=18, color="#4a6fa5")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel("runs-test z (negative = clustered)"); ax[0].set_ylabel("sessions")
    ax[0].set_title(f"(a) {s['n_significant']}/{len(z)} sessions p<0.05,\n"
                    f"{s['n_clustered']}/{len(z)} clustered", fontsize=9, loc="left")
    ax[1].bar(["observed", "i.i.d.\nexpectation"], [b["runlen_mean"], b["runlen_iid_expected"]],
              color=["#c0392b", "#95a5a6"])
    for i, v in enumerate([b["runlen_mean"], b["runlen_iid_expected"]]):
        ax[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    ax[1].set_ylabel("mean run length (fields)")
    ax[1].set_title("(b) bad runs are 3x longer than chance", fontsize=9, loc="left")
    icc, deff, neff = b["icc_session"], b["deff_session"], b["n_eff_session"]
    ax[2].bar(["ICC", "design\neffect / 20", "effective N\n/ 1000"],
              [icc, deff / 20, neff / 1000], color=["#4a6fa5", "#c0392b", "#c0392b"])
    for i, v in enumerate([icc, deff / 20, neff / 1000]):
        ax[2].text(i, v + 0.01, f"{[icc, deff, neff][i]:.3g}", ha="center", fontsize=9)
    ax[2].set_title("(c) 3,072 labels carry ~181\nindependent observations", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(F / "fig3_label_structure.png", dpi=150); plt.close(fig)


def fig_sampling():
    a = load("adaptive_sampling.json")
    ks = sorted(int(k) for k in a)
    pol = ["random", "window", "adaptive"]
    col = {"random": "#4a6fa5", "window": "#95a5a6", "adaptive": "#c0392b"}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    for p in pol:
        ax[0].plot(ks, [a[str(k)][p]["chip_acc"] for k in ks], "o-", color=col[p], label=p)
        ax[1].plot(ks, [a[str(k)][p]["bad_recall"] for k in ks], "o-", color=col[p], label=p)
        ax[2].plot(ks, [a[str(k)][p]["frac_err"] for k in ks], "o-", color=col[p], label=p)
    ax[0].set_ylabel("chip-call accuracy"); ax[1].set_ylabel("bad-region recall")
    ax[2].set_ylabel("|estimated - true bad fraction|")
    for i, t in enumerate(["(a) chip call: random wins", "(b) localisation: adaptive wins",
                           "(c) fraction estimate: random wins"]):
        ax[i].set_xlabel("fields sampled (budget)"); ax[i].set_title(t, fontsize=9, loc="left")
        ax[i].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "fig4_sampling_policy.png", dpi=150); plt.close(fig)


def fig_tool():
    t = load("tool_evaluation.json")
    seq = t["chip_sequential_by_min_fields"]
    mins = sorted(int(k) for k in seq)
    fixed = t["chip_fixed_k_spread"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    ax[0].plot(mins, [seq[str(m)]["chip_acc_among_confident"] for m in mins], "o-",
               color="#4a6fa5", label="sequential (spread fields)")
    ax[0].plot([int(k) for k in fixed], [fixed[k] for k in fixed], "s--",
               color="#95a5a6", label="fixed k")
    ax[0].set_xlabel("minimum fields before a stop is allowed / fixed budget")
    ax[0].set_ylabel("chip accuracy"); ax[0].legend(fontsize=8); ax[0].set_ylim(0.6, 0.9)
    ax[0].set_title("(a) chip-level accuracy", fontsize=9, loc="left")
    ax[1].plot(mins, [100 * seq[str(m)]["false_confident_rate"] for m in mins], "o-", color="#c0392b")
    ax[1].axhline(10, color="k", ls="--", lw=1, label="nominal 10%")
    ax[1].set_xlabel("minimum fields"); ax[1].set_ylabel("% confident but wrong")
    ax[1].legend(fontsize=8); ax[1].set_title("(b) false-confident calls", fontsize=9, loc="left")
    ax[2].plot(mins, [seq[str(m)]["mean_fields"] for m in mins], "o-", color="#4a6fa5")
    ax[2].axhline(20, color="k", ls="--", lw=1, label="budget 20")
    ax[2].set_xlabel("minimum fields"); ax[2].set_ylabel("mean fields used")
    ax[2].legend(fontsize=8); ax[2].set_title("(c) imaging cost", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(F / "fig5_tool.png", dpi=150); plt.close(fig)


def fig_label_vs_model():
    st = load("stopping_rule.json")
    t = load("tool_evaluation.json")
    seq = t["chip_sequential_by_min_fields"]["8"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    labels = ["label-only\nsimulation", "deployed model\n(25 unseen chips)"]
    ax[0].bar(labels, [st["naive"]["mean_fields"], seq["mean_fields"]], color=["#95a5a6", "#c0392b"])
    for i, v in enumerate([st["naive"]["mean_fields"], seq["mean_fields"]]):
        ax[0].text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    ax[0].set_ylabel("mean fields used"); ax[0].set_ylim(0, 12)
    ax[0].set_title("(a) imaging cost", fontsize=9, loc="left")
    ax[1].bar(labels, [100 * st["naive"]["accuracy"], 100 * seq["chip_acc_among_confident"]],
              color=["#95a5a6", "#c0392b"])
    for i, v in enumerate([100 * st["naive"]["accuracy"], 100 * seq["chip_acc_among_confident"]]):
        ax[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax[1].set_ylabel("chip accuracy (%)"); ax[1].set_ylim(0, 100)
    ax[1].set_title("(b) accuracy", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(F / "fig6_label_vs_model.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_dataset(); fig_leakage(); fig_label_structure(); fig_sampling(); fig_tool(); fig_label_vs_model()
    for p in sorted(F.glob("*.png")):
        print("wrote", p.relative_to(HERE))
