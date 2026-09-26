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
    for i, arm in enumerate(["disjoint", "leaky"]):          # x=i: accuracy left, AUC right
        for dx, key in ((-0.18, "acc"), (0.18, "auc")):
            v = c[arm][key]
            ax[0].text(i + dx, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["session-disjoint\n(no leakage)", "same sessions\nin train (leaky)"])
    ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title(f"(a) controlled A/B: +{100*c['inflation_acc']:.1f} pp accuracy, "
                    f"+{100*c['inflation_auc']:.1f} pp AUC", fontsize=9, loc="left")
    ax[0].set_ylabel("test score")
    tags = ["published\nsplit", "session-\ngrouped"]
    vals = [l["published"]["acc_mean"], l["cellaware_grouped"]["acc_mean"]]
    errs = [l["published"]["acc_sd"], l["cellaware_grouped"]["acc_sd"]]
    ax[1].bar(tags, vals, yerr=errs, color=["#c0392b", "#4a6fa5"], capsize=4)
    for i, v in enumerate(vals):
        ax[1].text(i + 0.08, v + 0.02, f"{v:.3f}", ha="left", fontsize=9)
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
    a = load("adaptive_sampling.json")          # label-based simulation
    m = load("policy_model_in_loop.json")       # deployed model
    ks = sorted(int(k) for k in a)
    pol = ["random", "window", "adaptive"]
    col = {"random": "#4a6fa5", "window": "#95a5a6", "adaptive": "#c0392b"}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.3))
    for p in pol:
        ax[0].plot(ks, [a[str(k)][p]["chip_acc"] for k in ks], "o-", color=col[p], label=p)
        ax[0].plot(ks, [a[str(k)][p]["bad_recall"] for k in ks], "o--", color=col[p], alpha=.6)
    ax[0].set_ylabel("accuracy (solid) / recall (dashed)")
    ax[0].set_title("(a) simulated on ground-truth labels", fontsize=9, loc="left")
    ax[0].set_xlabel("fields sampled (budget)"); ax[0].legend(fontsize=8)
    mk = sorted(int(k) for k in m)
    for p in pol:
        ax[1].plot(mk, [m[str(k)][p]["chip_acc"] for k in mk], "o-", color=col[p], label=p)
        ax[1].plot(mk, [m[str(k)][p]["bad_recall"] for k in mk], "o--", color=col[p], alpha=.6)
    ax[1].set_title("(b) with the model in the loop (25 unseen chips)", fontsize=9, loc="left")
    ax[1].set_xlabel("fields sampled (budget)"); ax[1].legend(fontsize=8)
    ax[1].set_ylim(ax[0].get_ylim())
    # paired differences with bootstrap CIs (k=12)
    cis = load("policy_bootstrap.json")
    c12 = cis["k12"]; labels = list(c12.keys())
    vals = np.array([c12[k]["mean"] for k in labels]); y = np.arange(len(labels))
    for lo_k, hi_k, lw, cap in (("bonf_lo", "bonf_hi", 1, 3), ("lo", "hi", 3, 0)):
        lo = np.array([c12[k][lo_k] for k in labels]); hi = np.array([c12[k][hi_k] for k in labels])
        ax[2].errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="o", color="#333", lw=lw, capsize=cap)
    ax[2].axvline(0, color="k", ls="--", lw=1)
    ax[2].set_yticks(y); ax[2].set_yticklabels(labels, fontsize=8)
    ax[2].set_xlabel("paired accuracy difference (k=12)")
    ax[2].set_title(f"(c) thick 95%, thin Bonferroni x6: all corrected\nintervals include zero (n={cis['n_chips_k12']} chips)",
                    fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(F / "fig4_sampling_policy.png", dpi=150); plt.close(fig)


def fig_tool():
    t = load("tool_evaluation.json")
    seq = t["chip_sequential_by_min_fields"]
    mins = sorted(int(k) for k in seq)
    fixed = t["chip_fixed_k_spread"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    ax[0].plot(mins, [seq[str(m)]["chip_acc_among_confident"] for m in mins], "o-",
               color="#4a6fa5", label="sequential, on the chips it calls")
    ax[0].plot([int(k) for k in fixed], [fixed[k] for k in fixed], "s--",
               color="#95a5a6", label="cap of k spread fields, all 25 chips")
    ax[0].set_xlabel("minimum fields before a stop is allowed / cap k")
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


def fig_efficiency():
    e = load("efficiency.json")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fx = sorted(int(k) for k in e["fixed_k"])
    ax.plot([e["fixed_k"][str(k)]["fields"] for k in fx], [e["fixed_k"][str(k)]["acc"] for k in fx],
            "s--", color="#95a5a6", label="fixed k spread (only chips with >= k fields)")
    cx = sorted(int(k) for k in e["cap_k"])
    ax.plot([e["cap_k"][str(k)]["fields"] for k in cx], [e["cap_k"][str(k)]["acc"] for k in cx],
            "D:", color="#7f8c8d", label="cap of k spread fields (all chips)")
    ax.scatter([e["mean_fields_per_chip"]], [e["all_fields_acc"]], marker="*", s=260,
               color="#4a6fa5", label=f"all fields ({e['mean_fields_per_chip']:.0f}/chip)")
    for mode, col, mk in (("force", "#c0392b", "o"), ("abstain", "#e67e22", "^")):
        keys = [k for k in e["sequential"] if e["sequential"][k]["mode"] == mode]
        keys.sort(key=lambda k: e["sequential"][k]["fields"])
        ax.plot([e["sequential"][k]["fields"] for k in keys],
                [e["sequential"][k]["acc"] for k in keys], mk + "-", color=col,
                label=f"sequential ({mode})")
    ax.axhline(e["all_fields_acc"], color="#4a6fa5", ls=":", lw=1)
    ax.annotate("same accuracy,\n2.9x fewer fields", xy=(9.5, 0.80), xytext=(14, 0.70),
                arrowprops=dict(arrowstyle="->", color="k", lw=1), fontsize=9)
    ax.set_xlabel("fields used per chip"); ax.set_ylabel("chip-level accuracy")
    ax.set_ylim(0.6, 0.9); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Field efficiency on 25 unseen chips", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(F / "fig7_efficiency.png", dpi=150); plt.close(fig)


def fig_label_vs_model():
    r = load("label_vs_model.json")
    keys = [("min1_labels", "labels\nno min"), ("min1_model", "model\nno min"),
            ("min8_labels", "labels\nmin 8"), ("min8_model", "model\nmin 8 (shipped)")]
    cols = ["#95a5a6", "#c0392b", "#95a5a6", "#c0392b"]
    names = [n for _, n in keys]
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.7))
    acc = [100 * r[k]["accuracy"] for k, _ in keys]
    ax[0].bar(names, acc, color=cols)
    for i, k in enumerate(keys):
        ax[0].text(i, acc[i] + 1, f"{r[k[0]]['n_correct']}/{r[k[0]]['n_called']}", ha="center", fontsize=8)
    ax[0].set_ylabel("chip accuracy among calls (%)"); ax[0].set_ylim(0, 100)
    ax[0].set_title("(a) accuracy, same rule, same 25 chips", fontsize=9, loc="left")
    wc = [r[k]["n_false_confident"] for k, _ in keys]
    ax[1].bar(names, wc, color=cols)
    for i, v in enumerate(wc):
        ax[1].text(i, v + 0.1, str(v), ha="center", fontsize=9)
    ax[1].set_ylabel("confident-but-wrong chips"); ax[1].set_ylim(0, 8)
    ax[1].set_title("(b) confident errors", fontsize=9, loc="left")
    for a in ax:
        a.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(F / "fig6_label_vs_model.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_dataset(); fig_leakage(); fig_label_structure(); fig_sampling(); fig_tool(); fig_label_vs_model()
    fig_efficiency()
    for p in sorted(F.glob("*.png")):
        print("wrote", p.relative_to(HERE))
