# -*- coding: utf-8 -*-
"""Can the decision rule (re-image vs discard) be validated with a constructed target?

Target  : a bad run "recovers" if the NEXT 5 fields are majority good.
          Runs that reach the session end get recovery = 0 (no recovery observed).
Features: available at decision time from the run itself —
          run length, reaches-session-end, focus (laplacian), dark fraction,
          fraction of run images that look like bubbles/blur.
Eval    : logistic regression, session-grouped 5-fold CV (no session spans folds),
          compared against a length-only baseline and a permutation null.
"""
import collections, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structure_probe import load

OUT = Path(__file__).resolve().parent
STATS_CACHE = OUT / "run_image_stats.json"
NEXT = 5


def img_stats(path):
    im = Image.open(path).convert("L").resize((256, 256))
    a = np.asarray(im, dtype=np.float32) / 255.0
    lap = float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())
    return dict(lap=lap, dark=float((a < 0.15).mean()), mean=float(a.mean()), std=float(a.std()))


def main():
    recs, meta = load()
    by = collections.defaultdict(list)
    for k, r in recs.items():
        by[r["session"]].append(r)
    runs = []
    for s, v in by.items():
        v = sorted(v, key=lambda r: r["idx"])
        y = np.array([1 if r["cls"] == "good" else 0 for r in v])
        n = len(y); i = 0
        while i < n:
            if y[i] == 0:
                j = i
                while j + 1 < n and y[j + 1] == 0:
                    j += 1
                after = y[j + 1:]
                rec = 0
                if len(after) >= NEXT:
                    rec = int(after[:NEXT].mean() >= 0.5)
                else:
                    rec = 0                       # includes runs that reach the end
                runs.append(dict(session=s, cell=v[i]["cell"], day=v[i].get("meta_day"),
                                 length=j - i + 1, start=i, end=j, n=n,
                                 reaches_end=int(j == n - 1),
                                 n_after=int(len(after)),
                                 recovery=rec, paths=[r["path"] for r in v[i:j + 1]]))
                i = j + 1
            else:
                i += 1
    print("runs:", len(runs), " recovery rate:", round(float(np.mean([r["recovery"] for r in runs])), 3))

    # image stats (cached)
    cache = {}
    if STATS_CACHE.exists():
        cache = json.loads(STATS_CACHE.read_text())
    need = [p for r in runs for p in r["paths"] if p not in cache]
    print("computing stats for", len(need), "images")
    for i, p in enumerate(need):
        cache[p] = img_stats(p)
        if (i + 1) % 300 == 0:
            print("  ", i + 1, flush=True)
    STATS_CACHE.write_text(json.dumps(cache))

    # feature table
    lap_all = np.array([cache[p]["lap"] for p in cache])
    dark_all = np.array([cache[p]["dark"] for p in cache])
    lap_thr = float(np.percentile(lap_all, 25))       # blurry = bottom quartile
    dark_thr = float(np.percentile(dark_all, 75))     # bubble-ish = top quartile
    rows = []
    for r in runs:
        st = [cache[p] for p in r["paths"]]
        lap = float(np.mean([x["lap"] for x in st]))
        dark = float(np.mean([x["dark"] for x in st]))
        std = float(np.mean([x["std"] for x in st]))
        frac_blur = float(np.mean([x["lap"] < lap_thr for x in st]))
        frac_dark = float(np.mean([x["dark"] > dark_thr for x in st]))
        rows.append(dict(session=r["session"], y=r["recovery"], length=r["length"],
                         reaches_end=r["reaches_end"], lap=lap, dark=dark, std=std,
                         frac_blur=frac_blur, frac_dark=frac_dark,
                         artifact_score=float(np.mean([(x["lap"] < lap_thr) or (x["dark"] > dark_thr) for x in st]))))
    y = np.array([r["y"] for r in rows])
    groups = np.array([r["session"] for r in rows])
    print("usable runs:", len(rows), " recovery:", int(y.sum()), "/", len(y))

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    def cv_auc(feats):
        X = np.array([[r[f] for f in feats] for r in rows], dtype=float)
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
            oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        return float(roc_auc_score(y, oof)) if len(set(y)) > 1 else float("nan")

    sets = {
        "length only": ["length"],
        "artifact signals only": ["lap", "dark", "frac_blur", "frac_dark"],
        "length + artifact": ["length", "reaches_end", "lap", "dark", "frac_blur", "frac_dark", "artifact_score"],
        "all": ["length", "reaches_end", "lap", "dark", "std", "frac_blur", "frac_dark", "artifact_score"],
    }
    res = {}
    for name, f in sets.items():
        a = cv_auc(f)
        res[name] = a
        print(f"  CV AUC [{name:<22}] {a:.3f}")

    # permutation null for the full set
    rng = np.random.default_rng(0)
    null = []
    X = np.array([[r[f] for f in sets["all"]] for r in rows], dtype=float)
    for _ in range(200):
        yp = rng.permutation(y)
        oof = np.zeros(len(yp))
        for tr, te in GroupKFold(n_splits=5).split(X, yp, groups):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), yp[tr])
            oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        null.append(roc_auc_score(yp, oof))
    null = np.array(null)
    print(f"  permutation null: mean {null.mean():.3f} p95 {np.percentile(null,95):.3f}")

    # simple rule check: length<=2 -> predict recover
    simple = float(np.mean([(r["length"] <= 2) == bool(r["y"]) for r in rows]))
    print(f"  simple rule (length<=2 => recover) accuracy {simple:.3f}  (base rate {y.mean():.3f})")
    (OUT / "recovery_test.json").write_text(json.dumps(dict(
        n_runs=len(rows), recovery_rate=float(y.mean()), cv_auc=res,
        null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
        simple_rule_acc=simple, lap_thr=lap_thr, dark_thr=dark_thr,
        feature_rows=[{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()} for r in rows],
    ), ensure_ascii=False, indent=1))
    print("saved", OUT / "recovery_test.json")


if __name__ == "__main__":
    main()
