# -*- coding: utf-8 -*-
"""Look for hidden structure in the OOC Image Dataset beyond "binary QC".

A) positional/acquisition structure: is the good/bad label autocorrelated within a session?
B) metadata association: label ~ day + seeding density + flow rate + cell line (observational),
   and per-cell-line bad rates over all fields
(image statistics are in recovery_test.py and 04b_run_image_stats.py)
"""
import collections, json, os, math, re
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("OOC_DATA", ROOT / "data" / "OOC_image_dataset"))
SHEET = DATA.parent / "OOC_datasheet.xlsx"            # downloaded next to the image folder
OUT = Path(__file__).resolve().parent.parent / "results"
PAT = re.compile(r"^(\d{6})_(\d+)\.png$")


def load():
    recs = {}
    for p in DATA.rglob("*.png"):
        parts = p.relative_to(DATA).parts
        m = PAT.match(parts[4])
        if not m:
            continue
        recs[f"{m.group(1)}_{int(m.group(2)):02d}"] = dict(
            path=str(p), split=parts[0], cls=parts[1], cell=parts[2], day=parts[3],
            session=m.group(1), idx=int(m.group(2)))
    wb = load_workbook(SHEET, read_only=True, data_only=True)
    ws = wb["Main"]; it = ws.iter_rows(values_only=True)
    hdr = list(next(it)); meta = {}
    for r in it:
        if not r or r[0] is None:
            continue
        key = str(r[0]).strip()
        meta[key] = dict(density=r[2], t_after=r[3], day=r[4], label=r[5], flow=r[6])
    wb.close()
    return recs, meta


def parse_density(x):
    if x is None:
        return None
    s = str(x).replace(",", "").strip()
    try:
        v = float(s)
        return v if v > 0 else None
    except Exception:
        return None


def parse_flow(x):
    if x is None:
        return None
    s = str(x).lower().replace("ul/min", "").replace(" ", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def runs_test(labels):
    """Wald-Wolfowitz runs test. Returns (n_runs, expected, z, p_two_sided)."""
    n1 = sum(labels); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0 or len(labels) < 4:
        return None
    runs = 1
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            runs += 1
    n = n1 + n0
    mu = 2 * n1 * n0 / n + 1
    var = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n * n * (n - 1))
    if var <= 0:
        return None
    z = (runs - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return runs, mu, z, p


def main():
    recs, meta = load()
    # attach metadata by imageID (datasheet uses YYMMDD_NN; zip name YYMMDD_N)
    joined = []
    miss = 0
    for k, r in recs.items():
        sess, idx = r["session"], r["idx"]
        cand = [f"{sess}_{idx:02d}", f"{sess}_{idx}"]
        m = None
        for c in cand:
            if c in meta:
                m = meta[c]; break
        if m is None:
            miss += 1
            continue
        joined.append(dict(**r, meta_day=m.get("day"), density=m.get("density"), t_after=m.get("t_after"), label=m.get("label"), flow=m.get("flow")))
    print(f"images {len(recs)}  joined {len(joined)}  unmatched {miss}")

    # ---------- A) acquisition-order structure ----------
    by_sess = collections.defaultdict(list)
    for r in joined:
        by_sess[r["session"]].append(r)
    res_A = []
    for s, v in sorted(by_sess.items()):
        v = sorted(v, key=lambda r: r["idx"])
        labs = [1 if r["cls"] == "good" else 0 for r in v]
        if len(set(labs)) < 2:
            continue
        rt = runs_test(labs)
        if rt:
            res_A.append(dict(session=s, n=len(labs), good_frac=round(float(np.mean(labs)), 3),
                              runs=rt[0], expected=round(rt[1], 1), z=round(rt[2], 2), p=round(rt[3], 4)))
    n_sig = sum(1 for r in res_A if r["p"] < 0.05)
    n_clustered = sum(1 for r in res_A if r["z"] < 0)
    print(f"\n[A] sessions tested {len(res_A)}  p<0.05 {n_sig}  clustered(z<0) {n_clustered}")
    for r in sorted(res_A, key=lambda x: x["p"])[:6]:
        print("   ", r)

    # ---------- B) metadata association ----------
    rows = []
    for r in joined:
        d = parse_density(r["density"]); f = parse_flow(r["flow"])
        day = None
        try:
            day = int(str(r["meta_day"]))
        except Exception:
            pass
        if d is None or day is None:
            continue
        rows.append(dict(y=1 if r["cls"] == "good" else 0, cell=r["cell"], day=day, session=r["session"],
                         density=d, flow=f if f is not None else np.nan))
    print(f"\n[B] rows with density+day: {len(rows)}  with flow: {sum(1 for r in rows if not np.isnan(r['flow']))}")
    # binned failure rate by day and by density tercile
    def rate(rs):
        return None if not rs else round(1 - np.mean([x["y"] for x in rs]), 3)
    by_day = collections.defaultdict(list)
    for r in rows:
        by_day[min(r["day"], 8)].append(r)     # cap at 8+ for display
    print("   bad-rate by day:", {k: (rate(v), len(v)) for k, v in sorted(by_day.items())})
    dens = np.array([r["density"] for r in rows])
    qs = np.quantile(dens, [0, 1/3, 2/3, 1])
    for i in range(3):
        sel = [r for r in rows if qs[i] <= r["density"] <= qs[i + 1]]
        print(f"   density tercile {i+1} ({qs[i]:.2e}~{qs[i+1]:.2e}): bad-rate {rate(sel)} n={len(sel)}")
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    print("   bad-rate by cell:", {k: (rate(v), len(v)) for k, v in sorted(by_cell.items())})
    # per-cell bad rates over ALL fields (the metadata join above drops some), and the range of
    # per-session bad rates within each cell line
    cell_all = collections.defaultdict(list)
    sess_rate = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in joined:
        b = r["cls"] != "good"
        cell_all[r["cell"]].append(b)
        sess_rate[r["cell"]][r["session"]].append(b)
    res_B = dict(n_rows_with_metadata=len(rows), n_fields_all=len(joined),
                 bad_rate_by_cell_all_fields={c: dict(bad_rate=float(np.mean(v)), n=len(v))
                                              for c, v in sorted(cell_all.items())},
                 session_bad_rate_range_by_cell={c: [float(min(np.mean(v) for v in ss.values())),
                                                     float(max(np.mean(v) for v in ss.values()))]
                                                 for c, ss in sorted(sess_rate.items())})
    # logistic regression (no sklearn dependency on categoricals beyond one-hot)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        cells = sorted({r["cell"] for r in rows})
        X = []
        for r in rows:
            x = [r["day"], math.log10(r["density"])]
            if not np.isnan(r["flow"]):
                x.append(r["flow"])
            else:
                x.append(np.nan)
            x += [1.0 if r["cell"] == c else 0.0 for c in cells]
            X.append(x)
        X = np.array(X, dtype=float)
        y = np.array([r["y"] for r in rows])
        mask = ~np.isnan(X).any(axis=1)
        Xf = X[mask]; yf = y[mask]
        sc = StandardScaler().fit(Xf)
        m = LogisticRegression(max_iter=3000).fit(sc.transform(Xf), yf)
        names = ["day", "log10 density", "flow"] + cells
        print("   logistic coefs (standardized, + = more likely GOOD):")
        for n, w in sorted(zip(names, m.coef_[0]), key=lambda t: -abs(t[1]))[:6]:
            print(f"      {n:<18} {w:+.3f}")
        print("   n used:", len(yf), " in-sample acc:", round(m.score(sc.transform(Xf), yf), 3))
        res_B.update(logistic_n=int(len(yf)), logistic_in_sample_acc=float(m.score(sc.transform(Xf), yf)))
    except Exception as e:
        print("   logistic skipped:", e)

    json.dump(dict(A=res_A, B=res_B, n_sessions_tested=len(res_A), n_significant=n_sig, n_clustered=n_clustered),
              open(OUT / "structure_probe.json", "w"), ensure_ascii=False, indent=1)
    print("\nsaved", OUT / "structure_probe.json")


if __name__ == "__main__":
    main()
