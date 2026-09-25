#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image statistics of isolated bad runs, long bad runs and good fields (REPORT §4.5).

Runs are maximal contiguous blocks of `bad` labels in acquisition order. "Isolated" = a single bad field between good ones,
"long" = length ≥ 5. Statistics are those of recovery_test.py (256 px grey: mean absolute gradient
as a focus proxy, dark fraction < 0.15, standard deviation as contrast).

Writes results/run_image_groups.json
"""
import collections, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leakage_experiment import index_images                   # noqa: E402
from recovery_test import img_stats, STATS_CACHE               # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"


def main():
    by = collections.defaultdict(list)
    for r in index_images():
        by[r["session"]].append(r)
    groups = dict(isolated_bad=[], long_bad=[], good=[])
    for v in by.values():
        v = sorted(v, key=lambda r: int(r["name"].split("_")[1].split(".")[0]))
        bad = [r["cls"] != "good" for r in v]
        i = 0
        while i < len(v):
            j = i
            while j < len(v) and bad[j] == bad[i]:
                j += 1
            paths = [str(r["path"]) for r in v[i:j]]
            if not bad[i]:
                groups["good"] += paths
            elif j - i == 1:
                groups["isolated_bad"] += paths
            elif j - i >= 5:
                groups["long_bad"] += paths
            i = j
    cache = json.loads(STATS_CACHE.read_text()) if STATS_CACHE.exists() else {}
    out = {}
    for g, ps in groups.items():
        st = [cache[p] if p in cache else img_stats(p) for p in ps]
        out[g] = dict(n=len(st), **{k: float(np.mean([s[k] for s in st])) for k in ("lap", "dark", "std", "mean")})
        print(g, {k: round(x, 3) if isinstance(x, float) else x for k, x in out[g].items()}, flush=True)
    (OUT / "run_image_groups.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
