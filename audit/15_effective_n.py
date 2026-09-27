# -*- coding: utf-8 -*-
"""Effective sample size of the 3,072 labels under three cluster-size conventions.

block_structure.py reports N / (1 + (m0 - 1) ICC) with m0 the ANOVA average cluster size (50.95).
Sessions range from 6 to 229 fields, so the design effect depends on how sessions are weighted:

  * pooled per-field metric (every field weighted equally): Kish's size-weighted cluster size
    m~ = sum n_i^2 / N  ->  deff = 1 + (m~ - 1) ICC                        (~80 effective labels)
  * ANOVA m0 (the value block_structure.py reports)                        (~181)
  * inverse-variance weighting of sessions: sum_i n_i / (1 + (n_i - 1) ICC) (~167)

Input: results/label_sufficiency.json (per-session good/bad counts). No images needed.
    python3 audit/15_effective_n.py   ->  results/effective_n.json
"""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"


def main():
    comp = json.loads((RES / "label_sufficiency.json").read_text())["session_composition"]
    n = np.array([v["n"] for v in comp.values()], dtype=float)
    bad = np.array([v["bad"] for v in comp.values()], dtype=float)
    N, k = n.sum(), len(n)
    p = bad / n
    pbar = bad.sum() / N
    msb = (n * (p - pbar) ** 2).sum() / (k - 1)
    msw = (n * p * (1 - p)).sum() / (N - k)
    m0 = (N - (n ** 2).sum() / N) / (k - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw)          # one-way ANOVA ICC(1), as block_structure.py
    m_kish = (n ** 2).sum() / N
    out = {
        "n_fields": int(N), "n_sessions": int(k),
        "session_size_min": int(n.min()), "session_size_max": int(n.max()),
        "icc_session": float(icc),
        "m0_anova": float(m0),
        "deff_anova": float(1 + (m0 - 1) * icc),
        "n_eff_anova": float(N / (1 + (m0 - 1) * icc)),
        "m_kish": float(m_kish),
        "deff_kish": float(1 + (m_kish - 1) * icc),
        "n_eff_kish": float(N / (1 + (m_kish - 1) * icc)),
        "n_eff_inverse_variance": float((n / (1 + (n - 1) * icc)).sum()),
        "note": "n_eff_kish applies to a pooled per-field metric (fields weighted equally); "
                "n_eff_anova is the value block_structure.py reports; the range is 80-181.",
    }
    (RES / "effective_n.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"ICC {icc:.3f} | N_eff: Kish {out['n_eff_kish']:.0f} (deff {out['deff_kish']:.1f}), "
          f"ANOVA m0 {out['n_eff_anova']:.0f} (deff {out['deff_anova']:.1f}), "
          f"inverse-variance {out['n_eff_inverse_variance']:.0f}")


if __name__ == "__main__":
    main()
