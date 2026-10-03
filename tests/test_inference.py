#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU-free tests for sequential indices, QC-map markers, and default CLI plate (44 fields)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from inference import (  # noqa: E402
    last_consumed_index, mark_qc_map, sequential_decision, spread_order,
    validate_rule_args,
)

TMP = Path("/tmp/claude-ai4s-fixes-20260930/inference")
SELECTED_100_GOOD = [0, 5, 10, 16, 21, 26, 31, 36]


def _legacy_seq(probs, thr_conf=0.9, max_fields=20, min_fields=8):
    """Pre-fix evaluate.py loop — metric definitions must match sequential_decision."""
    from inference import beta_p_bad
    order_idx = spread_order(len(probs), max_fields)
    bad = good = 0
    call = None
    for i, idx in enumerate(order_idx, 1):
        if probs[idx] > 0.5:
            bad += 1
        else:
            good += 1
        pb = beta_p_bad(bad, good)
        if i >= min_fields and (pb > thr_conf or (1 - pb) > thr_conf):
            call = int(pb > 0.5)
            return dict(n_fields=i, p_bad=pb, call="fail" if call else "pass",
                        stopped=True, field_indices=order_idx[:i])
    pb = beta_p_bad(bad, good)
    if 0.35 < pb < 0.65:
        return dict(n_fields=len(order_idx), p_bad=pb, call="inconclusive",
                    stopped=False, field_indices=order_idx)
    call = int(pb > 0.5)
    return dict(n_fields=len(order_idx), p_bad=pb, call="fail" if call else "pass",
                stopped=False, field_indices=order_idx)


class SpreadAndDecision(unittest.TestCase):
    def test_n100_all_good_selected_indices(self):
        probs = [0.1] * 100
        dec = sequential_decision(probs)
        self.assertEqual(dec["field_indices"], SELECTED_100_GOOD)
        self.assertEqual(dec["n_fields"], 8)
        self.assertEqual(dec["call"], "pass")
        self.assertTrue(dec["stopped"])
        self.assertEqual(last_consumed_index(dec["field_indices"]), 36)
        self.assertNotEqual(dec["n_fields"] - 0.5, 36)

    def test_spread_order_default_grid(self):
        self.assertEqual(spread_order(100, 20)[:8], SELECTED_100_GOOD)
        self.assertEqual(spread_order(12, 20), list(range(12)))

    def test_validate_rule_args(self):
        validate_rule_args(20, 8, 0.9)
        with self.assertRaises(ValueError):
            validate_rule_args(1, 8, 0.9)
        with self.assertRaises(ValueError):
            validate_rule_args(20, 0, 0.9)
        with self.assertRaises(ValueError):
            validate_rule_args(20, 8, 0.5)
        with self.assertRaises(ValueError):
            validate_rule_args(20, 8, 1.0)

    def test_spread_order_max1_no_divzero(self):
        # audit callers may pass k=1; CLI rejects it. Must not divide by zero.
        self.assertEqual(spread_order(100, 1), [0])

    def test_sequential_matches_legacy_metric_loop(self):
        cases = [
            [0.1] * 100,
            [0.9] * 12,
            [0.4, 0.6] * 10,
            [0.2] * 5,
            [0.51] * 20 + [0.1] * 80,
        ]
        for probs in cases:
            for min_f in (1, 8, 12):
                got = sequential_decision(probs, min_fields=min_f)
                old = _legacy_seq(probs, min_fields=min_f)
                self.assertEqual(got["call"], old["call"])
                self.assertEqual(got["n_fields"], old["n_fields"])
                self.assertEqual(got["stopped"], old["stopped"])
                self.assertEqual(got["field_indices"], old["field_indices"])
                self.assertAlmostEqual(got["p_bad"], old["p_bad"])


class QcMapMarker(unittest.TestCase):
    def test_marker_uses_last_consumed_index_not_nfields_minus_half(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        probs = [0.1] * 100
        dec = sequential_decision(probs)
        fig, ax = plt.subplots()
        last = mark_qc_map(ax, probs, dec["field_indices"])
        self.assertEqual(last, 36)
        vlines = []
        selected_x = None
        for line in ax.lines:
            xd = [int(round(float(v))) for v in line.get_xdata()]
            if len(xd) == 2 and xd[0] == xd[1]:
                vlines.append(float(xd[0]))
            elif xd == SELECTED_100_GOOD:
                selected_x = xd
        self.assertIn(36.0, vlines)
        self.assertNotIn(7.5, vlines)
        self.assertEqual(selected_x, SELECTED_100_GOOD)
        plt.close(fig)


class EvaluateDataRoot(unittest.TestCase):
    def test_honors_supplied_root_and_restores_module_global(self):
        import evaluate as ev

        le = mock.MagicMock()
        le.DATA = Path("/env-default")
        seen = {}

        def index_images():
            seen["during"] = le.DATA
            return []

        le.index_images = index_images
        split = mock.Mock(return_value={"disjoint": {"test": []}})
        lc = mock.Mock(split_controlled=split)
        with mock.patch.dict(sys.modules, {"leakage_experiment": le, "leakage_controlled": lc}):
            by = ev.load_test_chips("/tmp/supplied-root-ai4s")
        self.assertEqual(seen["during"], Path("/tmp/supplied-root-ai4s"))
        self.assertEqual(le.DATA, Path("/env-default"))
        self.assertEqual(dict(by), {})
        self.assertIsNone(os.environ.get("OOC_DATA_TEST_SENTINEL"))


class DefaultPlateCli(unittest.TestCase):
    def test_demo_plate_scores_44_and_keeps_calls(self):
        ckpt = REPO / "model" / "perfield_mnv3s_384_s0.pt"
        examples = REPO / "demo" / "examples"
        if not ckpt.is_file() or not examples.is_dir():
            self.skipTest("checkpoint or demo examples unavailable")
        TMP.mkdir(parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="plate44-", dir=str(TMP)))
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        proc = subprocess.run(
            [sys.executable, str(REPO / "inference.py"),
             "--plate", str(examples), "--out", str(out)],
            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(proc.returncode, 0, f"CLI failed: {proc.stderr[-800:]}")
        report = json.loads((out / "plate_report.json").read_text())
        self.assertEqual(report["summary"]["n_fields_scored"], 44)
        self.assertEqual(report["summary"]["fields_available"], 44)
        self.assertIn("resource_basis", report["summary"])
        calls = {r["chip"]: r["call"] for r in report["chips"]}
        self.assertEqual(calls.get("good_chip"), "pass")
        self.assertEqual(calls.get("bad_chip"), "fail")
        self.assertEqual(calls.get("borderline_chip"), "pass")
        used = {r["chip"]: r["fields_used"] for r in report["chips"]}
        self.assertEqual(used.get("good_chip"), 8)
        self.assertEqual(used.get("bad_chip"), 11)
        self.assertEqual(used.get("borderline_chip"), 20)
        for r in report["chips"]:
            self.assertIn("field_indices", r)
            self.assertEqual(r["n_fields_scored"], r["fields_available"])


if __name__ == "__main__":
    unittest.main()
