#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated tests for fail-closed claim checks.

Expected literals come from results/*.json via check_claims.expected_map —
tests do not hardcode headline values.
"""
from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import check_claims  # noqa: E402
import check_numbers  # noqa: E402


def _texts():
    ledger = check_claims.load_ledger(HERE / "claims.json")
    return {k: v for k, v in check_claims.read_texts(ledger, HERE).items() if v is not None}


class ClaimsPass(unittest.TestCase):
    def test_current_docs_pass(self):
        issues = check_claims.check_claims(here=HERE)
        self.assertEqual(issues, [], "\n".join(issues))


class IsolatedPaths(unittest.TestCase):
    def test_missing_optional_doc_is_not_green_by_assumption(self):
        ledger = check_claims.load_ledger(HERE / "claims.json")
        texts = _texts()
        texts.pop("writeup", None)
        # Pretend writeup was never listed as present: pass the remaining docs.
        issues = check_claims.check_claims(here=HERE, ledger=ledger, texts=texts)
        self.assertEqual(issues, [], "\n".join(issues))

    def test_required_doc_missing_fails(self):
        ledger = check_claims.load_ledger(HERE / "claims.json")
        texts = _texts()
        texts["report"] = None
        issues = check_claims.check_claims(here=HERE, ledger=ledger, texts=texts)
        self.assertTrue(any("required doc missing" in i or "no listed document" in i for i in issues))

    def test_explicit_optional_path_absent_skips(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "results").mkdir()
            for name in (
                "tool_evaluation.json",
                "efficiency.json",
                "vs_all_pass.json",
                "leakage_ci.json",
                "block_structure.json",
                "effective_n.json",
                "label_vs_model.json",
                "lolo_cellline.json",
            ):
                (root / "results" / name).write_text((HERE / "results" / name).read_text())
            ledger = check_claims.load_ledger(HERE / "claims.json")
            # Required docs exist with real text; optional writeup path is explicit and absent.
            (root / "REPORT.md").write_text((HERE / "REPORT.md").read_text())
            (root / "README.md").write_text((HERE / "README.md").read_text())
            (root / "audit").mkdir()
            (root / "audit" / "README.md").write_text((HERE / "audit" / "README.md").read_text())
            (root / "claims.json").write_text(json.dumps(ledger))
            issues = check_claims.check_claims(here=root, ledger=ledger, results_dir=root / "results")
            self.assertEqual(issues, [], "\n".join(issues))


class Mutations(unittest.TestCase):
    def test_one_19_23_mutation_fails_while_other_826_remain(self):
        ledger = check_claims.load_ledger(HERE / "claims.json")
        texts = _texts()
        report = texts["report"]
        n_before = report.count("0.826")
        self.assertGreaterEqual(n_before, 2)
        mutated, n = re.subn(r"19/23\s*=\s*0\.826", "19/23 = 0.827", report, count=1)
        self.assertEqual(n, 1)
        self.assertEqual(mutated.count("0.826"), n_before - 1)
        self.assertIn("0.826", mutated)
        texts["report"] = mutated
        issues = check_claims.check_claims(here=HERE, ledger=ledger, texts=texts)
        self.assertTrue(
            any("chip_acc_min8_called" in i and "0.827" in i for i in issues),
            issues,
        )

    def test_at_least_seven_headline_mutations_detected(self):
        reports = check_claims.run_mutations(here=HERE)
        detected = [r for r in reports if r["detected"]]
        self.assertGreaterEqual(
            len(detected),
            7,
            f"only {len(detected)} mutations detected: {reports}",
        )
        # PDF is not involved; these are in-memory text edits.
        self.assertTrue(all("pdf" not in r["doc"].lower() for r in reports))

    def test_wording_edit_without_number_change_still_passes(self):
        ledger = check_claims.load_ledger(HERE / "claims.json")
        texts = _texts()
        texts["report"] = texts["report"].replace(
            "A session-grouped, chip-level protocol",
            "A session-grouped (rephrased) chip-level protocol",
            1,
        )
        issues = check_claims.check_claims(here=HERE, ledger=ledger, texts=texts)
        self.assertEqual(issues, [], "\n".join(issues))

    def test_zero_anchor_fails_required_rule(self):
        ledger = check_claims.load_ledger(HERE / "claims.json")
        texts = _texts()
        texts["report"] = re.sub(r"19/23\s*=\s*0\.\d+", "nineteen of twenty-three", texts["report"])
        issues = check_claims.check_claims(here=HERE, ledger=ledger, texts=texts)
        self.assertTrue(any("report_19_23" in i and ("0 matches" in i or "need" in i) for i in issues), issues)

    def test_wrong_count_fails(self):
        ledger = copy.deepcopy(check_claims.load_ledger(HERE / "claims.json"))
        # Force exact=2 on the unique 19/23 anchor.
        for c in ledger["claims"]:
            if c["id"] == "chip_acc_min8_called":
                for a in c["anchors"]:
                    if a["id"] == "report_19_23":
                        a["exact"] = 2
        issues = check_claims.check_claims(here=HERE, ledger=ledger, texts=_texts())
        self.assertTrue(any("report_19_23" in i and "exact 2" in i for i in issues), issues)


class PdfSidecar(unittest.TestCase):
    def test_missing_manifest_warns_not_fail(self):
        with tempfile.TemporaryDirectory() as td:
            # Point helper at a temp HERE by monkeypatching module constants is
            # heavier than calling with explicit paths.
            issues = check_numbers.check_pdf_sources(
                pdf=Path(td) / "report.pdf",
                manifest_path=Path(td) / "report.sources.json",
            )
            self.assertEqual(issues, [])

    def test_drifted_source_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "REPORT.md").write_text("x")
            (root / "make_report_pdf.py").write_text("y")
            (root / "figures").mkdir()
            (root / "report.pdf").write_bytes(b"%PDF")
            man = {
                "sources": {
                    "REPORT.md": "0" * 64,
                    "make_report_pdf.py": check_numbers.sha256_file(root / "make_report_pdf.py"),
                },
                "pdf_sha256": check_numbers.sha256_file(root / "report.pdf"),
            }
            (root / "report.sources.json").write_text(json.dumps(man))
            old_here = check_numbers.HERE
            try:
                check_numbers.HERE = root
                issues = check_numbers.check_pdf_sources(
                    pdf=root / "report.pdf",
                    manifest_path=root / "report.sources.json",
                )
            finally:
                check_numbers.HERE = old_here
            self.assertTrue(any("drifted" in i for i in issues), issues)


if __name__ == "__main__":
    unittest.main()
