#!/usr/bin/env python3
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("evaluation", ROOT / "evaluate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvaluationEdges(unittest.TestCase):
    def test_cli_parameters_reject_invalid_values_before_io(self):
        for args in [(0, 20, .9), (-1, 20, .9), (32, 0, .9),
                     (32, 1, .9), (32, -1, .9), (32, 20, .5), (32, 20, 1),
                     (32, 20, float("nan"))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                module.validate_cli_parameters(*args)

    def test_batch_zero_main_rejects_before_data_or_model_io(self):
        with mock.patch.object(sys, "argv", ["evaluate.py", "--batch", "0"]), \
             mock.patch.object(module, "load_test_chips") as load_data, \
             mock.patch.object(module, "mobilenet_v3_small") as make_model, \
             self.assertRaises(SystemExit):
            module.main()
        load_data.assert_not_called()
        make_model.assert_not_called()

    def test_valid_cli_parameters_and_min_above_budget_policy(self):
        module.validate_cli_parameters(1, 2, .9)
        # min_fields belongs to the internal stopping rule and may exceed max_fields.
        out = module.sequential_decision([.1,.9], max_fields=2, min_fields=8)
        self.assertEqual(out["n_fields"], 2)

    def test_all_abstain_producer_writes_strict_json_nulls(self):
        class FakeModel:
            def __init__(self):
                self.classifier = [None, None, None, mock.Mock(in_features=1)]
            def load_state_dict(self, state): pass
            def eval(self): return self
            def __call__(self, batch):
                probs = torch.tensor([.1, .9, .1, .9])[:len(batch)]
                return torch.stack((torch.log(probs), torch.log1p(-probs)), dim=1)

        records = [dict(path=f"field-{i}.png", cls="bad" if i % 2 == 0 else "good")
                   for i in range(4)]
        fake_by = {"proxy": records}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "results").mkdir()
            with mock.patch.object(sys, "argv", ["evaluate.py", "--batch", "4", "--max-fields", "2"]), \
                 mock.patch.object(module, "HERE", root), \
                 mock.patch.object(module, "load_test_chips", return_value=fake_by), \
                 mock.patch.object(module, "mobilenet_v3_small", return_value=FakeModel()), \
                 mock.patch.object(module.nn, "Linear", return_value=mock.Mock()), \
                 mock.patch.object(module.torch, "load", return_value={}), \
                 mock.patch.object(module, "TF", side_effect=lambda image: torch.zeros(1)), \
                 mock.patch.object(module.Image, "open") as image_open:
                image_open.return_value.convert.return_value = object()
                module.main()
            text = (root / "results" / "tool_evaluation.json").read_text()
            out = json.loads(text)
            sys.path.insert(0, str(ROOT))
            import check_results
            fixture = root / "fixture"
            import shutil
            shutil.copytree(ROOT / "results", fixture / "results")
            shutil.copy2(ROOT / "results.schema.json", fixture / "results.schema.json")
            shutil.copy2(root / "results" / "tool_evaluation.json",
                         fixture / "results" / "tool_evaluation.json")
            self.assertEqual(check_results.check_results(fixture), [])
        m8 = out["chip_sequential_by_min_fields"]["8"]
        self.assertEqual(m8["n_confident"], 0)
        self.assertIsNone(m8["chip_acc_among_confident"])
        self.assertIsNone(m8["false_confident_rate"])
        self.assertIsNone(out["chip_acc_among_confident_wilson95"])
        self.assertNotIn("NaN", text)
        self.assertIn("0 confident calls", out["note_ci"])

    def test_wilson_normal_case_is_unchanged(self):
        self.assertEqual(module.wilson95_or_none(23, 19), [.629, .93])


if __name__ == "__main__":
    unittest.main()
