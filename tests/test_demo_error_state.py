#!/usr/bin/env python3
"""Local Gradio callback failure contracts with temporary inputs and a fake model."""
import ast
import collections
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import inference  # noqa: E402


class ModelStub:
    def __call__(self, value):
        return torch.tensor([[0.0, 2.0]])


def callbacks(model=None):
    # Use the exact callback bodies without model loading or Gradio/server boot.
    path = REPO / 'demo/app.py'
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('run_chip', 'run_plate')]
    namespace = dict(Path=Path, collections=collections, json=json, torch=torch,
                     Image=Image, MODEL=model or ModelStub(),
                     TF=lambda image: torch.zeros((3, 2, 2)),
                     MODEL_CARD=inference.MODEL_CARD,
                     RESOURCE_BASIS=inference.RESOURCE_BASIS,
                     natural_key=inference.natural_key,
                     validate_rule_args=inference.validate_rule_args,
                     sequential_decision=inference.sequential_decision,
                     short_chip_warning=inference.short_chip_warning,
                     last_consumed_index=inference.last_consumed_index,
                     mark_qc_map=inference.mark_qc_map)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    namespace['plt'] = plt
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class DemoErrorState(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.good = self.root / 'a-good'; self.good.mkdir()
        self.field = self.good / 'field.png'
        Image.new('RGB', (2, 2), 'white').save(self.field)
        self.bad = self.root / 'z-bad'; self.bad.mkdir()
        self.corrupt = self.bad / 'corrupt.png'
        self.corrupt.write_bytes(b'harmless invalid image fixture')

    def assert_chip_failed(self, output):
        figure, text, report = output
        self.assertIsNone(figure)
        self.assertIn('QC failed', text)
        self.assertIn('No chip result', text)
        self.assertNotIn('PASS', text)
        self.assertNotIn('FAIL**', text)
        self.assertEqual(report, '')
        self.assertNotIn(str(self.root), text)

    def test_chip_decode_failure_clears_all_outputs_after_success(self):
        funcs = callbacks()
        figure, _, report = funcs['run_chip']([str(self.field)] * 8)
        self.assertEqual(json.loads(report)['call'], 'pass')
        funcs['plt'].close(figure)
        self.assert_chip_failed(funcs['run_chip']([str(self.corrupt)]))

    def test_chip_missing_image_clears_outputs(self):
        self.assert_chip_failed(callbacks()['run_chip']([str(self.root / 'missing.png')]))

    def test_chip_supported_image_model_failure_clears_outputs(self):
        model = mock.Mock(side_effect=RuntimeError('controlled model failure'))
        self.assert_chip_failed(callbacks(model)['run_chip']([str(self.field)]))

    def test_chip_nonfinite_model_output_clears_outputs(self):
        model = mock.Mock(return_value=torch.tensor([[float('nan'), 0.0]]))
        self.assert_chip_failed(callbacks(model)['run_chip']([str(self.field)] * 8))

    def test_chip_failure_in_unused_tail_never_returns_earlier_pass(self):
        self.assert_chip_failed(callbacks()['run_chip'](
            [str(self.field)] * 8 + [str(self.corrupt)]))

    def test_plate_decode_failure_never_returns_partial_rows(self):
        text, rows = callbacks()['run_plate']([str(self.field)] * 8 + [str(self.corrupt)])
        self.assertIsNone(rows)
        self.assertIn('QC failed', text)
        self.assertIn('No plate result', text)
        self.assertNotIn(str(self.root), text)

    def test_plate_supported_image_model_failure_clears_rows(self):
        model = mock.Mock(side_effect=RuntimeError('controlled model failure'))
        text, rows = callbacks(model)['run_plate']([str(self.field)])
        self.assertIsNone(rows)
        self.assertIn('QC failed', text)
        self.assertIn('No plate result', text)

    def test_valid_chip_and_plate_outputs_stay_unchanged(self):
        funcs = callbacks()
        files = [str(self.field)] * 8
        figure, text, report = funcs['run_chip'](files)
        try:
            result = json.loads(report)
            self.assertEqual(result['call'], 'pass')
            self.assertEqual(result['fields_used'], 8)
            self.assertEqual(result['n_fields_scored'], 8)
            self.assertEqual(result['per_field'], [.1192] * 8)
            self.assertIn('chip call: **PASS**', text)
        finally:
            funcs['plt'].close(figure)
        _, rows = funcs['run_plate'](files)
        self.assertEqual(rows, [['a-good', 'pass', .002, .998, '8/8', '']])

    def test_empty_inputs_and_invalid_policy_keep_explicit_no_result(self):
        funcs = callbacks()
        self.assertEqual(funcs['run_chip']([]),
                         (None, 'upload at least one image (one chip per run)', ''))
        self.assertEqual(funcs['run_plate']([]),
                         ('upload a folder that contains one subfolder per chip', None))
        figure, _, report = funcs['run_chip']([str(self.field)], conf=.5)
        self.assertIsNone(figure)
        self.assertEqual(report, '')

    def test_process_interruption_is_not_swallowed(self):
        model = mock.Mock(side_effect=KeyboardInterrupt())
        funcs = callbacks(model)
        with self.assertRaises(KeyboardInterrupt):
            funcs['run_chip']([str(self.field)])
        with self.assertRaises(KeyboardInterrupt):
            funcs['run_plate']([str(self.field)])


if __name__ == '__main__':
    unittest.main()
