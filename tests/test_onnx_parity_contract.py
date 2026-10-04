#!/usr/bin/env python3
import json
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "audit" / "12_onnx_parity.py"


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def __getitem__(self, key):
        return Tensor(self.value[key])

    def numpy(self):
        return self.value


class NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class ParityContract(unittest.TestCase):
    def run_main(self, ort_output, torch_probs=((0.5, 0.5),)):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        onnx = root / "model.onnx"
        onnx.write_bytes(b"mock-onnx")
        out = root / "parity.json"

        torch = types.ModuleType("torch")
        torch.no_grad = lambda: NoGrad()
        torch.softmax = lambda value, dim: Tensor(torch_probs)

        class Model:
            def __call__(self, value):
                return Tensor([[0.0, 0.0]])

        inference = types.ModuleType("inference")
        inference.load_model = lambda path: Model()
        inference.TF = lambda image: Tensor([[0.0]])

        onnxruntime = types.ModuleType("onnxruntime")

        class Session:
            def __init__(self, path):
                pass

            def get_inputs(self):
                return [types.SimpleNamespace(name="input")]

            def run(self, *args):
                return [np.asarray(ort_output, dtype=float)]

        onnxruntime.InferenceSession = Session
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = lambda *args, **kwargs: self.fail("local ONNX must not fetch")
        pil = types.ModuleType("PIL")
        pil.Image = types.SimpleNamespace(
            open=lambda path: types.SimpleNamespace(convert=lambda mode: object())
        )
        argv = [str(SCRIPT), "--onnx", str(onnx), "--out", str(out)]
        modules = {
            "torch": torch,
            "inference": inference,
            "onnxruntime": onnxruntime,
            "huggingface_hub": hub,
            "PIL": pil,
        }
        self.last_tempdir, self.last_out = td, out
        with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, "argv", argv):
            runpy.run_path(str(SCRIPT), run_name="__main__")
        return td, out

    def assert_rejected_without_output(self, ort_output, torch_probs=((0.5, 0.5),)):
        try:
            with self.assertRaises(ValueError):
                self.run_main(ort_output, torch_probs)
            self.assertFalse(self.last_out.exists())
        finally:
            self.last_tempdir.cleanup()

    def test_rejects_nonfinite_onnx_outputs(self):
        for output in ([[np.nan, np.nan]], [[np.inf, 0.0]]):
            with self.subTest(output=output):
                self.assert_rejected_without_output(output)

    def test_rejects_wrong_onnx_shapes_and_missing_class(self):
        for output in ([0.5, 0.5], [[1.0]], [[0.2, 0.3, 0.5]]):
            with self.subTest(output=output):
                self.assert_rejected_without_output(output)

    def test_rejects_invalid_pytorch_probabilities(self):
        for probs in ([[np.nan, 0.5]], [[1.0]], [[0.2, 0.3, 0.5]]):
            with self.subTest(probs=probs):
                self.assert_rejected_without_output([[0.5, 0.5]], probs)

    def test_valid_probabilities_write_strict_zero_diff_result(self):
        td, out = self.run_main([[0.5, 0.5]])
        try:
            text = out.read_text()
            result = json.loads(text, parse_constant=lambda value: self.fail(value))
            self.assertEqual(result["n_fields"], 44)
            self.assertEqual(result["max_abs_diff"], 0.0)
            self.assertEqual(result["mean_abs_diff"], 0.0)
        finally:
            td.cleanup()

    def test_valid_logits_are_softmaxed(self):
        td, out = self.run_main([[0.0, 0.0]])
        try:
            self.assertEqual(json.loads(out.read_text())["max_abs_diff"], 0.0)
        finally:
            td.cleanup()


if __name__ == "__main__":
    unittest.main()
