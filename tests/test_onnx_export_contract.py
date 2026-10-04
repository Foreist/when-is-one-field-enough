#!/usr/bin/env python3
import hashlib
import json
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "audit" / "export_onnx.py"


class FakeOnnx:
    def __init__(self, payload=b"onnx", error=None):
        self.payload = payload
        self.error = error

    def export(self, model, tensor, path, **kwargs):
        Path(path).write_bytes(self.payload)
        if self.error:
            raise self.error


class OnnxExportContract(unittest.TestCase):
    def run_export(self, checkpoint, out, fake_onnx, load_model):
        torch = types.ModuleType("torch")
        torch.__version__ = "mock-torch"
        torch.zeros = lambda *shape: ("zeros", shape)
        torch.onnx = fake_onnx
        inference = types.ModuleType("inference")
        inference.load_model = load_model
        argv = [str(SCRIPT), "--checkpoint", str(checkpoint), "--out", str(out)]
        with mock.patch.dict(sys.modules, {"torch": torch, "inference": inference}), \
             mock.patch.object(sys, "argv", argv):
            runpy.run_path(str(SCRIPT), run_name="__main__")

    def test_existing_onnx_is_preserved_before_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            out = root / "model.onnx"
            out.write_bytes(b"DO-NOT-OVERWRITE")
            loaded = []
            with self.assertRaises(FileExistsError):
                self.run_export(checkpoint, out, FakeOnnx(), lambda path: loaded.append(path))
            self.assertEqual(out.read_bytes(), b"DO-NOT-OVERWRITE")
            self.assertEqual(loaded, [])

    def test_existing_manifest_is_preserved_before_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            out = root / "model.onnx"
            manifest = root / "model.manifest.json"
            manifest.write_text("DO-NOT-OVERWRITE")
            loaded = []
            with self.assertRaises(FileExistsError):
                self.run_export(checkpoint, out, FakeOnnx(), lambda path: loaded.append(path))
            self.assertFalse(out.exists())
            self.assertEqual(manifest.read_text(), "DO-NOT-OVERWRITE")
            self.assertEqual(loaded, [])

    def test_manifest_created_during_export_is_preserved_and_our_onnx_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            out = root / "model.onnx"
            manifest = out.with_suffix(".manifest.json")

            class RacingOnnx(FakeOnnx):
                def export(self, model, tensor, path, **kwargs):
                    super().export(model, tensor, path, **kwargs)
                    manifest.write_bytes(b"DO-NOT-OVERWRITE")

            with self.assertRaises(FileExistsError):
                self.run_export(checkpoint, out, RacingOnnx(), lambda path: object())
            self.assertFalse(out.exists())
            self.assertEqual(manifest.read_bytes(), b"DO-NOT-OVERWRITE")

    def test_onnx_created_during_export_is_preserved_and_manifest_is_not_published(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            out = root / "model.onnx"

            class RacingOnnx(FakeOnnx):
                def export(self, model, tensor, path, **kwargs):
                    super().export(model, tensor, path, **kwargs)
                    out.write_bytes(b"DO-NOT-OVERWRITE")

            with self.assertRaises(FileExistsError):
                self.run_export(checkpoint, out, RacingOnnx(), lambda path: object())
            self.assertEqual(out.read_bytes(), b"DO-NOT-OVERWRITE")
            self.assertFalse(out.with_suffix(".manifest.json").exists())

    def test_export_failure_publishes_nothing_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            out = root / "model.onnx"
            before = set(root.iterdir())
            with self.assertRaisesRegex(RuntimeError, "mock export failure"):
                self.run_export(checkpoint, out, FakeOnnx(error=RuntimeError("mock export failure")), lambda path: object())
            self.assertEqual(set(root.iterdir()), before)
            self.assertFalse(out.exists())
            self.assertFalse(out.with_suffix(".manifest.json").exists())

    def test_success_publishes_matching_onnx_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            out = root / "model.onnx"
            payload = b"mock-onnx-payload"
            self.run_export(checkpoint, out, FakeOnnx(payload=payload), lambda path: object())
            manifest = json.loads(out.with_suffix(".manifest.json").read_text())
            self.assertEqual(out.read_bytes(), payload)
            self.assertEqual(manifest["onnx_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(manifest["checkpoint_sha256"], hashlib.sha256(b"checkpoint").hexdigest())
            self.assertEqual(manifest["opset"], 17)
            self.assertEqual(manifest["input_shape"], [1, 3, 384, 384])


if __name__ == "__main__":
    unittest.main()
