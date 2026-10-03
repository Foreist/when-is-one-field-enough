#!/usr/bin/env python3
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import check_numbers


class PinErrors(unittest.TestCase):
    def test_invalidated_null_pin_is_failure_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'results').mkdir()
            (root/'results/value.json').write_text('{"p":null}')
            (root/'check_numbers_pins.txt').write_text('0.14 value.json:p .2f\n')
            with mock.patch.object(check_numbers,'HERE',root),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check_numbers.check_pins([]),1)

    def test_missing_result_key_is_failure_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'results').mkdir();(root/'results/value.json').write_text('{}')
            (root/'check_numbers_pins.txt').write_text('0.14 value.json:p .2f\n')
            with mock.patch.object(check_numbers,'HERE',root),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check_numbers.check_pins([]),1)


if __name__=='__main__':unittest.main()
