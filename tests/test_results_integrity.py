#!/usr/bin/env python3
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import check_results


class ResultIntegrity(unittest.TestCase):
    def copy_fixture(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        shutil.copytree(ROOT / "results", root / "results")
        shutil.copy2(ROOT / "results.schema.json", root / "results.schema.json")
        return td, root

    def test_current_results_pass(self):
        self.assertEqual(check_results.check_results(ROOT), [])

    def test_every_required_file_missing_fails(self):
        schema = json.loads((ROOT / "results.schema.json").read_text())
        for filename in schema["files"]:
            with self.subTest(filename=filename):
                td, root = self.copy_fixture()
                try:
                    (root / "results" / filename).unlink()
                    issues = check_results.check_results(root)
                    self.assertTrue(any(filename in issue and "required file missing" in issue for issue in issues), issues)
                finally:
                    td.cleanup()

    def test_every_required_root_mutation_fails(self):
        schema = json.loads((ROOT / "results.schema.json").read_text())
        for filename in schema["files"]:
            with self.subTest(filename=filename):
                td, root = self.copy_fixture()
                try:
                    (root / "results" / filename).write_text("[]")
                    issues = check_results.check_results(root)
                    self.assertTrue(any(filename in issue and "root type array" in issue for issue in issues), issues)
                finally:
                    td.cleanup()

    def test_policy_bootstrap_malformed_root_null_key_and_type_fail(self):
        mutations = [
            ("{malformed", "malformed JSON"),
            ("null", "root type null"),
            ('{"k8":{},"k12":{},"n_chips_k8":13}', "missing required key $.n_chips_k12"),
            ('{"k8":{},"k12":{},"n_chips_k8":13,"n_chips_k12":null}', "$.n_chips_k12 type null"),
        ]
        for payload, expected in mutations:
            with self.subTest(expected=expected):
                td, root = self.copy_fixture()
                try:
                    (root / "results" / "policy_bootstrap.json").write_text(payload)
                    issues = check_results.check_results(root)
                    self.assertTrue(any(expected in issue for issue in issues), issues)
                finally:
                    td.cleanup()

    def test_declared_nested_nulls_are_allowed_but_other_null_is_not(self):
        self.assertFalse(any("unexpected null" in issue for issue in check_results.check_results(ROOT)))
        td, root = self.copy_fixture()
        try:
            path = root / "results" / "policy_bootstrap.json"
            data = json.loads(path.read_text())
            data["k8"]["unexpected"] = None
            path.write_text(json.dumps(data))
            issues = check_results.check_results(root)
            self.assertTrue(any("unexpected null at $.k8.unexpected" in issue for issue in issues), issues)
        finally:
            td.cleanup()

    def test_corrupt_schema_is_descriptive_failure(self):
        for payload in ('[]','null','{"schema_version":1,"files":{}}',
                        '{"schema_version":1,"files":{"x.json":null}}',
                        '{"schema_version":1,"files":{"x.json":{"root_type":"unknown"}}}',
                        '{"schema_version":1,"files":{"x.json":{"root_type":[]}}}',
                        '{"schema_version":1,"files":{"x.json":{"root_type":"object","required":[]}}}',
                        '{"schema_version":1,"files":{"x.json":{"root_type":"object"}},"allow_null_paths":[]}',
                        '{"schema_version":1,"files":{},"files":{}}'):
            with self.subTest(payload=payload):
                td,root=self.copy_fixture()
                try:
                    (root/'results.schema.json').write_text(payload)
                    self.assertTrue(any('schema' in i for i in check_results.check_results(root)))
                finally:
                    td.cleanup()

    def test_duplicate_object_keys_fail_even_if_last_value_is_valid(self):
        for filename,payload in (
                ('policy_bootstrap.json','{"k8":{},"k12":{},"n_chips_k8":null,"n_chips_k8":13,"n_chips_k12":12}'),
                ('extra.json','{"x":{"value":null,"value":1}}')):
            with self.subTest(filename=filename):
                td,root=self.copy_fixture()
                try:
                    (root/'results'/filename).write_text(payload)
                    issues=check_results.check_results(root)
                    self.assertTrue(any(filename in i and 'duplicate key' in i for i in issues),issues)
                finally:
                    td.cleanup()

    def test_dotted_key_cannot_impersonate_allowed_nested_null(self):
        td,root=self.copy_fixture()
        try:
            path=root/'results/lolo_cellline.json'
            data=json.loads(path.read_text())
            data['per_cell']['cell_type_HUVEC.seen_line_auc']=None
            path.write_text(json.dumps(data))
            issues=check_results.check_results(root)
            self.assertTrue(any('unexpected null' in i and 'cell_type_HUVEC.seen_line_auc' in i
                                for i in issues),issues)
        finally:
            td.cleanup()

    def test_unlisted_json_is_parsed_and_walked_past_400_items(self):
        for payload, expected in [
            ('{"x": [0, 1e999]}', "non-finite number at $.x[1]"),
            ('{"x": [' + ','.join(["0"] * 401 + ["NaN"]) + ']}', "non-standard numeric constant NaN"),
            ("{bad", "malformed JSON"),
        ]:
            with self.subTest(expected=expected):
                td, root = self.copy_fixture()
                try:
                    (root / "results" / "extra.json").write_text(payload)
                    issues = check_results.check_results(root)
                    self.assertTrue(any("extra.json" in issue and expected in issue for issue in issues), issues)
                finally:
                    td.cleanup()


if __name__ == "__main__":
    unittest.main()
