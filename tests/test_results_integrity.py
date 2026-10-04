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

    def test_json_schema_numeric_type_semantics(self):
        accepted = [
            (0, "number"), (0.0, "number"), (-3, "number"), (-3.0, "number"),
            (7, "number"), (0, "integer"), (0.0, "integer"),
            (-3, "integer"), (-3.0, "integer"), (7, "integer"),
        ]
        rejected = [
            (1.5, "integer"), (-1.5, "integer"),
            (True, "number"), (False, "number"),
            (True, "integer"), (False, "integer"),
            (float("nan"), "number"), (float("inf"), "number"),
            (float("-inf"), "integer"),
        ]
        for value, expected in accepted:
            with self.subTest(value=value, expected=expected):
                self.assertTrue(check_results.matches_type(value, expected))
        for value, expected in rejected:
            with self.subTest(value=value, expected=expected):
                self.assertFalse(check_results.matches_type(value, expected))

    def test_numeric_semantics_apply_to_roots_required_keys_and_unions(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "results").mkdir()
            schema = {
                "schema_version": 1,
                "files": {
                    "number_root.json": {"root_type": "number"},
                    "integer_root.json": {"root_type": "integer"},
                    "required.json": {
                        "root_type": "object",
                        "required": {
                            "number": "number",
                            "integer": "integer",
                            "union": ["integer", "str"],
                        },
                    },
                },
            }
            (root / "results.schema.json").write_text(json.dumps(schema))
            (root / "results" / "number_root.json").write_text("0")
            (root / "results" / "integer_root.json").write_text("-3.0")
            (root / "results" / "required.json").write_text(
                json.dumps({"number": 7, "integer": 0.0, "union": -3.0})
            )
            self.assertEqual(check_results.check_results(root), [])

            (root / "results" / "integer_root.json").write_text("1.5")
            required = {"number": True, "integer": False, "union": 1.5}
            (root / "results" / "required.json").write_text(json.dumps(required))
            issues = check_results.check_results(root)
            self.assertTrue(any("root type number, expected integer" in issue for issue in issues), issues)
            self.assertTrue(any("$.number type boolean, expected number" in issue for issue in issues), issues)
            self.assertTrue(any("$.integer type boolean, expected integer" in issue for issue in issues), issues)
            self.assertTrue(any("$.union type number, expected integer or str" in issue for issue in issues), issues)

    def test_integral_real_result_can_be_regenerated_as_integer(self):
        td, root = self.copy_fixture()
        try:
            path = root / "results" / "block_structure.json"
            data = json.loads(path.read_text())
            self.assertEqual(data["runlen_median"], 2.0)
            data["runlen_median"] = 2
            path.write_text(json.dumps(data))
            self.assertEqual(check_results.check_results(root), [])
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
