#!/usr/bin/env python3
"""Fail-closed integrity checks for committed results/*.json artifacts."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def json_kind(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


def matches_type(value, expected: str) -> bool:
    """Match JSON Schema numeric semantics while retaining json_kind diagnostics."""
    if isinstance(value, bool):
        return expected == "boolean"
    if expected == "number":
        return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
    if expected == "integer":
        return (isinstance(value, int)
                or (isinstance(value, float) and math.isfinite(value) and value.is_integer()))
    return json_kind(value) == expected


def _tool_evaluation_allowed_nulls(value, issues: list[str]) -> set[str]:
    """Validate abstention-dependent metrics and return their legitimate null paths."""
    allowed: set[str] = set()
    if not isinstance(value, dict):
        return allowed
    sequential = value.get("chip_sequential_by_min_fields")
    if not isinstance(sequential, dict):
        return allowed
    n_chips = value.get("n_chips")
    valid_n_chips = type(n_chips) is int and n_chips >= 0

    for min_fields, metrics in sequential.items():
        item_path = f"$.chip_sequential_by_min_fields.{min_fields}"
        if not isinstance(min_fields, str) or not min_fields.isdigit():
            issues.append(f"RESULT tool_evaluation.json: invalid min-fields key {min_fields!r}")
            continue
        if not isinstance(metrics, dict):
            issues.append(f"RESULT tool_evaluation.json: {item_path} type {json_kind(metrics)}, expected object")
            continue
        n_confident = metrics.get("n_confident")
        if (type(n_confident) is not int or n_confident < 0
                or not valid_n_chips or n_confident > n_chips):
            issues.append(
                f"RESULT tool_evaluation.json: {item_path}.n_confident must be an integer "
                "between 0 and n_chips"
            )
            continue
        for name in ("chip_acc_among_confident", "false_confident_rate"):
            metric_path = f"{item_path}.{name}"
            if name not in metrics:
                issues.append(f"RESULT tool_evaluation.json: missing required metric {metric_path}")
                continue
            metric = metrics[name]
            if n_confident == 0:
                if metric is None:
                    allowed.add(metric_path)
                else:
                    issues.append(
                        f"RESULT tool_evaluation.json: {metric_path} must be null when n_confident is 0"
                    )
            elif not matches_type(metric, "number") or not 0 <= metric <= 1:
                issues.append(
                    f"RESULT tool_evaluation.json: {metric_path} type {json_kind(metric)}, "
                    "expected number in [0, 1]"
                )

    min8 = sequential.get("8")
    if not isinstance(min8, dict):
        issues.append("RESULT tool_evaluation.json: required min-fields 8 metrics missing or invalid")
    elif type(min8.get("n_confident")) is int and min8["n_confident"] == 0:
        ci_path = "$.chip_acc_among_confident_wilson95"
        if value.get("chip_acc_among_confident_wilson95") is None:
            allowed.add(ci_path)
        else:
            issues.append(
                f"RESULT tool_evaluation.json: {ci_path} must be null when min-fields 8 n_confident is 0"
            )
    return allowed


def _walk(value, path: str, filename: str, allowed_nulls: set[str], issues: list[str]) -> None:
    if value is None:
        if path not in allowed_nulls:
            issues.append(f"RESULT {filename}: unexpected null at {path}")
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            issues.append(f"RESULT {filename}: non-finite number at {path}: {value!r}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (f"{path}.{key}" if re.fullmatch(r"[A-Za-z0-9_]+", key)
                          else f"{path}[{json.dumps(key)}]")
            _walk(child, child_path, filename, allowed_nulls, issues)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _walk(child, f"{path}[{index}]", filename, allowed_nulls, issues)


def _unique_object(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate key {key!r}")
        value[key] = child
    return value


def _reject_constant(token):
    raise ValueError(f"non-standard numeric constant {token}")


def _read_json(path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def _schema_error(schema):
    kinds = {"null", "boolean", "integer", "number", "object", "array", "str"}
    if (not isinstance(schema, dict) or type(schema.get("schema_version")) is not int
            or schema["schema_version"] != 1 or not isinstance(schema.get("files"), dict)
            or not schema["files"]):
        return "expected schema_version 1 and nonempty object files"
    for filename, spec in schema["files"].items():
        if not filename.endswith('.json') or Path(filename).name != filename:
            return f"invalid result filename {filename!r}"
        if (not isinstance(spec, dict) or not isinstance(spec.get("root_type"), str)
                or spec["root_type"] not in kinds):
            return f"invalid root specification for {filename}"
        required = spec.get("required", {})
        if not isinstance(required, dict):
            return f"invalid required keys for {filename}"
        for key, expected in required.items():
            accepted = [expected] if isinstance(expected, str) else expected
            if (not isinstance(accepted, list) or not accepted
                    or any(not isinstance(k, str) or k not in kinds for k in accepted)):
                return f"invalid type declaration for {filename}.{key}"
    null_rules = schema.get("allow_null_paths", {})
    if not isinstance(null_rules, dict):
        return "allow_null_paths must be an object"
    for filename, paths in null_rules.items():
        if (filename not in schema["files"] or not isinstance(paths, list)
                or any(not isinstance(p, str) or not p.startswith('$') for p in paths)):
            return f"invalid null paths for {filename}"
    return None


def check_results(here: Path | None = None, schema_path: Path | None = None) -> list[str]:
    here = Path(here) if here else HERE
    schema_path = Path(schema_path) if schema_path else here / "results.schema.json"
    try:
        schema = _read_json(schema_path)
    except (ValueError, OSError, RecursionError) as error:
        return [f"RESULT schema unreadable: {error}"]
    error = _schema_error(schema)
    if error:
        return [f"RESULT schema invalid: {error}"]

    results_dir = here / "results"
    declared = schema["files"]
    null_rules = schema.get("allow_null_paths", {})
    issues: list[str] = []
    parsed: dict[str, object] = {}

    for filename, spec in declared.items():
        path = results_dir / filename
        if not path.is_file():
            issues.append(f"RESULT {filename}: required file missing")
            continue
        try:
            value = _read_json(path)
        except Exception as error:
            issues.append(f"RESULT {filename}: malformed JSON ({error})")
            continue
        parsed[filename] = value
        semantic_nulls = set()
        if filename == "tool_evaluation.json":
            semantic_nulls = _tool_evaluation_allowed_nulls(value, issues)
        got_root = json_kind(value)
        expected_root = spec.get("root_type")
        if not matches_type(value, expected_root):
            issues.append(f"RESULT {filename}: root type {got_root}, expected {expected_root}")
        if isinstance(value, dict):
            for key, expected in spec.get("required", {}).items():
                if key not in value:
                    issues.append(f"RESULT {filename}: missing required key $.{key}")
                    continue
                got = json_kind(value[key])
                accepted = [expected] if isinstance(expected, str) else expected
                if not any(matches_type(value[key], kind) for kind in accepted):
                    issues.append(f"RESULT {filename}: $.{key} type {got}, expected {' or '.join(accepted)}")
        _walk(value, "$", filename,
              set(null_rules.get(filename, [])) | semantic_nulls, issues)

    for path in sorted(results_dir.glob("*.json")):
        if path.name in parsed or path.name in declared:
            continue
        try:
            value = _read_json(path)
        except Exception as error:
            issues.append(f"RESULT {path.name}: malformed JSON ({error})")
            continue
        _walk(value, "$", path.name, set(), issues)
    return issues


def main() -> int:
    issues = check_results()
    for issue in issues:
        print(issue)
    print(f"{len(issues)} result integrity issues")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
