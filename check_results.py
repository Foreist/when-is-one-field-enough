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
        got_root = json_kind(value)
        expected_root = spec.get("root_type")
        if got_root != expected_root:
            issues.append(f"RESULT {filename}: root type {got_root}, expected {expected_root}")
        if isinstance(value, dict):
            for key, expected in spec.get("required", {}).items():
                if key not in value:
                    issues.append(f"RESULT {filename}: missing required key $.{key}")
                    continue
                got = json_kind(value[key])
                accepted = [expected] if isinstance(expected, str) else expected
                if got not in accepted:
                    issues.append(f"RESULT {filename}: $.{key} type {got}, expected {' or '.join(accepted)}")
        _walk(value, "$", filename, set(null_rules.get(filename, [])), issues)

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
