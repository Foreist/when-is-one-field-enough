#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed headline claim checker.

Each claim maps a results/*.json key (or a named derived expression of keys) to
one or more *context regexes*. Every regex match is checked: a single mutated
number fails even when other copies of the same literal remain. Zero matches on
a required anchor fail. Optional documents are listed explicitly; a missing
optional file is skipped, never treated as a pass.

    python3 check_claims.py                 # exit 1 on any claim failure
    python3 check_claims.py --mutations     # in-memory mutation self-test
    python3 check_claims.py --here PATH
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE_DEFAULT = Path(__file__).resolve().parent
NUM_IN_CAPTURE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?", re.I)


def load_ledger(path: Path) -> dict:
    return json.loads(path.read_text())


def format_value(v, fmt: str) -> str:
    x = float(v)
    if fmt.startswith("%"):
        x *= 100.0
        fmt = fmt[1:]
    if fmt in {"d", ",d"} or (fmt.endswith("d") and "f" not in fmt and "e" not in fmt):
        return format(int(round(x)), fmt)
    return format(x, fmt)


def leaf(here: Path, spec: str, results_dir: Path):
    fname, path = spec.split(":", 1)
    obj = json.loads((results_dir / fname).read_text())
    for k in path.split("."):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    if isinstance(obj, bool):
        raise TypeError(f"{spec} is boolean, not a number")
    return obj


def _num(here: Path, spec_or_num, results_dir: Path) -> float:
    if isinstance(spec_or_num, (int, float)):
        return float(spec_or_num)
    return float(leaf(here, spec_or_num, results_dir))


def derived_value(here: Path, spec: dict, results_dir: Path) -> float:
    op = spec["op"]
    args = [_num(here, a, results_dir) for a in spec["args"]]
    if op == "div":
        return args[0] / args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul_round":
        return float(round(args[0] * args[1]))
    if op == "pct_saved":
        return 100.0 * (1.0 - args[0] / args[1])
    if op == "floor":
        return float(math.floor(args[0]))
    if op == "round":
        nd = int(spec.get("ndigits", 0))
        return float(round(args[0], nd))
    raise ValueError(f"unknown derived op {op!r}")


def expected_map(claim: dict, here: Path, results_dir: Path) -> dict[str, str]:
    """group name -> formatted literal."""
    out = {}
    groups = claim.get("groups")
    if groups:
        for gname, g in groups.items():
            fmt = g.get("format", claim.get("format", ".3f"))
            if "derived" in g:
                val = derived_value(here, g["derived"], results_dir)
            else:
                val = leaf(here, g["result"], results_dir)
            out[gname] = format_value(val, fmt)
        return out
    fmt = claim["format"]
    if "derived" in claim:
        val = derived_value(here, claim["derived"], results_dir)
    else:
        val = leaf(here, claim["result"], results_dir)
    out[claim.get("group", "num")] = format_value(val, fmt)
    return out


def norm_lit(s: str) -> str:
    return (
        s.replace(",", "")
        .replace("−", "-")
        .replace("–", "-")
        .replace("*", "")
        .replace("_", "")
        .strip()
    )


def valid_number_literal(value: str) -> bool:
    raw = value.replace('−', '-').strip()
    return bool(re.fullmatch(r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?(?:e[+-]?\d+)?', raw, re.I))


def lits_match(captured: str, expected: str) -> bool:
    if not valid_number_literal(captured) or not valid_number_literal(expected):
        return False
    a, b = norm_lit(captured), norm_lit(expected)
    if a == b:
        return True
    # "+7.9" in JSON format vs "7.9" in running text (sign in the regex around it)
    if a.lstrip("+") == b.lstrip("+"):
        return True
    return False


def resolve_docs(ledger: dict, here: Path) -> dict[str, Path]:
    out = {}
    for doc_id, meta in ledger["docs"].items():
        out[doc_id] = (here / meta["path"]).resolve()
    return out


def read_texts(ledger: dict, here: Path, texts: dict[str, str] | None = None) -> dict[str, str | None]:
    """None means the optional file is absent. Missing required files stay None too (caller fails)."""
    paths = resolve_docs(ledger, here)
    out: dict[str, str | None] = {}
    for doc_id, path in paths.items():
        if texts is not None and doc_id in texts:
            out[doc_id] = texts[doc_id]
            continue
        if path.is_file():
            out[doc_id] = path.read_text()
        else:
            out[doc_id] = None
    return out


def _anchor_optional(anchor: dict, ledger: dict) -> bool:
    if "optional" in anchor:
        return bool(anchor["optional"])
    docs_meta = ledger["docs"]
    return all(not docs_meta[d].get("required", True) for d in anchor["docs"])


def iter_anchor_matches(pattern: re.Pattern, text: str):
    yield from pattern.finditer(text)


def complete_number_capture(text: str, match: re.Match, group: str | int) -> bool:
    """A valid prefix of a larger token is not the claimed numeric literal."""
    start, end = match.span(group)
    if start < 0:
        return False
    return any(token.end() == end and (token.start() == start or
               (token.start() == start - 1 and text[token.start()] == '+'))
               for token in NUM_IN_CAPTURE.finditer(text))


def check_claims(
    here: Path | None = None,
    ledger: dict | None = None,
    texts: dict[str, str] | None = None,
    results_dir: Path | None = None,
    ledger_path: Path | None = None,
) -> list[str]:
    here = Path(here) if here else HERE_DEFAULT
    ledger_path = Path(ledger_path) if ledger_path else here / "claims.json"
    if ledger is None:
        if not ledger_path.is_file():
            return [f"FAIL: claim ledger missing: {ledger_path}"]
        ledger = load_ledger(ledger_path)
    results_dir = Path(results_dir) if results_dir else here / "results"
    issues: list[str] = []
    doc_texts = read_texts(ledger, here, texts)
    docs_meta = ledger["docs"]

    for claim in ledger["claims"]:
        cid = claim["id"]
        try:
            expect = expected_map(claim, here, results_dir)
        except Exception as e:
            issues.append(f"CLAIM {cid}: cannot read result key ({e})")
            continue
        claim_required = claim.get("required", True)
        for anchor in claim["anchors"]:
            aid = anchor.get("id", "anchor")
            try:
                pat = re.compile(anchor["regex"], re.M)
            except re.error as e:
                issues.append(f"CLAIM {cid}/{aid}: bad regex ({e})")
                continue
            min_n = int(anchor.get("min", 1))
            exact_n = anchor.get("exact")
            optional = _anchor_optional(anchor, ledger)
            total = 0
            present_docs = []
            missing_required = []
            for doc_id in anchor["docs"]:
                required_doc = docs_meta[doc_id].get("required", True)
                text = doc_texts.get(doc_id)
                if text is None:
                    if required_doc:
                        missing_required.append(doc_id)
                    continue
                present_docs.append(doc_id)
                matches = list(iter_anchor_matches(pat, text))
                total += len(matches)
                per_doc = anchor.get("counts", {}).get(doc_id)
                if per_doc is None:
                    issues.append(f"CLAIM {cid}/{aid}: no expected count declared for {doc_id}")
                elif len(matches) != per_doc:
                    issues.append(f"CLAIM {cid}/{aid} {doc_id}: {len(matches)} matches, need exact {per_doc}")
                named = set(pat.groupindex)
                check_names = [g for g in expect if g in named or (g.isdigit() and int(g) <= pat.groups)]
                if not check_names:
                    issues.append(f"CLAIM {cid}/{aid}: anchor has no capture mapped to a result key")
                    continue
                for i, m in enumerate(matches, 1):
                    for gname in check_names:
                        exp = expect[gname]
                        if gname.isdigit():
                            captured = m.group(int(gname))
                        else:
                            captured = m.group(gname)
                        if captured is None:
                            issues.append(
                                f"CLAIM {cid}/{aid} {doc_id}#{i}: group {gname!r} missing"
                            )
                            continue
                        if not complete_number_capture(text, m, int(gname) if gname.isdigit() else gname):
                            issues.append(f"CLAIM {cid}/{aid} {doc_id}#{i}: incomplete numeric token for {gname}")
                        if not lits_match(captured, exp):
                            issues.append(
                                f"CLAIM {cid}/{aid} {doc_id}#{i}: {gname}={captured!r} != {exp!r}"
                            )
            if missing_required:
                issues.append(
                    f"CLAIM {cid}/{aid}: required doc missing: {', '.join(missing_required)}"
                )
                continue
            if not present_docs:
                if optional:
                    continue
                issues.append(
                    f"CLAIM {cid}/{aid}: no listed document exists (fail-closed; paths must be explicit)"
                )
                continue
            if total < min_n:
                issues.append(
                    f"CLAIM {cid}/{aid}: {total} matches in {present_docs}, need >= {min_n}"
                )
            if exact_n is not None and total != int(exact_n):
                issues.append(
                    f"CLAIM {cid}/{aid}: {total} matches in {present_docs}, need exact {exact_n}"
                )
            if claim_required and total == 0 and not optional:
                issues.append(f"CLAIM {cid}/{aid}: zero anchors for required rule")
    return issues


def _replace_group(text: str, m: re.Match, group: str | int, new: str) -> str:
    i = m.start(group)
    j = m.end(group)
    if i < 0 or j < 0:
        return text
    return text[:i] + new + text[j:]


def _mutated_literal(lit: str) -> str:
    """Change the last digit (or last char of integer) so the token is visibly wrong."""
    s = lit
    for i in range(len(s) - 1, -1, -1):
        if s[i].isdigit():
            nxt = "1" if s[i] == "0" else "0" if s[i] == "9" else str((int(s[i]) + 1) % 10)
            # avoid a no-op if +1 wrapped back; prefer increment with wrap 9->0
            if nxt == s[i]:
                nxt = "1" if s[i] != "1" else "2"
            return s[:i] + nxt + s[i + 1 :]
    return s + "0"


def run_mutations(
    here: Path | None = None,
    ledger: dict | None = None,
    results_dir: Path | None = None,
    ledger_path: Path | None = None,
    limit: int | None = None,
) -> list[dict]:
    """In-memory only. Mutate the first capture of each required anchor; expect a failure."""
    here = Path(here) if here else HERE_DEFAULT
    ledger_path = Path(ledger_path) if ledger_path else here / "claims.json"
    ledger = ledger or load_ledger(ledger_path)
    results_dir = Path(results_dir) if results_dir else here / "results"
    base_texts = {k: v for k, v in read_texts(ledger, here).items() if v is not None}
    baseline = check_claims(here=here, ledger=ledger, texts=base_texts, results_dir=results_dir)
    if baseline:
        return [
            {
                "claim": "_baseline",
                "anchor": "green",
                "doc": "",
                "from": "",
                "to": "",
                "detected": False,
                "n_issues": len(baseline),
                "issues": baseline,
            }
        ]
    reports = []
    for claim in ledger["claims"]:
        if not claim.get("required", True):
            continue
        cid = claim["id"]
        try:
            expect = expected_map(claim, here, results_dir)
        except Exception:
            continue
        for anchor in claim["anchors"]:
            if _anchor_optional(anchor, ledger):
                continue
            aid = anchor.get("id", "anchor")
            pat = re.compile(anchor["regex"], re.M)
            for doc_id in anchor["docs"]:
                text = base_texts.get(doc_id)
                if not text:
                    continue
                groups = [g for g in expect if g in pat.groupindex]
                for occurrence, m in enumerate(pat.finditer(text), 1):
                    for gname in groups:
                        captured = m.group(gname)
                        if captured is None:
                            continue
                        variants = [_mutated_literal(captured), captured + "9", captured + "e2"]
                        for new in dict.fromkeys(variants):
                            mutated = dict(base_texts)
                            mutated[doc_id] = _replace_group(text, m, gname, new)
                            issues = check_claims(here=here, ledger=ledger, texts=mutated, results_dir=results_dir)
                            tag = f"CLAIM {cid}/{aid}"
                            reports.append(dict(claim=cid, anchor=aid, group=gname, doc=doc_id,
                                                occurrence=occurrence, **{"from": captured, "to": new},
                                                detected=any(tag in issue for issue in issues), n_issues=len(issues)))
                            if limit and len(reports) >= limit:
                                return reports
    return reports


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--here", type=Path, default=HERE_DEFAULT)
    p.add_argument("--mutations", action="store_true", help="in-memory mutation self-test")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    here = args.here.resolve()
    if args.mutations:
        reports = run_mutations(here=here)
        missed = [r for r in reports if not r["detected"]]
        n_ok = len(reports) - len(missed)
        print(f"{n_ok}/{len(reports)} in-memory headline mutations detected", file=sys.stderr)
        for r in missed:
            print(
                f"MISS mutation {r['claim']}/{r['anchor']} {r['doc']} {r['from']!r}->{r['to']!r}",
                file=sys.stderr,
            )
        if args.json:
            print(json.dumps(reports, indent=2))
        return 1 if missed or n_ok < 7 else 0
    issues = check_claims(here=here)
    for i in issues:
        print(i)
    print(f"{len(issues)} claim issues", file=sys.stderr)
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
