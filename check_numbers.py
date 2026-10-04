#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lint + auxiliary pins + fail-closed headline claims.

    python3 check_numbers.py [--numbers-only] [extra.md ...]

This script is *not* a proof that every prose number is correct.

0. Artifact integrity (`check_results.py` + `results.schema.json`): required result
   membership, JSON/root/top-level types, finite numbers and declared nulls.
   This runs even with --numbers-only; it is not full nested semantic validation.
1. Generic lint: each numeric token in REPORT/README/audit README must equal
   *some* results/*.json leaf (at written precision) or an allow-list constant.
   Coincidence with an unrelated leaf is enough to pass. This is a leftover
   scan for invented tokens, not verification of what a number *means*.
2. Auxiliary pins (`check_numbers_pins.txt`): the formatted JSON leaf still
   equals the pinned literal, and that literal still appears *somewhere* in
   the concatenated docs. A single mutated copy can still pass if another
   occurrence of the same digits remains.
3. Fail-closed claims (`check_claims.py` + `claims.json`): each headline is
   a result key (or a named derived expression of keys) bound to a context
   regex. Every match is checked; zero matches or a wrong count fail.

Cell-by-cell table checks remain in `check_tables.py`. PDF freshness uses
`report.sources.json` (SHA-256 of REPORT.md, make_report_pdf.py, figures,
and the PDF) — not git timestamps. `--numbers-only` skips PDF and tables.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = [HERE / "REPORT.md", HERE / "README.md", HERE / "audit" / "README.md"]
NUM = re.compile(r"(?<![\w.])[−-]?\d[\d,]*(?:\.\d+)?(?:e[−-]?\d+)?(?![\w])")
MANIFEST = HERE / "report.sources.json"


def leaves(o):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        yield float(o)
    elif isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list) and len(o) <= 400:
        for v in o:
            yield from leaves(v)


def json_values():
    vals = set()
    for f in sorted((HERE / "results").glob("*.json")):
        d = json.loads(f.read_text())
        for v in leaves(d):
            vals.add(v)
    return vals


def reps(v):
    """Strings a value may legitimately be written as."""
    out = set()
    for x in (abs(v), 100 * abs(v)):
        for d in range(0, 4):
            out.add(f"{x:.{d}f}")
        out.add(f"{x:.3g}")
        out.add(f"{x:.1e}")
    return out


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def current_pdf_sources() -> dict[str, str]:
    """Same membership as make_report_pdf.source_manifest (do not import playwright)."""
    sources = [HERE / "REPORT.md", HERE / "make_report_pdf.py"]
    sources += sorted((HERE / "figures").glob("*.png"))
    return {str(p.relative_to(HERE)): sha256_file(p) for p in sources}


def check_pdf_sources(pdf: Path = HERE / "report.pdf", manifest_path: Path = MANIFEST) -> list[str]:
    """Compare the live tree to the sidecar written by make_report_pdf.py.

    Missing sidecar: legacy PDF — warn, do not fail (first rebuild writes it).
    Present sidecar: fail if a listed source hash or pdf_sha256 drifted.
    """
    issues: list[str] = []
    if not manifest_path.is_file():
        print(
            "report.sources.json missing — PDF freshness unverified "
            "(legacy PDF; rebuild with make_report_pdf.py to pin hashes)",
            file=sys.stderr,
        )
        return issues
    try:
        man = json.loads(manifest_path.read_text())
    except Exception as e:
        return [f"report.sources.json unreadable: {e}"]
    listed = man.get("sources") or {}
    live = current_pdf_sources()
    for rel, digest in listed.items():
        p = HERE / rel
        if not p.is_file():
            issues.append(f"PDF source missing: {rel}")
            continue
        got = sha256_file(p)
        if got != digest:
            issues.append(f"PDF source drifted: {rel}")
    for rel in live:
        if rel not in listed:
            issues.append(f"PDF source not in manifest (rebuild PDF): {rel}")
    want_pdf = man.get("pdf_sha256")
    if not isinstance(want_pdf, str) or not re.fullmatch(r"[0-9a-f]{64}", want_pdf):
        issues.append("report.sources.json has no valid pdf_sha256")
    if pdf.is_file():
        if want_pdf and sha256_file(pdf) != want_pdf:
            issues.append("report.pdf sha256 != report.sources.json pdf_sha256 — rebuild")
    else:
        issues.append("report.pdf missing")
    return issues


def check_pins(docs):
    """Auxiliary: formatted leaf == pin literal, and the literal appears somewhere.

    This is not occurrence-anchored. Prefer claims.json for headlines.
    """
    pf = HERE / "check_numbers_pins.txt"
    if not pf.exists():
        return 0
    text = "\n".join(d.read_text() for d in docs if d.is_file())
    bad = 0
    for line in pf.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            lit, src, fmt = line.split()
            fname, path = src.split(":", 1)
            v = json.loads((HERE / "results" / fname).read_text())
            for k in path.split("."):
                v = v[int(k)] if isinstance(v, list) else v[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("pin result is not a numeric value")
            if fmt.startswith("%"):
                v, fmt = 100 * v, fmt[1:]
            got = format(v, fmt)
        except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
            bad += 1
            print(f"PIN cannot resolve {line}: {error}")
            continue
        if got != lit:
            bad += 1
            print(f"PIN {lit} != {src} -> {got}")
        elif lit.lstrip("+") not in text:
            bad += 1
            print(f"PIN {lit} ({src}) no longer appears in the docs -- drop or update the pin")
    print(f"{bad} pin mismatches (auxiliary; claims.json is the fail-closed check)", file=sys.stderr)
    return bad


def lint_unexplained(docs, known, allow):
    bad = 0
    for doc in docs:
        if not doc.is_file():
            continue
        text = doc.read_text()
        for ln, line in enumerate(text.splitlines(), 1):
            if line.startswith("## References"):
                break
            if (line.lstrip().startswith(("![", "http", "```", "curl", "python3"))
                    or re.search(r"doi|DOI|arXiv|zenodo|\*Data\* 20|20\d\d \[", line)):
                continue
            for m in NUM.finditer(line):
                tok = m.group(0).replace(",", "").replace("−", "-").lstrip("-")
                if tok in known or tok in allow:
                    continue
                if re.fullmatch(r"\d", tok):
                    continue
                if re.fullmatch(r"2[23][01]\d{3}", tok):
                    continue
                if re.fullmatch(r"20[12]\d", tok) and re.search(r"\(20[12]\d\)|et al", line):
                    continue
                if line[m.start() - 1:m.start()] == "[" and re.search(r"\[[\d,–-]+\]", line[m.start() - 1:]):
                    continue
                bad += 1
                print(f"{doc.name}:{ln}: {m.group(0)!r}  | {line.strip()[:110]}")
    print(f"{bad} unexplained numbers (generic lint, not claim verification)", file=sys.stderr)
    return bad


def check_page_count_recommend(pdf: Path) -> None:
    """Competition asks for 15-20 body pages; recommendation only, not a hard fail."""
    try:
        import pypdf
    except ImportError:
        return
    if not pdf.is_file():
        return
    pages = pypdf.PdfReader(str(pdf)).pages
    ref_page = next((i for i, pg in enumerate(pages)
                     if "\nReferences\n" in "\n" + pg.extract_text() + "\n"
                     and "[1] Mov" in pg.extract_text()), len(pages) - 1)
    body = ref_page if pages[ref_page].extract_text().lstrip().startswith("References") else ref_page + 1
    if body > 20 or body < 15:
        print(
            f"note: report.pdf body is {body} pages (references start on page {ref_page + 1}); "
            f"the rules recommend 15-20 excluding references (not a hard checker fail)",
            file=sys.stderr,
        )


def main():
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    extra = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    numbers_only = "--numbers-only" in flags
    docs = DOCS + extra
    import check_results
    result_issues = check_results.check_results(here=HERE)
    for issue in result_issues:
        print(issue)
    print(f"{len(result_issues)} result integrity issues", file=sys.stderr)
    if result_issues:
        sys.exit(1)
    vals = json_values()
    known = set()
    for v in vals:
        known |= reps(v)
    # Named derived claims live in claims.json. Do not invent ratios of arbitrary leaves.
    allow = set()
    af = HERE / "check_numbers_allow.txt"
    if af.exists():
        for line in af.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            allow.update(line.replace(",", "").split())
    bad = lint_unexplained(docs, known, allow)
    bad += check_pins(docs)

    import check_claims
    claim_issues = check_claims.check_claims(here=HERE)
    for i in claim_issues:
        print(i)
    print(f"{len(claim_issues)} claim issues", file=sys.stderr)
    bad += len(claim_issues)

    if not numbers_only:
        bad += subprocess.call([sys.executable, str(HERE / "check_tables.py")])
        for msg in check_pdf_sources():
            print(msg, file=sys.stderr)
            bad += 1
        check_page_count_recommend(HERE / "report.pdf")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
