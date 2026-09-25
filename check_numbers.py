#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Every number in the prose must trace to results/*.json (or be declared a constant).

    python3 check_numbers.py [extra.md ...]      # exit 1 if any number is unexplained

A number in the text is explained if it equals, at the precision it is written, some leaf of
results/*.json (as a fraction, a percentage, or a count), a simple ratio/difference of two such
leaves, or an entry of check_numbers_allow.txt (design constants, citation years, figure numbers).
"""
import itertools, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = [HERE / "REPORT.md", HERE / "README.md", HERE / "audit" / "README.md"]
NUM = re.compile(r"(?<![\w.])[−-]?\d[\d,]*(?:\.\d+)?(?![\w])")


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
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
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
    return out


def main():
    docs = DOCS + [Path(p) for p in sys.argv[1:]]
    vals = json_values()
    small = sorted(v for v in vals if abs(v) < 1e5)
    known = set()
    for v in vals:
        known |= reps(v)
    # ratios and differences of two leaves (fold changes, pp gaps) -- limited to plausible magnitudes
    base = [v for v in small if 0 < abs(v) < 3000]
    for a, b in itertools.product(base[:4000], repeat=2) if len(base) < 1500 else []:
        if b and 1 < a / b < 20:
            known.add(f"{a / b:.1f}")
        if 0 < a - b < 1:
            known.add(f"{100 * (a - b):.1f}")
    allow = set()
    af = HERE / "check_numbers_allow.txt"
    if af.exists():
        for line in af.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            allow.update(line.replace(",", "").split())
    bad = 0
    for doc in docs:
        text = doc.read_text()
        for ln, line in enumerate(text.splitlines(), 1):
            if line.startswith("## References"):
                break                                  # bibliography: years, pages, DOIs
            if (line.lstrip().startswith(("![", "http", "```", "curl", "python3"))
                    or re.search(r"doi|DOI|arXiv|zenodo|\*Data\* 20|20\d\d \[", line)):
                continue
            for m in NUM.finditer(line):
                tok = m.group(0).replace(",", "").replace("−", "-").lstrip("-")
                if tok in known or tok in allow:
                    continue
                if re.fullmatch(r"\d", tok):          # single digits: counts, list numbers
                    continue
                if re.fullmatch(r"2[23][01]\d{3}", tok):  # session ids (yymmdd)
                    continue
                if re.fullmatch(r"20[12]\d", tok) and re.search(r"\(20[12]\d\)|et al", line):
                    continue                           # citation year in running text
                if line[m.start() - 1:m.start()] == "[" and re.search(r"\[[\d,–-]+\]", line[m.start() - 1:]):
                    continue                           # citation brackets [10,11]
                bad += 1
                print(f"{doc.name}:{ln}: {m.group(0)!r}  | {line.strip()[:110]}")
    print(f"\n{bad} unexplained numbers", file=sys.stderr)
    bad += check_pins(docs)
    sys.exit(1 if bad else 0)


def check_pins(docs):
    """Headline literals must still equal their JSON leaf and still appear in the docs.
    Decimals like 0.8xx match *some* leaf by chance, so the generic pass cannot catch a stale headline."""
    pf = HERE / "check_numbers_pins.txt"
    if not pf.exists():
        return 0
    text = "\n".join(d.read_text() for d in docs)
    bad = 0
    for line in pf.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        lit, src, fmt = line.split()
        fname, path = src.split(":", 1)
        v = json.loads((HERE / "results" / fname).read_text())
        for k in path.split("."):
            v = v[k]
        if fmt.startswith("%"):
            v, fmt = 100 * v, fmt[1:]
        got = format(v, fmt)
        if got != lit:
            bad += 1
            print(f"PIN {lit} != {src} -> {got}")
        elif lit.lstrip("+") not in text:
            bad += 1
            print(f"PIN {lit} ({src}) no longer appears in the docs -- drop or update the pin")
    print(f"{bad} pin mismatches", file=sys.stderr)
    return bad


if __name__ == "__main__":
    main()
