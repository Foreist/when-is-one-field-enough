#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REPORT.md -> report.pdf (A4, print CSS, figures embedded)."""
import base64
import hashlib
import html as html_lib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
MD = HERE / "REPORT.md"
PDF = HERE / "report.pdf"

CSS = """
@page { size: A4; margin: 15mm 16mm; }
body { font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
       font-size: 9.8pt; line-height: 1.45; color: #1a1a1a; }
h2.refs { break-before: page; }
h2.refs ~ p { font-size: 8pt; line-height: 1.25; margin: 0.6mm 0; }
h1 { font-size: 19pt; margin: 0 0 4mm 0; line-height: 1.25; }
h2 { font-size: 13.5pt; margin: 8mm 0 2mm 0; border-bottom: 1px solid #ccc; padding-bottom: 1mm;
     page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 5mm 0 1.5mm 0; page-break-after: avoid; }
p, li { text-align: justify; }
body { font-variant-ligatures: none; }
a { overflow-wrap: anywhere; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 4mm 0; font-size: 8.8pt;
        }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 1.2mm 2mm; text-align: left; }
th { background: #f0f3f7; }
img { max-width: 100%; display: block; margin: 3mm auto 1mm auto; page-break-inside: avoid; }
em { color: #333; }
code { background: #f4f4f4; padding: 0 1px; font-size: 9pt; }
pre { background: #f7f7f7; border: 1px solid #ddd; padding: 2mm; font-size: 7.8pt; overflow-x: hidden;
      white-space: pre-wrap; }
hr { border: none; border-top: 1px solid #ddd; margin: 5mm 0; }
"""


def img_data_uri(p):
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


class Linkify(HTMLParser):
    """Make bare public URLs clickable without touching existing links or code."""
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        self.parts.append(self.get_starttag_text())
        if tag not in {"br", "hr", "img", "meta", "link", "input"}:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.parts.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        self.parts.append(f"</{tag}>")
        if tag in self.stack:
            self.stack = self.stack[:len(self.stack) - 1 - self.stack[::-1].index(tag)]

    def handle_data(self, data):
        if any(tag in self.stack for tag in ("a", "code", "pre", "script", "style")):
            self.parts.append(data)
            return
        def anchor(match):
            token = match.group()
            url = token.rstrip(".,;)")
            suffix = token[len(url):]
            return f'<a href="{html_lib.escape(url, quote=True)}">{url}</a>{suffix}'
        self.parts.append(re.sub(r"https?://[^\s<>]+", anchor, data))

    def handle_entityref(self, name):
        self.parts.append(f"&{name};")

    def handle_charref(self, name):
        self.parts.append(f"&#{name};")


def source_manifest():
    sources = [MD, Path(__file__)] + sorted((HERE / "figures").glob("*.png"))
    return {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources}


def main():
    before = source_manifest()
    text = MD.read_text()
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    links = Linkify()
    links.feed(html)
    html = "".join(links.parts)
    # inline the figures so the HTML is self-contained
    html = html.replace("<h2>References</h2>", '<h2 class="refs">References</h2>')  # smaller type for the reference list
    for png in sorted((HERE / "figures").glob("*.png")):
        html = html.replace(f'src="figures/{png.name}"', f'src="{img_data_uri(png)}"')
    doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    tmp = HERE / "_report.html"
    tmp.write_text(doc)
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_page()
        page.goto(tmp.as_uri())
        page.pdf(path=str(PDF), format="A4", print_background=True,
                 margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"})
        b.close()
    tmp.unlink()
    after = source_manifest()
    if before != after:
        raise RuntimeError("report sources changed during rendering; rebuild from a stable tree")
    manifest = dict(sources=after,
                    pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest())
    (HERE / "report.sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", PDF, f"({PDF.stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
