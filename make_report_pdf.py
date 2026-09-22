#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REPORT.md -> report.pdf (A4, print CSS, figures embedded)."""
import base64
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
MD = HERE / "REPORT.md"
PDF = HERE / "report.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
       font-size: 10pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 19pt; margin: 0 0 4mm 0; line-height: 1.25; }
h2 { font-size: 13.5pt; margin: 8mm 0 2mm 0; border-bottom: 1px solid #ccc; padding-bottom: 1mm;
     page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 5mm 0 1.5mm 0; page-break-after: avoid; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 4mm 0; font-size: 8.8pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 1.2mm 2mm; text-align: left; }
th { background: #f0f3f7; }
img { max-width: 100%; display: block; margin: 3mm auto 1mm auto; page-break-inside: avoid; }
em { color: #333; }
code { background: #f4f4f4; padding: 0 1px; font-size: 9pt; }
pre { background: #f7f7f7; border: 1px solid #ddd; padding: 2mm; font-size: 8.6pt; overflow-x: hidden;
      white-space: pre-wrap; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #ddd; margin: 5mm 0; }
"""


def img_data_uri(p):
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def main():
    text = MD.read_text()
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    # inline the figures so the HTML is self-contained
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
    print("wrote", PDF, f"({PDF.stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
