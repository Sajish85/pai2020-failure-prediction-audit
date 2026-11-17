"""Render PAPER-DRAFT.md to PAPER-DRAFT.pdf (CommonMark + tables via markdown-it-py -> HTML -> WeasyPrint).
Same renderer and CSS as s1/scripts/render-note-pdf.py, plus a base URL so figures/*.png resolve.
This is NOT Paper 1's PDF toolchain (not found on this Mac); Markdown is the deliverable of record.

Usage (from r1/):  ../s1/.venv/bin/python scripts/render-draft-pdf.py PAPER-DRAFT.md PAPER-DRAFT.pdf
"""
import pathlib
import sys

from markdown_it import MarkdownIt
from weasyprint import HTML

CSS = """
@page { size: A4; margin: 22mm 20mm 22mm 20mm;
        @bottom-center { content: counter(page) " / " counter(pages); font-size: 9pt; color: #555; } }
body { font-family: "Charter", "Georgia", serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 8pt; }
h2 { font-size: 13pt; margin: 16pt 0 6pt; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
p, li { text-align: left; }
blockquote { margin: 8pt 0; padding: 4pt 10pt; border-left: 3pt solid #999; background: #f4f4f4; }
code { font-family: "Menlo", monospace; font-size: 8.8pt; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8pt 0; }
th, td { border: 0.5pt solid #888; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #eaeaea; }
tr, blockquote { break-inside: avoid; }
img { max-width: 100%; display: block; margin: 6pt auto; }
a { color: #0b4f8a; text-decoration: none; }
"""

src, dst = pathlib.Path(sys.argv[1]).resolve(), sys.argv[2]
text = src.read_text(encoding="utf-8")
body = MarkdownIt("commonmark").enable("table").render(text)
title = text.splitlines()[0].lstrip("# ").strip()
html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
        f"<style>{CSS}</style></head><body>{body}</body></html>")
HTML(string=html, base_url=str(src.parent) + "/").write_pdf(dst)
print(f"Wrote {dst}")
