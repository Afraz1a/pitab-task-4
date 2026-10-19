"""Render final_report.md to report/final_report.pdf (plain, print-friendly)."""
import sys
import types
from pathlib import Path

CSS = """
@page { size: a4 portrait; margin: 1.6cm 1.4cm; @frame footer {
        -pdf-frame-content: footer; bottom: 0.8cm; margin-left: 1.4cm;
        margin-right: 1.4cm; height: 1cm; } }
body { font-family: Helvetica, Arial, sans-serif; font-size: 9.5pt;
       line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 16pt; color: #111; margin: 4pt 0 2pt 0; padding-bottom: 4pt;
     border-bottom: 2pt solid #333; }
h2 { font-size: 13.5pt; color: #111; margin: 16pt 0 5pt 0; padding: 4pt 0;
     border-top: 1.2pt solid #333; border-bottom: 0.5pt solid #bbb; }
h3 { font-size: 10.5pt; color: #222; margin: 10pt 0 3pt 0; }
p  { margin: 4pt 0; }
strong { color: #111; }
table { border: 0.5pt solid #bbb; margin: 6pt 0 10pt 0; width: 100%;
        -pdf-keep-in-frame-mode: shrink; }
th { background-color: #eeeeee; border: 0.5pt solid #bbb; padding: 4pt 5pt;
     font-size: 9pt; font-weight: bold; text-align: left; }
td { border: 0.5pt solid #d5d5d5; padding: 4pt 5pt; font-size: 9pt; }
ul, ol { margin: 4pt 0 4pt 14pt; }
li { margin: 1.5pt 0; }
hr { border-top: 0.5pt solid #ddd; margin: 10pt 0; }
a { color: #1a1a1a; text-decoration: none; }
code { font-family: Courier, monospace; font-size: 8.5pt; background-color: #f2f2f2; }
blockquote { margin: 4pt 0 4pt 10pt; padding-left: 8pt; border-left: 2pt solid #bbb; color: #444; }
.footer { font-size: 7.5pt; color: #888; text-align: center; }
"""

FOOTER = ('<div id="footer" class="footer">AIRI Team PITB &mdash; AI Internship Task 4 '
          '&mdash; CitizenSense Final Report &mdash; '
          'page <pdf:pagenumber> of <pdf:pagecount></div>')


def main() -> int:
    import markdown
    try:
        import bidi  # noqa: F401
    except Exception:
        stub = types.ModuleType("bidi")
        stub.get_display = lambda text, *a, **kw: text
        algo = types.ModuleType("bidi.algorithm")
        algo.get_display = stub.get_display
        stub.algorithm = algo
        sys.modules["bidi"] = stub
        sys.modules["bidi.algorithm"] = algo

    from xhtml2pdf import pisa

    here = Path(__file__).parent
    src = here / "final_report.md"
    out = here / "final_report.pdf"
    text = src.read_text(encoding="utf-8")
    for bad, good in (("—", "&mdash;"), ("–", "&ndash;"), ("±", "+/-"), ("’", "&rsquo;")):
        text = text.replace(bad, good)

    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    html = (f'<html><head><meta charset="utf-8">'
            f'<style>{CSS}</style></head><body>{FOOTER}{body}</body></html>')

    with out.open("wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
    if result.err:
        sys.exit(f"PDF generation reported {result.err} error(s)")

    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
