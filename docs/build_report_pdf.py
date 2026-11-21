"""Build docs/Project_Report.pdf from docs/Project_Report.md.
Needs: pip install markdown playwright && playwright install chromium
Usage: python docs/build_report_pdf.py
"""
import pathlib, markdown
from playwright.sync_api import sync_playwright

D = pathlib.Path(__file__).resolve().parent
html_body = markdown.markdown((D / "Project_Report.md").read_text(encoding="utf-8"),
                              extensions=["tables", "fenced_code"])
css = """
@page { size: A4; margin: 16mm 15mm; }
body { font: 9.6pt/1.38 'Segoe UI', Arial, sans-serif; color: #111; }
h1 { font-size: 17pt; margin: 0 0 4px; } h2 { font-size: 12.5pt; margin: 14px 0 4px; border-bottom: 1px solid #ccc; padding-bottom: 2px; }
h3 { font-size: 10.5pt; margin: 10px 0 3px; } p { margin: 4px 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 8.4pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2px 5px; text-align: left; vertical-align: top; } th { background: #f0f0f0; }
img { width: 88%; display: block; margin: 6px auto; page-break-inside: avoid; }
code { font: 8.4pt Consolas, monospace; background: #f4f4f4; padding: 0 2px; }
"""
html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html_body}</body></html>"
tmp = D / "_report.html"; tmp.write_text(html, encoding="utf-8")
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto(tmp.as_uri()); pg.pdf(path=str(D / "Project_Report.pdf"), format="A4", print_background=True)
    b.close()
tmp.unlink(); print("wrote", D / "Project_Report.pdf")
