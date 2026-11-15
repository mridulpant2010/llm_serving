#!/usr/bin/env python3
"""Compile docs/Project_Report.md into docs/Project_Report.pdf using Microsoft Edge / Chrome."""

import subprocess
from pathlib import Path
import markdown


def main():
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    md_path = docs_dir / "Project_Report.md"
    html_path = docs_dir / "Project_Report.html"
    pdf_path = docs_dir / "Project_Report.pdf"

    if not md_path.exists():
        raise FileNotFoundError(f"Missing {md_path}")

    md_text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    css = """
    <style>
    @page {
        size: A4;
        margin: 18mm 18mm 18mm 18mm;
        @bottom-right {
            content: counter(page);
        }
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 10pt;
        line-height: 1.5;
        color: #1a1a1a;
    }
    h1 {
        font-size: 19pt;
        border-bottom: 2.5px solid #1f4e79;
        padding-bottom: 6px;
        color: #1f4e79;
        margin-top: 0;
        margin-bottom: 12pt;
    }
    h2 {
        font-size: 13.5pt;
        border-bottom: 1px solid #b0c4de;
        padding-bottom: 4px;
        color: #2b5797;
        margin-top: 18pt;
        margin-bottom: 8pt;
        page-break-after: avoid;
    }
    h3 {
        font-size: 11pt;
        color: #333333;
        margin-top: 12pt;
        margin-bottom: 6pt;
        page-break-after: avoid;
    }
    h4 {
        font-size: 10pt;
        color: #444444;
        margin-top: 8pt;
        page-break-after: avoid;
    }
    p, ul, ol {
        margin-top: 4pt;
        margin-bottom: 6pt;
    }
    li {
        margin-bottom: 3pt;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 10pt 0;
        font-size: 9pt;
        page-break-inside: avoid;
    }
    th, td {
        border: 1px solid #c0c0c0;
        padding: 5px 8px;
        text-align: left;
    }
    th {
        background-color: #f0f4f8;
        font-weight: 600;
        color: #1f4e79;
    }
    tr:nth-child(even) {
        background-color: #fafbfc;
    }
    code {
        font-family: "Consolas", "Courier New", monospace;
        background: #f4f5f7;
        padding: 1.5px 4px;
        font-size: 8.5pt;
        border-radius: 3px;
        border: 1px solid #e1e4e8;
    }
    pre {
        background: #f8f9fa;
        border: 1px solid #dcdcdc;
        padding: 8px 10px;
        font-size: 8.5pt;
        border-radius: 4px;
        overflow-x: auto;
        page-break-inside: avoid;
    }
    blockquote {
        border-left: 3.5px solid #2b5797;
        margin: 8pt 0;
        padding: 4pt 10pt;
        background: #f4f8fc;
        font-size: 9.5pt;
    }
    hr {
        border: 0;
        border-top: 1px solid #ddd;
        margin: 14pt 0;
    }
    </style>
    """

    full_html = f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n<title>Project Report - CS5903</title>\n{css}\n</head>\n<body>\n{body}\n</body>\n</html>"
    html_path.write_text(full_html, encoding="utf-8")
    print(f"Wrote HTML: {html_path}")

    edge_candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    edge_bin = next((str(p) for p in edge_candidates if p.exists()), "msedge")

    temp_data_dir = repo_root / ".temp_edge_user_data"
    temp_data_dir.mkdir(exist_ok=True)

    cmd = [
        edge_bin,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        f"--user-data-dir={temp_data_dir.resolve()}",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.resolve()}",
        str(html_path.resolve()),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if pdf_path.exists():
        print(f"Successfully generated {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
    else:
        print("PDF generation failed.")


if __name__ == "__main__":
    main()
