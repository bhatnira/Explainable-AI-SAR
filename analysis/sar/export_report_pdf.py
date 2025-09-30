import io
import os
from pathlib import Path

# Convert Markdown to HTML then to PDF

def md_to_pdf(md_path: Path, pdf_path: Path):
    try:
        from markdown import markdown
    except ImportError:
        raise SystemExit("Missing dependency: markdown. Install with `pip install markdown`." )

    try:
        from xhtml2pdf import pisa
    except ImportError:
        raise SystemExit("Missing dependency: xhtml2pdf. Install with `pip install xhtml2pdf`." )

    text = md_path.read_text(encoding="utf-8")

    html_body = markdown(
        text,
        extensions=[
            "extra",
            "sane_lists",
            "toc",
            "tables",
            "fenced_code",
            "codehilite",
        ],
        output_format="html5",
    )

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  @page {{ size: A4; margin: 1in; }}
  body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #111; }}
  h1, h2, h3, h4 {{ font-weight: 600; margin: 0.8em 0 0.4em; }}
  h1 {{ font-size: 18pt; }}
  h2 {{ font-size: 14pt; }}
  h3 {{ font-size: 12pt; }}
  p {{ margin: 0.2em 0 0.6em; }}
  ul, ol {{ margin: 0.2em 0 0.8em 1.4em; }}
  code, pre {{ font-family: Courier, monospace; font-size: 9pt; }}
  pre {{ background: #f6f8fa; padding: 8px; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.6em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 9pt; }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf_path.open("wb") as f:
        # xhtml2pdf accepts a unicode string directly
        result = pisa.CreatePDF(src=html, dest=f, encoding='utf-8')
        if result.err:
            raise SystemExit("Failed to create PDF from Markdown.")

    print(f"Wrote PDF: {pdf_path}")


if __name__ == "__main__":
    repo_root = Path(".").resolve()
    md_path = repo_root / "analysis" / "sar" / "REPORT.md"
    pdf_path = repo_root / "analysis" / "sar" / "REPORT.pdf"

    if not md_path.exists():
        raise SystemExit(f"Markdown report not found: {md_path}")

    md_to_pdf(md_path, pdf_path)
