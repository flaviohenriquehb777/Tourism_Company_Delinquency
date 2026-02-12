from __future__ import annotations

import shutil
from pathlib import Path


def _ensure_favicon(html: str) -> str:
    if "rel=\"icon\"" in html or "rel='icon'" in html:
        return html
    if "<head" not in html:
        return html
    insert = "<link rel=\"icon\" type=\"image/x-icon\" href=\"./favicon.ico\"/>\n"
    if "</title>" in html:
        return html.replace("</title>", "</title>\n" + insert, 1)
    return html.replace("<head>", "<head>\n" + insert, 1)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    src_html = reports / "dashboard.html"
    src_ico = reports / "favicon.ico"
    src_md = reports / "respostas_empresa.md"
    if not src_html.exists():
        raise SystemExit("reports/dashboard.html não encontrado. Execute o notebook 03 antes.")
    if not src_ico.exists():
        raise SystemExit("reports/favicon.ico não encontrado.")
    if not src_md.exists():
        raise SystemExit("reports/respostas_empresa.md não encontrado. Gere com scripts/build_deliverables.py.")

    html = src_html.read_text(encoding="utf-8")
    (docs / "dashboard.html").write_text(_ensure_favicon(html), encoding="utf-8")
    shutil.copy2(src_ico, docs / "favicon.ico")
    shutil.copy2(src_md, docs / "respostas_empresa.md")
    (docs / ".nojekyll").write_text("", encoding="utf-8")

    print(str(docs / "dashboard.html"))


if __name__ == "__main__":
    main()

