from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INTEREST_RATE_MONTHLY, PROC_DIR, REPORTS_DIR
from src.company_report import render_company_answers


def main() -> None:
    ctx_path = PROC_DIR / "analysis_context.json"
    if not ctx_path.exists():
        raise SystemExit("Arquivo data/processed/analysis_context.json não encontrado. Execute o notebook 03 antes.")

    context = json.loads(ctx_path.read_text(encoding="utf-8"))
    answers_md = REPORTS_DIR / "respostas_empresa.md"
    render_company_answers(context, answers_md, interest_rate=f"{INTEREST_RATE_MONTHLY*100:.2f}%".replace(".", ","))

    print(str(answers_md))


if __name__ == "__main__":
    main()

