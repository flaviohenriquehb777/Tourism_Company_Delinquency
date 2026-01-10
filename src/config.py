from pathlib import Path
from datetime import date
import re
from PyPDF2 import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

INTEREST_RATE_MONTHLY = 0.0249
IMAGES_DIR = REPORTS_DIR / "images"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DEFAULT_FILES = {
    "xlsx": RAW_DIR / "Base Pagamentos.xlsx",
    "pdf": RAW_DIR / "Tourism Company Delinquency.pdf",
}

EVENTS = {
    "high_ticket_start": date(2021, 1, 1),
    "recurring_expansion": date(2021, 7, 1),
    "premium_offer": date(2024, 8, 1),
    "whatsapp_start": None,
}

COLUMN_PATTERNS = {
    "customer_id": [r"^id_?cliente", r"^cliente_?id", r"^customer", r"^id$"],
    "contract_id": [r"^id_?contrato", r"^contrato_?id", r"^contract"],
    "purchase_date": [r"^data_?compra", r"^purchase_?date", r"^data_?venda", r"^inicio_?compra"],
    "product_name": [r"^produto", r"^product"],
    "product_price": [r"^valor_?produto", r"^preco", r"^ticket", r"^valor_?total", r"^amount_?total"],
    "installments_total": [r"^n_?parcelas", r"^parcelas", r"^qtd_?parcelas", r"^installments"],
    "installment_value": [r"^valor_?parcela", r"^parcela_?valor", r"^installment_?amount"],
    "recurring_flag": [r"^recorrente", r"^parcelamento_?tipo", r"^tipo_?parcelamento", r"^recurring"],
    "interest_rate": [r"^juros", r"^taxa", r"^interest"],
    "payment_date": [r"^data_?pagamento", r"^payment_?date", r"^data_?recebimento"],
    "payment_amount": [r"^valor_?pago", r"^valor_?recebido", r"^payment_?amount"],
}

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def detect_whatsapp_start(pdf_path: Path) -> date | None:
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        m = re.search(r"WhatsApp.*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text, flags=re.I | re.S)
        if not m:
            m = re.search(r"WhatsApp.*?(\d{4}[/-]\d{1,2}[/-]\d{1,2})", text, flags=re.I | re.S)
        if m:
            s = m.group(1)
            parts = re.split(r"[/-]", s)
            if len(parts[0]) == 4:
                y, mo, d = map(int, parts)
            else:
                d, mo, y = map(int, parts)
                if y < 100:
                    y += 2000
            return date(y, mo, d).replace(day=1)
    except Exception:
        return None
    return None

def get_events():
    p = DEFAULT_FILES.get("pdf")
    w = detect_whatsapp_start(p) if p and p.exists() else None
    e = dict(EVENTS)
    e["whatsapp_start"] = w
    return e
