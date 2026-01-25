from pathlib import Path
import re
import pandas as pd
from .config import COLUMN_PATTERNS
from .io_utils import read_excel_any

def _first_match(cols, patterns):
    for pat in patterns:
        r = re.compile(pat, re.I)
        for c in cols:
            if r.search(c):
                return c
    return None

def detect_tables(sheets: dict[str, pd.DataFrame]):
    candidates = []
    for name, df in sheets.items():
        cols = set(df.columns)
        score_sales = 0
        if _first_match(cols, COLUMN_PATTERNS["purchase_date"]): score_sales += 1
        if _first_match(cols, COLUMN_PATTERNS["installments_total"]): score_sales += 1
        if _first_match(cols, COLUMN_PATTERNS["product_price"]): score_sales += 1
        score_pay = 0
        if _first_match(cols, COLUMN_PATTERNS["payment_date"]): score_pay += 1
        if _first_match(cols, COLUMN_PATTERNS["payment_amount"]): score_pay += 1
        candidates.append((name, score_sales, score_pay))
    sales_name = max(candidates, key=lambda x: x[1])[0]
    pay_name = max(candidates, key=lambda x: x[2])[0]
    return sales_name, pay_name

def normalize_sales(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    m = {}
    for k in ["customer_id","contract_id","purchase_date","product_name","product_price","installments_total","installment_value","recurring_flag","interest_rate"]:
        c = _first_match(cols, COLUMN_PATTERNS[k]) if k in COLUMN_PATTERNS else None
        m[k]=c
    out = pd.DataFrame()
    if m["customer_id"] in df: out["customer_id"] = df[m["customer_id"]]
    else: out["customer_id"] = pd.util.hash_pandas_object(df.index).astype(str)
    if m["contract_id"] in df: out["contract_id"] = df[m["contract_id"]]
    else: out["contract_id"] = out["customer_id"].astype(str)+"-"+pd.util.hash_pandas_object(df.index).astype(str)
    out["purchase_date"] = pd.to_datetime(df[m["purchase_date"]]) if m["purchase_date"] in df else pd.NaT
    out["product_name"] = df[m["product_name"]] if m["product_name"] in df else None
    out["product_price"] = pd.to_numeric(df[m["product_price"]], errors="coerce") if m["product_price"] in df else None
    out["installments_total"] = pd.to_numeric(df[m["installments_total"]], errors="coerce") if m["installments_total"] in df else None
    out["installment_value"] = pd.to_numeric(df[m["installment_value"]], errors="coerce") if m["installment_value"] in df else None
    if m["recurring_flag"] in df:
        v = df[m["recurring_flag"]].astype(str).str.lower()
        out["recurring"] = v.str.contains("recorr|recur|mensal|auto")
    else:
        out["recurring"] = False
    out["interest_rate"] = pd.to_numeric(df[m["interest_rate"]], errors="coerce") if m["interest_rate"] in df else None
    return out

def normalize_payments(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    pm = _first_match(cols, COLUMN_PATTERNS["payment_amount"])
    pdcol = _first_match(cols, COLUMN_PATTERNS["payment_date"])
    cid = _first_match(cols, COLUMN_PATTERNS["contract_id"]) or _first_match(cols, COLUMN_PATTERNS["customer_id"])
    out = pd.DataFrame()
    out["payment_date"] = pd.to_datetime(df[pdcol]) if pdcol in df else pd.NaT
    out["payment_amount"] = pd.to_numeric(df[pm], errors="coerce") if pm in df else None
    if cid in df: out["ref_id"] = df[cid].astype(str)
    else: out["ref_id"] = None
    return out

def load_base(xlsx_path: Path):
    sheets = read_excel_any(xlsx_path)
    if len(sheets) == 1:
        df0 = next(iter(sheets.values()))
        cols = set(map(str, df0.columns))
        if {"id_compra", "data_compra", "valor"}.issubset(cols):
            df = df0.copy()
            df["id_compra"] = df["id_compra"].astype(str)
            df["data_compra"] = pd.to_datetime(df["data_compra"], errors="coerce")
            df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

            total_rec = pd.to_numeric(df.get("total_parcelas_recorrentes"), errors="coerce").fillna(0)
            qtd_parc = pd.to_numeric(df.get("qtd_parcelas"), errors="coerce").fillna(1)
            recurring = total_rec > 0
            installments_total = total_rec.where(recurring, qtd_parc).fillna(1).astype(int).clip(lower=1)

            payments = pd.DataFrame(
                {
                    "payment_date": df["data_compra"],
                    "payment_amount": df["valor"],
                    "ref_id": df["id_compra"],
                }
            ).dropna(subset=["payment_date", "payment_amount", "ref_id"])

            customer_id = None
            if "e-mail_cliente" in df.columns:
                customer_id = df["e-mail_cliente"].astype(str)
            elif "nome_cliente" in df.columns:
                customer_id = df["nome_cliente"].astype(str)
            else:
                customer_id = pd.util.hash_pandas_object(df["id_compra"]).astype(str)

            df_sales_base = pd.DataFrame(
                {
                    "contract_id": df["id_compra"].astype(str),
                    "customer_id": customer_id,
                    "purchase_date": df["data_compra"],
                    "product_name": df.get("produto"),
                    "product_code": df.get("codigo_produto"),
                    "region": df.get("região"),
                    "gender": df.get("gênero"),
                    "payment_method": df.get("forma_pagamento"),
                    "installments_total": installments_total,
                    "installment_value": df["valor"],
                    "recurring": recurring,
                    "interest_rate": pd.NA,
                }
            )
            df_sales = (
                df_sales_base.groupby("contract_id", dropna=False)
                .agg(
                    customer_id=("customer_id", "first"),
                    purchase_date=("purchase_date", "min"),
                    product_name=("product_name", "first"),
                    product_code=("product_code", "first"),
                    region=("region", "first"),
                    gender=("gender", "first"),
                    payment_method=("payment_method", "first"),
                    installments_total=("installments_total", "max"),
                    installment_value=("installment_value", "median"),
                    recurring=("recurring", "max"),
                    interest_rate=("interest_rate", "first"),
                )
                .reset_index()
            )
            df_sales["product_price"] = df_sales["installment_value"] * df_sales["installments_total"]
            sales = df_sales.dropna(subset=["purchase_date"])
            return sales, payments

    sname, pname = detect_tables(sheets)
    sales = normalize_sales(sheets[sname]).dropna(subset=["purchase_date"])
    payments = normalize_payments(sheets[pname]).dropna(subset=["payment_date","payment_amount"])
    return sales, payments
