from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROC_DIR, REPORTS_DIR


def _month(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").values.astype("datetime64[M]")


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(PROC_DIR / "sales_enriched.csv", parse_dates=["purchase_date"])
    allocated = pd.read_csv(PROC_DIR / "allocated.csv", parse_dates=["due_date", "payment_date_eff"])
    return sales, allocated


def _prep_alloc_dim(sales: pd.DataFrame, allocated: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    a = allocated.copy()

    for col in ["product_name", "region", "payment_method", "gender", "product_code"]:
        if col not in s.columns:
            s[col] = None

    a = a.merge(
        s[
            [
                "contract_id",
                "purchase_date",
                "product_name",
                "region",
                "payment_method",
                "premium",
            ]
        ],
        on="contract_id",
        how="left",
    )
    a["month"] = _month(a["due_date"])
    a["purchase_month"] = _month(a["purchase_date"])
    a["expected"] = pd.to_numeric(a["expected_amount"], errors="coerce").fillna(0.0)
    a["received"] = pd.to_numeric(a["paid_amount"], errors="coerce").fillna(0.0)
    a["recurring"] = a["recurring"].fillna(False)

    a["product_name"] = a["product_name"].fillna("Desconhecido").astype(str)
    a["region"] = a["region"].fillna("Desconhecida").astype(str)
    a["payment_method"] = a["payment_method"].fillna("Desconhecida").astype(str)

    g = (
        a.groupby(["month", "product_name", "region", "payment_method", "recurring"], as_index=False)
        .agg(expected=("expected", "sum"), received=("received", "sum"))
        .sort_values(["month", "product_name"])
    )
    g["month"] = pd.to_datetime(g["month"], errors="coerce").dt.to_period("M").astype(str)
    return g


def _prep_pay_dim(sales: pd.DataFrame, allocated: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    a = allocated.copy()

    for col in ["product_name", "region", "payment_method", "gender", "product_code"]:
        if col not in s.columns:
            s[col] = None

    a = a.merge(
        s[
            [
                "contract_id",
                "purchase_date",
                "product_name",
                "region",
                "payment_method",
                "premium",
            ]
        ],
        on="contract_id",
        how="left",
    )
    a = a[a["payment_date_eff"].notna()].copy()
    a["month"] = _month(a["payment_date_eff"])
    a["expected"] = 0.0
    a["received"] = pd.to_numeric(a["paid_amount"], errors="coerce").fillna(0.0)
    a["recurring"] = a["recurring"].fillna(False)

    a["product_name"] = a["product_name"].fillna("Desconhecido").astype(str)
    a["region"] = a["region"].fillna("Desconhecida").astype(str)
    a["payment_method"] = a["payment_method"].fillna("Desconhecida").astype(str)

    g = (
        a.groupby(["month", "product_name", "region", "payment_method", "recurring"], as_index=False)
        .agg(expected=("expected", "sum"), received=("received", "sum"))
        .sort_values(["month", "product_name"])
    )
    g["month"] = pd.to_datetime(g["month"], errors="coerce").dt.to_period("M").astype(str)
    return g


def _prep_sales_cube(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["contract_id"] = s["contract_id"].astype(str)
    s["customer_id"] = s["customer_id"].astype(str)
    s["recurring"] = s["recurring"].fillna(False).astype(bool)
    s["product_name"] = s["product_name"].fillna("Desconhecido").astype(str)
    s["region"] = s["region"].fillna("Desconhecida").astype(str)
    s["payment_method"] = s["payment_method"].fillna("Desconhecida").astype(str)

    all_dims = ["product_name", "region", "payment_method"]

    def build(groups: list[str], include_recurring: bool) -> pd.DataFrame:
        gcols = list(groups)
        if include_recurring:
            gcols = gcols + ["recurring"]
        if gcols:
            out = (
                s.groupby(gcols, dropna=False)
                .agg(
                    vendas=("contract_id", "nunique"),
                    clientes=("customer_id", "nunique"),
                    vendas_rec=("recurring", "sum"),
                )
                .reset_index()
            )
        else:
            out = pd.DataFrame(
                {
                    "vendas": [int(s["contract_id"].nunique())],
                    "clientes": [int(s["customer_id"].nunique())],
                    "vendas_rec": [int(s["recurring"].sum())],
                }
            )

        for d in all_dims:
            if d not in out.columns:
                out[d] = "*"

        if include_recurring:
            out["recurring_filter"] = out["recurring"].map(lambda x: "1" if bool(x) else "0")
            out = out.drop(columns=["recurring"])
        else:
            out["recurring_filter"] = "all"

        out["pct_recorrentes"] = out.apply(
            lambda r: float(r["vendas_rec"] / r["vendas"]) if float(r["vendas"]) else 0.0, axis=1
        )
        return out[["product_name", "region", "payment_method", "recurring_filter", "vendas", "clientes", "vendas_rec", "pct_recorrentes"]]

    group_sets: list[list[str]] = [
        [],
        ["product_name"],
        ["region"],
        ["payment_method"],
        ["product_name", "region"],
        ["product_name", "payment_method"],
        ["region", "payment_method"],
        ["product_name", "region", "payment_method"],
    ]

    frames = []
    for gs in group_sets:
        frames.append(build(gs, include_recurring=False))
        frames.append(build(gs, include_recurring=True))

    cube = pd.concat(frames, ignore_index=True)
    cube["vendas"] = pd.to_numeric(cube["vendas"], errors="coerce").fillna(0).astype(int)
    cube["clientes"] = pd.to_numeric(cube["clientes"], errors="coerce").fillna(0).astype(int)
    cube["vendas_rec"] = pd.to_numeric(cube["vendas_rec"], errors="coerce").fillna(0).astype(int)
    cube["pct_recorrentes"] = pd.to_numeric(cube["pct_recorrentes"], errors="coerce").fillna(0.0).astype(float)
    return cube


def _prep_sales_cohort_dim(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["contract_id"] = s["contract_id"].astype(str)
    s["recurring"] = s["recurring"].fillna(False).astype(bool)
    s["product_name"] = s["product_name"].fillna("Desconhecido").astype(str)
    s["region"] = s["region"].fillna("Desconhecida").astype(str)
    s["payment_method"] = s["payment_method"].fillna("Desconhecida").astype(str)
    s["cohort"] = _month(s["purchase_date"])
    s = s[s["cohort"].notna()].copy()

    g = (
        s.groupby(["cohort", "product_name", "region", "payment_method", "recurring"], dropna=False)
        .agg(vendas=("contract_id", "nunique"))
        .reset_index()
        .sort_values(["cohort", "product_name"])
    )
    g["cohort"] = pd.to_datetime(g["cohort"], errors="coerce").dt.to_period("M").astype(str)
    g["vendas"] = pd.to_numeric(g["vendas"], errors="coerce").fillna(0).astype(int)
    return g


def _prep_recurring_matured_dim(sales: pd.DataFrame, allocated: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    a = allocated.copy()

    for col in ["product_name", "region", "payment_method"]:
        if col not in s.columns:
            s[col] = None

    s["contract_id"] = s["contract_id"].astype(str)
    s["purchase_date"] = pd.to_datetime(s["purchase_date"], errors="coerce")
    s["installments_total"] = (
        pd.to_numeric(s.get("installments_total"), errors="coerce").fillna(0).astype(int).clip(lower=0)
    )
    s["recurring"] = s.get("recurring", False)
    s["recurring"] = s["recurring"].fillna(False).astype(bool)

    vmax = s["purchase_date"].max()
    if pd.isna(vmax):
        return pd.DataFrame(columns=["product_name", "region", "payment_method", "expected", "received"])

    s_rec = s[(s["recurring"] == True) & s["purchase_date"].notna()].copy()
    if len(s_rec) == 0:
        return pd.DataFrame(columns=["product_name", "region", "payment_method", "expected", "received"])

    months_diff = (vmax.year - s_rec["purchase_date"].dt.year) * 12 + (vmax.month - s_rec["purchase_date"].dt.month)
    s_rec["matured"] = months_diff.astype(int) >= s_rec["installments_total"].astype(int)

    dim = s_rec.loc[
        s_rec["matured"] == True,
        ["contract_id", "product_name", "region", "payment_method"],
    ].copy()
    if len(dim) == 0:
        return pd.DataFrame(columns=["product_name", "region", "payment_method", "expected", "received"])

    dim["product_name"] = dim["product_name"].fillna("Desconhecido").astype(str)
    dim["region"] = dim["region"].fillna("Desconhecida").astype(str)
    dim["payment_method"] = dim["payment_method"].fillna("Desconhecida").astype(str)

    a["contract_id"] = a["contract_id"].astype(str)
    a = a.merge(dim, on="contract_id", how="inner")
    a["expected"] = pd.to_numeric(a["expected_amount"], errors="coerce").fillna(0.0)
    a["received"] = pd.to_numeric(a["paid_amount"], errors="coerce").fillna(0.0)

    g = (
        a.groupby(["product_name", "region", "payment_method"], as_index=False)
        .agg(expected=("expected", "sum"), received=("received", "sum"))
        .sort_values(["product_name", "region", "payment_method"])
    )
    return g


def _prep_recurring_matured_fact(sales: pd.DataFrame, allocated: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    a = allocated.copy()

    for col in ["product_name", "region", "payment_method"]:
        if col not in s.columns:
            s[col] = None

    s["contract_id"] = s["contract_id"].astype(str)
    s["purchase_date"] = pd.to_datetime(s["purchase_date"], errors="coerce")
    s["installments_total"] = (
        pd.to_numeric(s.get("installments_total"), errors="coerce").fillna(0).astype(int).clip(lower=0)
    )
    s["recurring"] = s.get("recurring", False)
    s["recurring"] = s["recurring"].fillna(False).astype(bool)

    vmax = s["purchase_date"].max()
    if pd.isna(vmax):
        return pd.DataFrame(
            columns=["purchase_month", "product_name", "region", "payment_method", "expected", "received"]
        )

    s_rec = s[(s["recurring"] == True) & s["purchase_date"].notna()].copy()
    if len(s_rec) == 0:
        return pd.DataFrame(
            columns=["purchase_month", "product_name", "region", "payment_method", "expected", "received"]
        )

    months_diff = (vmax.year - s_rec["purchase_date"].dt.year) * 12 + (vmax.month - s_rec["purchase_date"].dt.month)
    s_rec["matured"] = months_diff.astype(int) >= s_rec["installments_total"].astype(int)
    s_rec["purchase_month"] = _month(s_rec["purchase_date"])

    dim = s_rec.loc[
        (s_rec["matured"] == True) & s_rec["purchase_month"].notna(),
        ["contract_id", "purchase_month", "product_name", "region", "payment_method"],
    ].copy()
    if len(dim) == 0:
        return pd.DataFrame(
            columns=["purchase_month", "product_name", "region", "payment_method", "expected", "received"]
        )

    dim["product_name"] = dim["product_name"].fillna("Desconhecido").astype(str)
    dim["region"] = dim["region"].fillna("Desconhecida").astype(str)
    dim["payment_method"] = dim["payment_method"].fillna("Desconhecida").astype(str)

    a["contract_id"] = a["contract_id"].astype(str)
    a = a.merge(dim, on="contract_id", how="inner")
    a["expected"] = pd.to_numeric(a["expected_amount"], errors="coerce").fillna(0.0)
    a["received"] = pd.to_numeric(a["paid_amount"], errors="coerce").fillna(0.0)

    g = (
        a.groupby(["purchase_month", "product_name", "region", "payment_method"], as_index=False)
        .agg(expected=("expected", "sum"), received=("received", "sum"))
        .sort_values(["purchase_month", "product_name"])
    )
    g["purchase_month"] = pd.to_datetime(g["purchase_month"], errors="coerce").dt.to_period("M").astype(str)
    return g


def _prep_cohort_dim(sales: pd.DataFrame, allocated: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    a = allocated.copy()

    for col in ["product_name", "region", "payment_method"]:
        if col not in s.columns:
            s[col] = None

    a = a.merge(
        s[["contract_id", "purchase_date", "product_name", "region", "payment_method"]],
        on="contract_id",
        how="left",
    )
    a = a[a["purchase_date"].notna() & a["due_date"].notna()].copy()
    a["cohort"] = _month(a["purchase_date"])
    a["due_month"] = _month(a["due_date"])
    a["age"] = (
        (a["due_month"].dt.year - a["cohort"].dt.year) * 12 + (a["due_month"].dt.month - a["cohort"].dt.month)
    ).astype(int)
    a = a[a["age"] >= 0]

    a["expected"] = pd.to_numeric(a["expected_amount"], errors="coerce").fillna(0.0)
    a["received"] = pd.to_numeric(a["paid_amount"], errors="coerce").fillna(0.0)
    a["product_name"] = a["product_name"].fillna("Desconhecido").astype(str)
    a["region"] = a["region"].fillna("Desconhecida").astype(str)
    a["payment_method"] = a["payment_method"].fillna("Desconhecida").astype(str)
    a["expected_rec"] = a["expected"].where(a["recurring"] == True, 0.0)

    g = (
        a.groupby(["cohort", "age", "product_name", "region", "payment_method", "recurring"], as_index=False)
        .agg(expected=("expected", "sum"), received=("received", "sum"), expected_rec=("expected_rec", "sum"))
        .sort_values(["cohort", "age"])
    )
    g["cohort"] = pd.to_datetime(g["cohort"], errors="coerce").dt.to_period("M").astype(str)
    cohort_p = pd.PeriodIndex(g["cohort"], freq="M")
    g["due_month"] = (cohort_p + g["age"].astype(int)).astype(str)
    return g


def _html(
    data_alloc_due: list[dict],
    data_alloc_pay: list[dict],
    data_cohort: list[dict],
    data_sales_cube: list[dict],
    data_sales_cohort: list[dict],
    data_rec_matured: list[dict],
    data_rec_matured_fact: list[dict],
) -> str:
    payload_alloc_due = json.dumps(data_alloc_due, ensure_ascii=False)
    payload_alloc_pay = json.dumps(data_alloc_pay, ensure_ascii=False)
    payload_cohort = json.dumps(data_cohort, ensure_ascii=False)
    payload_sales_cube = json.dumps(data_sales_cube, ensure_ascii=False)
    payload_sales_cohort = json.dumps(data_sales_cohort, ensure_ascii=False)
    payload_rec_matured = json.dumps(data_rec_matured, ensure_ascii=False)
    payload_rec_matured_fact = json.dumps(data_rec_matured_fact, ensure_ascii=False)

    tpl = """<!doctype html>
<html lang="pt-br">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Dashboard Pagamentos Recorrentes</title>
    <link rel="icon" type="image/x-icon" href="./favicon.ico"/>
    <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-locale-pt-br-latest.js"></script>
    <style>
      :root {{
        --bg: #0b1020;
        --panel: rgba(24, 28, 44, 0.92);
        --panel2: rgba(26, 32, 54, 0.92);
        --border: rgba(255,255,255,0.08);
        --text: #e8edf7;
        --muted: rgba(232,237,247,0.75);
        --accent: #7c5cff;
        --accent2: #63b3ed;
        --danger: #ef4444;
      }}
      html, body {{ height: 100%; margin: 0; background: radial-gradient(1200px 600px at 20% 0%, rgba(124,92,255,0.25), transparent 60%), radial-gradient(900px 500px at 90% 10%, rgba(99,179,237,0.18), transparent 55%), var(--bg); color: var(--text); font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
      .app {{ display: grid; grid-template-columns: 72px 1fr; height: 100%; }}
      .sidebar {{ background: rgba(10, 14, 28, 0.88); border-right: 1px solid var(--border); padding: 14px 8px; display:flex; flex-direction:column; gap:10px; }}
      .navbtn {{ width: 56px; height: 56px; border-radius: 14px; background: rgba(255,255,255,0.04); border: 1px solid var(--border); color: var(--text); cursor: pointer; display:flex; align-items:center; justify-content:center; font-size: 18px; }}
      .navbtn.active {{ border-color: rgba(124,92,255,0.65); box-shadow: 0 0 0 2px rgba(124,92,255,0.18) inset; }}
      .content {{ padding: 18px 18px 24px; overflow: auto; }}
      .topbar {{ display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom: 14px; }}
      .title {{ font-weight: 800; letter-spacing: 0.5px; font-size: 20px; }}
      .filters {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
      .filters select {{ background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: var(--text); border-radius: 12px; padding: 10px 12px; outline:none; }}
      .gridTop {{ display:grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: stretch; margin-bottom: 14px; }}
      .gridBottom {{ display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; align-items: stretch; }}
      .span2 {{ grid-column: 1 / span 2; }}
      .summaryCards {{ display:flex; flex-direction:column; gap: 8px; }}
      .summaryCards .card {{ padding: 8px 12px; }}
      .summaryCards .card .k {{ font-size: 11px; }}
      .summaryCards .card .v {{ font-size: 18px; margin-top: 2px; }}
      .summaryCards .card .s {{ font-size: 11px; margin-top: 3px; }}
      .panel {{ background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02)); border: 1px solid var(--border); border-radius: 18px; padding: 12px 12px; }}
      .panel h3 {{ margin: 6px 8px 10px; font-size: 13px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.7px; }}
      .cards {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 14px; }}
      .cards4 {{ display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 14px; }}
      .card {{ background: linear-gradient(180deg, rgba(124,92,255,0.10), rgba(255,255,255,0.02)); border: 1px solid var(--border); border-radius: 18px; padding: 14px 14px; }}
      .card .k {{ color: var(--muted); font-size: 12px; }}
      .card .v {{ font-size: 22px; font-weight: 800; margin-top: 4px; }}
      .card .s {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
      .rightbox {{ background: linear-gradient(180deg, rgba(99,179,237,0.10), rgba(255,255,255,0.02)); border: 1px solid var(--border); border-radius: 18px; padding: 14px; }}
      .rightbox h3 {{ margin: 0 0 10px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.7px; color: var(--muted); }}
      .metricrow {{ display:flex; justify-content:space-between; padding: 10px 0; border-bottom: 1px solid var(--border); }}
      .metricrow:last-child {{ border-bottom: none; }}
      .metricrow .lbl {{ color: var(--muted); }}
      .metricrow .val {{ font-weight: 800; }}
      .tab {{ display:none; }}
      .tab.active {{ display:block; }}
      .twoCols {{ display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
      .full {{ grid-column: 1 / -1; }}
      .plot {{ width: 100%; height: 360px; }}
      .plotTall {{ width: 100%; height: 520px; }}
      .plotScroll {{ width: 100%; height: 520px; overflow-x: auto; overflow-y: hidden; }}
      .plotInner {{ height: 520px; min-width: 100%; }}
      .tableWrap {{ overflow:auto; max-height: 520px; }}
      table {{ width:100%; border-collapse: collapse; font-size: 12px; }}
      thead th {{ position: sticky; top: 0; background: rgba(10,14,28,0.95); border-bottom: 1px solid var(--border); padding: 10px 8px; text-align:left; }}
      tbody td {{ border-bottom: 1px solid var(--border); padding: 8px; color: var(--text); }}
      .hint {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
    </style>
  </head>
  <body>
    <div class="app">
      <aside class="sidebar">
        <button class="navbtn active" data-tab="tab-geral" title="Informações Gerais">📊</button>
        <button class="navbtn" data-tab="tab-caixa" title="Fluxo de Caixa">💳</button>
        <button class="navbtn" data-tab="tab-cohort" title="Análise Cohort">👥</button>
        <button class="navbtn" data-tab="tab-cohort-det" title="Análise Cohort Detalhada">🧾</button>
        <button class="navbtn" data-tab="tab-respostas" title="Respostas">📝</button>
      </aside>
      <main class="content">
        <div class="topbar">
          <div class="title" id="topTitle">Informações Gerais</div>
          <div class="filters">
            <select id="fProduto"></select>
            <select id="fRegiao"></select>
            <select id="fForma"></select>
            <select id="fRecorrente"></select>
            <select id="fReset"><option value="0">Limpar seleção</option></select>
          </div>
        </div>

        <section id="tab-geral" class="tab active">
          <div class="cards">
            <div class="card"><div class="k">Receita Esperada (parcelas)</div><div class="v" id="kpiEsperada">—</div><div class="s">Soma do valor esperado</div></div>
            <div class="card"><div class="k">Receita Real (parcelas)</div><div class="v" id="kpiRecebida">—</div><div class="s">Soma do valor recebido</div></div>
            <div class="card"><div class="k">% Inadimplência (financeira)</div><div class="v" id="kpiInad">—</div><div class="s">1 - recebido/esperado</div></div>
          </div>
          <div class="gridTop">
            <div class="panel">
              <h3>% Inadimplência por Produto</h3>
              <div id="plotProd" class="plot"></div>
            </div>
            <div class="panel">
              <h3>% Inadimplência por Região</h3>
              <div id="plotReg" class="plot"></div>
            </div>
          </div>
          <div class="gridBottom">
            <div class="panel span2">
              <h3>% Inadimplência e Vendas Recorrentes por Forma de Pagamento</h3>
              <div id="plotForma" class="plot"></div>
            </div>
            <div class="summaryCards">
              <div class="card"><div class="k">Vendas (contratos)</div><div class="v" id="kpiVendas">—</div><div class="s">Total de vendas</div></div>
              <div class="card"><div class="k">Vendas Pgto Normal</div><div class="v" id="kpiVendasNorm">—</div><div class="s">Sem recorrência</div></div>
              <div class="card"><div class="k">Vendas Recorrentes</div><div class="v" id="kpiVendasRec">—</div><div class="s">Com recorrência</div></div>
              <div class="card"><div class="k">% Recorrentes</div><div class="v" id="kpiRecShare">—</div><div class="s">Recorrentes / total</div></div>
              <div class="card"><div class="k">Qtd Clientes</div><div class="v" id="kpiClientes">—</div><div class="s">Clientes únicos</div></div>
              <div class="card"><div class="k">Forma selecionada</div><div class="v" id="kpiForma">—</div><div class="s">Filtro atual</div></div>
            </div>
          </div>
        </section>

        <section id="tab-caixa" class="tab">
          <div class="cards">
            <div class="card"><div class="k">Receita Esperada (fluxo de caixa)</div><div class="v" id="kpiEspMensal">—</div><div class="s">Base: mês de vencimento</div></div>
            <div class="card"><div class="k">Receita Real (fluxo de caixa)</div><div class="v" id="kpiRecMensal">—</div><div class="s">Base: mês de pagamento</div></div>
            <div class="card"><div class="k">Inadimplência (fluxo de caixa)</div><div class="v" id="kpiInadMensal">—</div><div class="s">Esperado - recebido</div></div>
          </div>
          <div class="panel">
            <h3>Receita Esperada vs Receita Real (área) — gap = inadimplência</h3>
            <div id="plotCashWrap" class="plotScroll"><div id="plotCash" class="plotInner"></div></div>
          </div>
        </section>

        <section id="tab-cohort" class="tab">
          <div class="cards4">
            <div class="card"><div class="k">Receita Esperada Parc.</div><div class="v" id="kpiCohEspParc">—</div><div class="s">Recorrentes</div></div>
            <div class="card"><div class="k">Receita Real Parc.</div><div class="v" id="kpiCohRecParc">—</div><div class="s">Recorrentes</div></div>
            <div class="card"><div class="k">Vendas</div><div class="v" id="kpiCohVendas">—</div><div class="s">Total de vendas</div></div>
            <div class="card"><div class="k">% Recorrentes</div><div class="v" id="kpiCohPctRec">—</div><div class="s">Vendas recorrentes / vendas</div></div>
          </div>
          <div class="panel">
            <h3>Análise Cohort (% Inadimplência e % Recorrente por cohort de compra)</h3>
            <div id="plotCohSummary" class="plotTall"></div>
          </div>
        </section>

        <section id="tab-cohort-det" class="tab">
          <div class="twoCols">
            <div class="panel">
              <h3>Cohort selecionada (curva por maturidade)</h3>
              <div id="plotCohDetail" class="plotTall"></div>
              <div class="hint">Selecione a cohort no dropdown do gráfico.</div>
            </div>
            <div class="panel">
              <h3>Tabela (cohort → último mês observado)</h3>
              <div class="tableWrap">
                <table id="tblCoh"></table>
              </div>
            </div>
          </div>
        </section>

        <section id="tab-respostas" class="tab">
          <div class="panel">
            <h3>Respostas às perguntas do negócio</h3>
            <div id="mdRespostas" style="padding: 6px 8px; color: var(--text);"></div>
          </div>
        </section>
      </main>
    </div>

    <script>
      const dataAllocDue = __DATA_ALLOC_DUE__;
      const dataAllocPay = __DATA_ALLOC_PAY__;
      const dataCohort = __DATA_COHORT__;
      const dataSalesCube = __DATA_SALES_CUBE__;
      const dataSalesCohort = __DATA_SALES_COHORT__;
      const dataRecMatured = __DATA_REC_MATURED__;
      const dataRecMaturedFact = __DATA_REC_MATURED_FACT__;

      const state = {{
        product: null,
        region: null,
        method: null,
        recurring: null
      }};

      const PLOTLY_CONFIG = {{ displayModeBar: false, responsive: true }};
      try {{
        if (window.Plotly && Plotly.setPlotConfig) Plotly.setPlotConfig({{ locale: 'pt-br' }});
      }} catch (e) {{}}

      const fmtMoney = (v) => {{
        const n = Number(v || 0);
        return n.toLocaleString('pt-BR', {{ style: 'currency', currency: 'BRL', maximumFractionDigits: 0 }});
      }};
      const fmtPct = (v) => {{
        const n = Number(v || 0);
        return (n*100).toLocaleString('pt-BR', {{ minimumFractionDigits: 1, maximumFractionDigits: 1 }}) + '%';
      }};

      const uniq = (arr) => Array.from(new Set(arr));
      const byKey = (rows, k) => rows.map(r => r[k]);

      const toBool = (v) => v === true || v === 1 || v === '1' || v === 'true' || v === 'True';
      const todayMonth = () => new Date().toISOString().slice(0, 7);
      const cutoffMonth = (rows) => {{
        let maxM = null;
        for (const r of rows) {{
          if (!maxM || r.month > maxM) maxM = r.month;
        }}
        const t = todayMonth();
        if (!maxM) return t;
        return (maxM < t) ? maxM : t;
      }};
      const observedRows = (rows, mCutoff) => mCutoff ? rows.filter(r => r.month <= mCutoff) : rows;

      const filtersOk = (r, exceptKey) => {{
        if (r.region === 'Desconhecido' || r.region === 'Desconhecida') return false;
        if (exceptKey !== 'product' && state.product && r.product_name !== state.product) return false;
        if (exceptKey !== 'region' && state.region && r.region !== state.region) return false;
        if (exceptKey !== 'method' && state.method && r.payment_method !== state.method) return false;
        if (state.recurring === true && r.recurring !== true) return false;
        if (state.recurring === false && r.recurring !== false) return false;
        return true;
      }};

      const recFiltersOk = (r, exceptKey) => {{
        if (r.region === 'Desconhecido' || r.region === 'Desconhecida') return false;
        if (exceptKey !== 'product' && state.product && r.product_name !== state.product) return false;
        if (exceptKey !== 'region' && state.region && r.region !== state.region) return false;
        if (exceptKey !== 'method' && state.method && r.payment_method !== state.method) return false;
        return true;
      }};

      const cashFiltersOk = (r, exceptKey) => {{
        if (r.region === 'Desconhecido' || r.region === 'Desconhecida') return false;
        if (exceptKey !== 'product' && state.product && r.product_name !== state.product) return false;
        if (exceptKey !== 'region' && state.region && r.region !== state.region) return false;
        if (exceptKey !== 'method' && state.method && r.payment_method !== state.method) return false;
        return true;
      }};

      const salesKey = (p, r, m, recKey) => `${{p}}|${{r}}|${{m}}|${{recKey}}`;
      const SALES_CUBE = (() => {{
        const mp = new Map();
        for (const row of dataSalesCube) {{
          const k = salesKey(String(row.product_name), String(row.region), String(row.payment_method), String(row.recurring_filter));
          mp.set(k, row);
        }}
        return mp;
      }})();

      const allocDueRowsAll = (exceptKey) => dataAllocDue
        .map(r => ({{
          ...r,
          month: String(r.month),
          expected: Number(r.expected) || 0,
          received: Number(r.received) || 0,
          recurring: toBool(r.recurring),
        }}))
        .filter(r => filtersOk(r, exceptKey));

      const allocPayRowsAll = (exceptKey) => dataAllocPay
        .map(r => ({{
          ...r,
          month: String(r.month),
          expected: Number(r.expected) || 0,
          received: Number(r.received) || 0,
          recurring: toBool(r.recurring),
        }}))
        .filter(r => r.month && r.month !== 'NaT')
        .filter(r => filtersOk(r, exceptKey));

      const cashDueRowsAll = (exceptKey) => dataAllocDue
        .map(r => ({{
          ...r,
          month: String(r.month),
          expected: Number(r.expected) || 0,
          received: Number(r.received) || 0,
          recurring: toBool(r.recurring),
        }}))
        .filter(r => r.recurring === true)
        .filter(r => cashFiltersOk(r, exceptKey));

      const cashPayRowsAll = (exceptKey) => dataAllocPay
        .map(r => ({{
          ...r,
          month: String(r.month),
          expected: Number(r.expected) || 0,
          received: Number(r.received) || 0,
          recurring: toBool(r.recurring),
        }}))
        .filter(r => r.month && r.month !== 'NaT')
        .filter(r => r.recurring === true)
        .filter(r => cashFiltersOk(r, exceptKey));

      const recMaturedRowsAll = (exceptKey) => dataRecMatured
        .map(r => ({{
          ...r,
          expected: Number(r.expected) || 0,
          received: Number(r.received) || 0,
        }}))
        .filter(r => recFiltersOk(r, exceptKey));

      const toggleFilter = (key, value) => {{
        if (state[key] === value) state[key] = null;
        else state[key] = value;
        if (key === 'product') document.getElementById('fProduto').value = state.product || '';
        if (key === 'region') document.getElementById('fRegiao').value = state.region || '';
        if (key === 'method') document.getElementById('fForma').value = state.method || '';
        renderAll();
      }};

      const aggTotals = (rows) => {{
        let exp = 0, rec = 0, expRec = 0;
        for (const r of rows) {{
          exp += r.expected;
          rec += r.received;
          if (r.recurring) expRec += r.expected;
        }}
        const inad = exp > 0 ? Math.max(0, 1 - (rec/exp)) : 0;
        const recShare = exp > 0 ? (expRec/exp) : 0;
        return {{ exp, rec, inad, recShare }};
      }};

      const aggBy = (rows, key, topN=12) => {{
        const m = new Map();
        for (const r of rows) {{
          const k = r[key] || 'Desconhecido';
          const cur = m.get(k) || {{ exp:0, rec:0, expRec:0 }};
          cur.exp += r.expected;
          cur.rec += r.received;
          if (r.recurring) cur.expRec += r.expected;
          m.set(k, cur);
        }}
        const out = Array.from(m.entries()).map(([k,v]) => {{
          const inad = v.exp > 0 ? Math.max(0, 1 - v.rec/v.exp) : 0;
          const recShare = v.exp > 0 ? (v.expRec/v.exp) : 0;
          return {{ k, ...v, inad, recShare }};
        }});
        out.sort((a,b) => b.inad - a.inad);
        return out.slice(0, topN);
      }};

      const aggRecBy = (rows, key, topN=12) => {{
        const m = new Map();
        for (const r of rows) {{
          const k = r[key] || 'Desconhecido';
          const cur = m.get(k) || {{ exp:0, rec:0 }};
          cur.exp += r.expected;
          cur.rec += r.received;
          m.set(k, cur);
        }}
        const out = Array.from(m.entries()).map(([k,v]) => {{
          const inad = v.exp > 0 ? Math.max(0, 1 - v.rec/v.exp) : 0;
          return {{ k, ...v, inad }};
        }});
        out.sort((a,b) => b.inad - a.inad);
        return out.slice(0, topN);
      }};

      const aggMonthly = (rows) => {{
        const m = new Map();
        for (const r of rows) {{
          const k = r.month;
          const cur = m.get(k) || {{ exp:0, rec:0 }};
          cur.exp += r.expected;
          cur.rec += r.received;
          m.set(k, cur);
        }}
        const out = Array.from(m.entries()).map(([month,v]) => {{
          const inad = v.exp > 0 ? Math.max(0, 1 - v.rec/v.exp) : 0;
          return {{ month, ...v, inad }};
        }});
        out.sort((a,b) => a.month.localeCompare(b.month));
        return out;
      }};

      const buildFilters = () => {{
        const allProd = uniq(byKey(dataAllocDue, 'product_name')).sort();
        const allReg = uniq(byKey(dataAllocDue, 'region'))
          .map(v => (v === null || v === undefined) ? '' : String(v).trim())
          .filter(v => v && v !== 'Desconhecido' && v !== 'Desconhecida')
          .sort();
        const allMet = uniq(byKey(dataAllocDue, 'payment_method')).sort();

        const sel = (id, label, values) => {{
          const el = document.getElementById(id);
          el.innerHTML = '';
          const opt0 = document.createElement('option');
          opt0.value = '';
          opt0.textContent = label + ': Todos';
          el.appendChild(opt0);
          for (const v of values) {{
            const o = document.createElement('option');
            o.value = v;
            o.textContent = label + ': ' + v;
            el.appendChild(o);
          }}
        }};

        sel('fProduto', 'Produto', allProd);
        sel('fRegiao', 'Localidade', allReg);
        sel('fForma', 'Forma Pagamento', allMet);

        const rec = document.getElementById('fRecorrente');
        rec.innerHTML = '';
        [['', 'Tipo: Todos'], ['1','Tipo: Recorrente'], ['0','Tipo: Tradicional']].forEach(([v,t]) => {{
          const o = document.createElement('option');
          o.value = v;
          o.textContent = t;
          rec.appendChild(o);
        }});

        document.getElementById('fProduto').addEventListener('change', (e) => {{
          state.product = e.target.value || null;
          renderAll();
        }});
        document.getElementById('fRegiao').addEventListener('change', (e) => {{
          state.region = e.target.value || null;
          renderAll();
        }});
        document.getElementById('fForma').addEventListener('change', (e) => {{
          state.method = e.target.value || null;
          renderAll();
        }});
        document.getElementById('fRecorrente').addEventListener('change', (e) => {{
          if (e.target.value === '') state.recurring = null;
          else state.recurring = (e.target.value === '1');
          renderAll();
        }});

        document.getElementById('fReset').addEventListener('change', () => {{
          state.product = null;
          state.region = null;
          state.method = null;
          state.recurring = null;
          document.getElementById('fProduto').value = '';
          document.getElementById('fRegiao').value = '';
          document.getElementById('fForma').value = '';
          document.getElementById('fRecorrente').value = '';
          renderAll();
        }});
      }};

      const setKpis = () => {{
        const recRows = recMaturedRowsAll();
        const tRec = aggTotals(recRows.map(r => ({{...r, recurring: true}})));
        document.getElementById('kpiEsperada').textContent = fmtMoney(tRec.exp);
        document.getElementById('kpiRecebida').textContent = fmtMoney(tRec.rec);
        document.getElementById('kpiInad').textContent = fmtPct(tRec.inad);
        document.getElementById('kpiForma').textContent = state.method || 'Todas';

        const p = state.product || '*';
        const r = state.region || '*';
        const m = state.method || '*';
        const rowAll = SALES_CUBE.get(salesKey(p, r, m, 'all'));
        const vendas = Number(rowAll?.vendas || 0);
        const clientes = Number(rowAll?.clientes || 0);
        const vendasRec = Number(rowAll?.vendas_rec || 0);
        const vendasNorm = Math.max(0, vendas - vendasRec);
        const pctRec = vendas > 0 ? (vendasRec / vendas) : 0;
        document.getElementById('kpiVendas').textContent = vendas.toLocaleString('pt-BR');
        document.getElementById('kpiVendasNorm').textContent = vendasNorm.toLocaleString('pt-BR');
        document.getElementById('kpiVendasRec').textContent = vendasRec.toLocaleString('pt-BR');
        document.getElementById('kpiClientes').textContent = clientes.toLocaleString('pt-BR');
        document.getElementById('kpiRecShare').textContent = fmtPct(pctRec);
      }};

      const renderProd = () => {{
        const rows = recMaturedRowsAll('product');
        const agg = aggRecBy(rows, 'product_name', 10);
        const yFull = agg.map(x => x.k).reverse();
        const x = agg.map(x => x.inad*100).reverse();
        const maxX = x.length ? Math.max(...x, 0) : 0;
        const txt = x.map(v => Number(v || 0).toLocaleString('pt-BR', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}) + '%');
        const y = yFull.map((s) => {{
          const v = String(s || '');
          return v.length > 28 ? (v.slice(0, 28) + '...') : v;
        }});
        const cd = agg.map(x => [x.k, x.exp, x.rec]).reverse();
        const colors = yFull.map(name =>
          state.product ? (name === state.product ? 'rgba(124, 92, 255, 0.95)' : 'rgba(124, 92, 255, 0.30)') : 'rgba(124, 92, 255, 0.75)'
        );

        const fig = {{
          data: [{{
            type: 'bar',
            orientation: 'h',
            x: x,
            y: y,
            text: txt,
            textposition: 'outside',
            textangle: 0,
            textfont: {{ color: 'rgba(232, 237, 247, 0.92)', size: 12 }},
            cliponaxis: false,
            marker: {{ color: colors }},
            customdata: cd,
            hoverlabel: {{ namelength: -1 }},
            hovertemplate: 'Produto: %{customdata[0]}<br>% Inadimplência: %{x:.2f}%<br>Esperado: R$ %{customdata[1]:,.0f}<br>Recebido: R$ %{customdata[2]:,.0f}<extra></extra>'
          }}],
          layout: {{
            margin: {{l: 170, r: 30, t: 10, b: 10}},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: {{ showticklabels: false, showgrid: false, zeroline: false, title: '', range: [0, maxX*1.12] }},
            yaxis: {{ automargin: true, tickfont: {{ size: 12 }} }},
            bargap: 0.28,
            height: 340,
            autosize: true
          }},
          config: PLOTLY_CONFIG
        }};
        Plotly.react('plotProd', fig.data, fig.layout, fig.config);
        const div = document.getElementById('plotProd');
        if (div && div.removeAllListeners) div.removeAllListeners('plotly_click');
        div.on('plotly_click', (ev) => {{
          const v = ev.points?.[0]?.customdata?.[0];
          if (!v) return;
          toggleFilter('product', v);
        }});
      }};

      const renderReg = () => {{
        const rows = recMaturedRowsAll('region');
        const agg = aggRecBy(rows, 'region', 12);
        const y = agg.map(x => x.k).reverse();
        const x = agg.map(x => x.inad*100).reverse();
        const maxX = x.length ? Math.max(...x, 0) : 0;
        const txt = x.map(v => Number(v || 0).toLocaleString('pt-BR', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}) + '%');
        const cd = agg.map(x => [x.exp, x.rec]).reverse();
        const colors = y.map(name =>
          state.region ? (name === state.region ? 'rgba(99, 179, 237, 0.95)' : 'rgba(99, 179, 237, 0.28)') : 'rgba(99, 179, 237, 0.75)'
        );
        Plotly.react('plotReg', [{{
          type:'bar', orientation:'h',
          x, y, marker:{{color: colors}},
          text: txt,
          textposition: 'outside',
          textangle: 0,
          textfont: {{ color: 'rgba(232, 237, 247, 0.92)', size: 12 }},
          cliponaxis: false,
          customdata: cd,
          hovertemplate: 'Região: %{y}<br>% Inadimplência: %{x:.2f}%<br>Esperado: R$ %{customdata[0]:,.0f}<br>Recebido: R$ %{customdata[1]:,.0f}<extra></extra>'
        }}], {{
          margin: {{l: 160, r: 30, t: 10, b: 10}},
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          xaxis: {{ showticklabels: false, showgrid: false, zeroline: false, title: '', range: [0, maxX*1.12] }},
          height: 340,
          autosize: true
        }}, PLOTLY_CONFIG);
        const div = document.getElementById('plotReg');
        if (div && div.removeAllListeners) div.removeAllListeners('plotly_click');
        div.on('plotly_click', (ev) => {{
          const v = ev.points?.[0]?.y;
          if (!v) return;
          toggleFilter('region', v);
        }});
      }};

      const renderForma = () => {{
        const rows = recMaturedRowsAll('method');
        const agg = aggRecBy(rows, 'payment_method', 12);
        agg.sort((a,b) => b.exp - a.exp);
        const x = agg.map(x => x.k);
        const inad = agg.map(x => x.inad*100);
        const inadTxt = inad.map(v => Number(v || 0).toLocaleString('pt-BR', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}) + '%');
        const vendasRec = x.map((met) => {{
          const p = state.product || '*';
          const r = state.region || '*';
          const row = SALES_CUBE.get(salesKey(p, r, met, 'all'));
          return Number(row?.vendas_rec || 0);
        }});
        const vendasTxt = vendasRec.map(v => {{
          const n = Number(v || 0);
          if (n >= 1_000_000) return (n/1_000_000).toLocaleString('pt-BR', {{ maximumFractionDigits: 1 }}) + ' Mi';
          if (n >= 1_000) return (n/1_000).toLocaleString('pt-BR', {{ maximumFractionDigits: 1 }}) + ' Mil';
          return n.toLocaleString('pt-BR');
        }});
        const cd = agg.map(x => [x.exp, x.rec]);
        const barColors = x.map(name =>
          state.method ? (name === state.method ? 'rgba(124, 92, 255, 0.90)' : 'rgba(124, 92, 255, 0.28)') : 'rgba(124, 92, 255, 0.65)'
        );
        Plotly.react('plotForma', [
          {{
            type: 'bar',
            x, y: inad,
            name: '% Inadimplência',
            marker: {{ color: barColors }},
            text: inadTxt,
            textposition: 'inside',
            insidetextanchor: 'middle',
            textfont: {{ color: 'rgba(232, 237, 247, 0.92)', size: 12 }},
            customdata: cd,
            hovertemplate: 'Forma: %{x}<br>% Inadimplência: %{y:.2f}%<br>Esperado: R$ %{customdata[0]:,.0f}<br>Recebido: R$ %{customdata[1]:,.0f}<extra></extra>'
          }},
          {{
            type: 'scatter',
            x, y: vendasRec,
            name: 'Vendas Recorrentes',
            mode: 'lines+markers+text',
            text: vendasTxt,
            textposition: 'top center',
            textfont: {{ color: 'rgba(232, 237, 247, 0.92)', size: 12 }},
            yaxis: 'y2',
            line: {{ color: 'rgba(232, 237, 247, 0.85)' }},
            hovertemplate: 'Forma: %{x}<br>Vendas recorrentes: %{y:,.0f}<extra></extra>'
          }}
        ], {{
          margin: {{l: 60, r: 60, t: 10, b: 40}},
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          barmode: 'group',
          yaxis: {{ showticklabels: false, showgrid: false, zeroline: false }},
          yaxis2: {{ overlaying: 'y', side: 'right', showgrid: false, showticklabels: false, zeroline: false }},
          legend: {{ orientation: 'h', y: 1.2 }},
          height: 360,
          autosize: true
        }}, PLOTLY_CONFIG);
        const div = document.getElementById('plotForma');
        if (div && div.removeAllListeners) div.removeAllListeners('plotly_click');
        div.on('plotly_click', (ev) => {{
          const v = ev.points?.[0]?.x;
          if (!v) return;
          toggleFilter('method', v);
        }});
      }};

      const renderCash = () => {{
        const dueAll = cashDueRowsAll();
        const payAll = cashPayRowsAll();
        const expAgg = aggMonthly(dueAll);
        const recAgg = aggMonthly(payAll);
        const expMap = new Map(expAgg.map(x => [x.month, x.exp]));
        const recMap = new Map(recAgg.map(x => [x.month, x.rec]));
        const monthsAll = uniq([...Array.from(expMap.keys()), ...Array.from(recMap.keys())]).sort();
        const months = monthsAll;
        const xDates = months.map(m => String(m).slice(0, 7) + '-01');
        const wrap = document.getElementById('plotCashWrap');
        const wrapW = wrap ? wrap.clientWidth : 0;
        const targetW = Math.max(wrapW || 0, Math.min(2600, Math.max(1100, xDates.length * 36)));

        const expected = months.map(m => expMap.get(m) || 0);
        const received = months.map(m => recMap.get(m) || 0);
        const gap = months.map((m, i) => expected[i] - received[i]);

        const sumE = expected.reduce((a, x) => a + x, 0);
        const sumR = received.reduce((a, x) => a + x, 0);
        const sumG = gap.reduce((a, x) => a + x, 0);
        document.getElementById('kpiEspMensal').textContent = fmtMoney(sumE);
        document.getElementById('kpiRecMensal').textContent = fmtMoney(sumR);
        document.getElementById('kpiInadMensal').textContent = fmtMoney(sumG);

        const custom = months.map((m, i) => {{
          const e = expected[i] || 0;
          const r = received[i] || 0;
          const g = e - r;
          const inadPct = e > 0 ? (1 - (r / e)) : 0;
          return [r, e, g, inadPct];
        }});

        Plotly.react('plotCash', [
          {{
            type: 'scatter',
            x: xDates, y: received,
            name: 'Receita Real Fluxo de Caixa',
            mode: 'lines',
            fill: 'tozeroy',
            line: {{ color: 'rgba(124, 92, 255, 1)', width: 2 }},
            fillcolor: 'rgba(124, 92, 255, 0.25)',
            customdata: custom,
            hovertemplate: 'Mês: %{x|%m/%Y}<br>Recebido: R$ %{customdata[0]:,.0f}<br>Esperado: R$ %{customdata[1]:,.0f}<br>Inadimplência (R$): %{customdata[2]:,.0f}<br>% Inadimplência: %{customdata[3]:.1%}<extra></extra>'
          }},
          {{
            type: 'scatter',
            x: xDates, y: expected,
            mode: 'lines',
            fill: 'tozeroy',
            line: {{ color: 'rgba(233, 214, 170, 0.95)', width: 2 }},
            fillcolor: 'rgba(233, 214, 170, 0.28)',
            customdata: custom,
            hovertemplate: 'Mês: %{x|%m/%Y}<br>Recebido: R$ %{customdata[0]:,.0f}<br>Esperado: R$ %{customdata[1]:,.0f}<br>Inadimplência (R$): %{customdata[2]:,.0f}<br>% Inadimplência: %{customdata[3]:.1%}<extra></extra>',
            name: 'Receita Esperada Fluxo de Caixa'
          }}
        ], {{
          margin: {{l: 60, r: 20, t: 10, b: 40}},
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          legend: {{ orientation: 'h', y: 1.2 }},
          yaxis: {{ title: 'R$', gridcolor: 'rgba(255,255,255,0.08)' }},
          xaxis: {{
            type: 'date',
            gridcolor: 'rgba(255,255,255,0.06)',
            tickformat: '%m/%Y'
          }},
          height: 520,
          width: targetW,
          autosize: false
        }}, PLOTLY_CONFIG);
      }};

      const cohortRows = (cutoffMonth) => dataCohort
        .map(r => ({{
          ...r,
          cohort: String(r.cohort),
          due_month: String(r.due_month),
          expected: Number(r.expected) || 0,
          received: Number(r.received) || 0,
          expected_rec: Number(r.expected_rec) || 0,
          recurring: toBool(r.recurring),
        }}))
        .filter(r => {{
          if (state.product && r.product_name !== state.product) return false;
          if (state.region && r.region !== state.region) return false;
          if (state.method && r.payment_method !== state.method) return false;
          if (state.recurring === true && r.recurring !== true) return false;
          if (state.recurring === false && r.recurring !== false) return false;
          if (cutoffMonth && r.due_month && r.due_month > cutoffMonth) return false;
          return true;
        }});

      const cohortAgg = (rows) => {{
        const m = new Map();
        for (const r of rows) {{
          const k = r.cohort + '|' + r.age;
          const cur = m.get(k) || {{ cohort:r.cohort, age:Number(r.age), expected:0, received:0, expected_rec:0 }};
          cur.expected += r.expected;
          cur.received += r.received;
          cur.expected_rec += r.expected_rec;
          m.set(k, cur);
        }}
        const out = Array.from(m.values());
        out.sort((a,b) => (a.cohort.localeCompare(b.cohort) || a.age - b.age));
        return out;
      }};

      const cohortSummary = (rows) => {{
        const agg = cohortAgg(rows);
        const byC = new Map();
        for (const r of agg) {{
          const cur = byC.get(r.cohort) || [];
          cur.push(r);
          byC.set(r.cohort, cur);
        }}
        const out = [];
        for (const [c, arr] of byC.entries()) {{
          arr.sort((a,b) => a.age-b.age);
          let cumE=0, cumR=0;
          for (const r of arr) {{
            cumE += r.expected; cumR += r.received;
          }}
          const inad = cumE > 0 ? Math.max(0, 1 - cumR/cumE) : 0;
          out.push({{ cohort:c, expected:cumE, received:cumR, inad }});
        }}
        out.sort((a,b) => a.cohort.localeCompare(b.cohort));
        return out;
      }};

      const renderCohort = () => {{
        const cutoff = cutoffMonth(allocPayRowsAll()) || cutoffMonth(allocDueRowsAll());
        const rows = cohortRows(cutoff);
        const sum0 = cohortSummary(rows);
        const COH_START = '2020-09';
        const COH_END = '2024-12';
        const sum = sum0.filter(r => r.cohort >= COH_START && r.cohort <= COH_END);
        const rowsInRange = rows.filter(r => String(r.cohort) >= COH_START && String(r.cohort) <= COH_END);

        const x = sum.map(r => r.cohort);
        const xDates = x.map(c => String(c).slice(0, 7) + '-01');
        const inadY = sum.map(r => r.inad*100);

        let expParc = 0, recParc = 0;
        for (const r of dataRecMaturedFact) {{
          const pm = String(r.purchase_month);
          if (pm < COH_START || pm > COH_END) continue;
          if (state.product && String(r.product_name) !== state.product) continue;
          if (state.region && String(r.region) !== state.region) continue;
          if (state.method && String(r.payment_method) !== state.method) continue;
          expParc += Number(r.expected) || 0;
          recParc += Number(r.received) || 0;
        }}

        let vendas = 0, vendasRec = 0;
        for (const r of dataSalesCohort) {{
          const c = String(r.cohort);
          if (c < COH_START || c > COH_END) continue;
          const product = String(r.product_name);
          const region = String(r.region);
          const method = String(r.payment_method);
          if (state.product && product !== state.product) continue;
          if (state.region && region !== state.region) continue;
          if (state.method && method !== state.method) continue;
          const v = Number(r.vendas) || 0;
          vendas += v;
          if (toBool(r.recurring)) vendasRec += v;
        }}
        const pctRec = vendas > 0 ? (vendasRec / vendas) : 0;
        document.getElementById('kpiCohEspParc').textContent = fmtMoney(expParc);
        document.getElementById('kpiCohRecParc').textContent = fmtMoney(recParc);
        document.getElementById('kpiCohVendas').textContent = vendas.toLocaleString('pt-BR');
        document.getElementById('kpiCohPctRec').textContent = fmtPct(pctRec);

        const recMap = new Map();
        for (const r of dataSalesCohort) {{
          const cohort = String(r.cohort);
          if (cohort < COH_START || cohort > COH_END) continue;
          const product = String(r.product_name);
          const region = String(r.region);
          const method = String(r.payment_method);
          if (state.product && product !== state.product) continue;
          if (state.region && region !== state.region) continue;
          if (state.method && method !== state.method) continue;
          const recFlag = toBool(r.recurring);
          if (state.recurring === true && recFlag !== true) continue;
          if (state.recurring === false && recFlag !== false) continue;
          const cur = recMap.get(cohort) || {{ total: 0, rec: 0 }};
          const v = Number(r.vendas) || 0;
          cur.total += v;
          if (recFlag) cur.rec += v;
          recMap.set(cohort, cur);
        }}
        const recShareArr = sum.map(r => {{
          const cur = recMap.get(r.cohort);
          if (!cur || !cur.total) return 0;
          return cur.rec / cur.total;
        }});
        const recY = recShareArr.map(v => v*100);
        Plotly.react('plotCohSummary', [
          {{
            type:'scatter',
            x: xDates, y: inadY,
            name:'% Inadimplência',
            mode:'lines',
            fill:'tozeroy',
            line:{{color:'rgba(239, 68, 68, 1)', width:2}},
            fillcolor:'rgba(239, 68, 68, 0.20)',
            customdata: sum.map(r => [r.received, r.expected]),
            hovertemplate:'Mês: %{x|%m/%Y}<br>% Inadimplência: %{y:.2f}%<br>Recebido (acum.): R$ %{customdata[0]:,.0f}<br>Esperado (acum.): R$ %{customdata[1]:,.0f}<extra></extra>'
          }},
          {{
            type:'scatter',
            x: xDates, y: recY,
            name:'% Recorrentes',
            mode:'lines',
            fill:'tozeroy',
            line:{{color:'rgba(99, 179, 237, 1)', width:2}},
            fillcolor:'rgba(99, 179, 237, 0.16)',
            hovertemplate:'Mês: %{x|%m/%Y}<br>% Recorrentes: %{y:.2f}%<extra></extra>'
          }}
        ], {{
          margin: {{l: 60, r: 20, t: 10, b: 40}},
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          legend: {{ orientation: 'h', y: 1.2 }},
          yaxis: {{ title: '%', gridcolor: 'rgba(255,255,255,0.08)' }},
          xaxis: {{
            type: 'date',
            range: ['2020-09-01', '2024-12-31'],
            tickformat: '%m/%Y',
            gridcolor: 'rgba(255,255,255,0.06)'
          }},
          height: 520,
          autosize: true
        }}, PLOTLY_CONFIG);

        const tbl = document.getElementById('tblCoh');
        tbl.innerHTML = '';
        const thead = document.createElement('thead');
        thead.innerHTML = '<tr><th>Cohort</th><th>Receita Esperada</th><th>Receita Real</th><th>% Inadimplência</th><th>% Recorrentes</th></tr>';
        tbl.appendChild(thead);
        const tbody = document.createElement('tbody');
        for (const r of sum) {{
          const idx = sum.findIndex(x => x.cohort === r.cohort);
          const recShare = (idx >= 0) ? recShareArr[idx] : 0;
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${{r.cohort}}</td><td>${{fmtMoney(r.expected)}}</td><td>${{fmtMoney(r.received)}}</td><td>${{fmtPct(r.inad)}}</td><td>${{fmtPct(recShare)}}</td>`;
          tbody.appendChild(tr);
        }}
        tbl.appendChild(tbody);

        renderCohortDetail(rowsInRange);
      }};

      const renderCohortDetail = (rows) => {{
        const agg = cohortAgg(rows);
        const cohorts = uniq(agg.map(r => r.cohort)).sort();
        if (!cohorts.length) {{
          Plotly.react('plotCohDetail', [], {{paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)'}}, PLOTLY_CONFIG);
          return;
        }}

        const byC = (c) => agg.filter(r => r.cohort === c).sort((a,b) => a.age - b.age);
        const build = (c) => {{
          const arr = byC(c);
          let cumE=0, cumR=0, cumER=0;
          const x = [];
          const inad = [];
          const rec = [];
          const cd = [];
          for (const r of arr) {{
            cumE += r.expected; cumR += r.received; cumER += r.expected_rec;
            x.push(r.age);
            inad.push((cumE>0 ? Math.max(0, 1-cumR/cumE) : 0)*100);
            rec.push((cumE>0 ? (cumER/cumE) : 0)*100);
            cd.push([cumR, cumE]);
          }}
          return {{ x, inad, rec, cd }};
        }};

        const c0 = cohorts[cohorts.length-1];
        const d0 = build(c0);

        const buttons = cohorts.map(c => {{
          const d = build(c);
          return {{
            label: c,
            method: 'update',
            args: [
              {{ x: [d.x, d.x], y: [d.inad, d.rec], customdata: [d.cd, d.cd] }},
              {{ title: 'Cohort detalhada — ' + c }}
            ]
          }};
        }});

        Plotly.react('plotCohDetail', [
          {{
            type:'scatter',
            x: d0.x, y: d0.inad,
            name:'% Inadimplência (acum.)',
            mode:'lines',
            fill:'tozeroy',
            line:{{color:'rgba(239, 68, 68, 1)', width:2}},
            fillcolor:'rgba(239, 68, 68, 0.18)',
            customdata: d0.cd,
            hovertemplate:'Cohort: ' + c0 + '<br>Mês após compra: %{x}<br>% Inadimplência: %{y:.2f}%<br>Recebido (acum.): R$ %{customdata[0]:,.0f}<br>Esperado (acum.): R$ %{customdata[1]:,.0f}<extra></extra>'
          }},
          {{
            type:'scatter',
            x: d0.x, y: d0.rec,
            name:'% Recorrentes (share)',
            mode:'lines',
            fill:'tozeroy',
            line:{{color:'rgba(99, 179, 237, 1)', width:2}},
            fillcolor:'rgba(99, 179, 237, 0.14)',
            hovertemplate:'Cohort: ' + c0 + '<br>Mês após compra: %{x}<br>% Recorrentes: %{y:.2f}%<extra></extra>'
          }}
        ], {{
          title: 'Cohort detalhada — ' + c0,
          updatemenus: [{{ type:'dropdown', x: 1, y: 1.18, xanchor:'right', yanchor:'top', buttons, showactive:true }}],
          margin: {{l: 60, r: 20, t: 50, b: 40}},
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          legend: {{ orientation: 'h', y: 1.18 }},
          yaxis: {{ title:'%', gridcolor:'rgba(255,255,255,0.08)' }},
          xaxis: {{ title:'Mês após compra', dtick: 1 }},
          height: 520,
          autosize: true
        }}, PLOTLY_CONFIG);
      }};

      const loadMd = async () => {{
        try {{
          const r = await fetch('./respostas_empresa.md?ts=' + Date.now());
          const txt = await r.text();
          const esc = (s) => s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
          const html = esc(txt).split('\\n').map(line => {{
            if (line.startsWith('### ')) return `<h3>${{esc(line.slice(4))}}</h3>`;
            if (line.startsWith('## ')) return `<h2>${{esc(line.slice(3))}}</h2>`;
            if (line.startsWith('# ')) return `<h1>${{esc(line.slice(2))}}</h1>`;
            if (line.startsWith('- ')) return `<li>${{esc(line.slice(2))}}</li>`;
            if (line.trim() === '') return '<br/>';
            return `<p>${{esc(line)}}</p>`;
          }}).join('');
          document.getElementById('mdRespostas').innerHTML = `<div style=\"line-height:1.35\">${{html}}</div>`;
        }} catch (e) {{
          document.getElementById('mdRespostas').textContent = 'Não foi possível carregar o arquivo respostas_empresa.md no GitHub Pages.';
        }}
      }};

      const resizeAll = () => {{
        if (!window.Plotly || !Plotly.Plots || !Plotly.Plots.resize) return;
        ['plotProd','plotReg','plotForma','plotCash','plotCohSummary','plotCohDetail'].forEach(id => {{
          const el = document.getElementById(id);
          if (el && el.data) Plotly.Plots.resize(el);
        }});
      }};
      window.addEventListener('resize', () => setTimeout(resizeAll, 60));

      const renderAll = () => {{
        setKpis();
        renderProd();
        renderReg();
        renderForma();
        renderCash();
        renderCohort();
        setTimeout(resizeAll, 60);
      }};

      const bindTabs = () => {{
        const btns = Array.from(document.querySelectorAll('.navbtn'));
        btns.forEach(b => b.addEventListener('click', () => {{
          btns.forEach(x => x.classList.remove('active'));
          b.classList.add('active');
          const tabId = b.getAttribute('data-tab');
          document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
          document.getElementById(tabId).classList.add('active');
          const ttl = b.getAttribute('title') || '';
          const top = document.getElementById('topTitle');
          if (top && ttl) top.textContent = ttl.toUpperCase();
          if (tabId === 'tab-respostas') loadMd();
          setTimeout(resizeAll, 60);
        }}));
      }};

      buildFilters();
      bindTabs();
      loadMd();
      renderAll();
    </script>
  </body>
</html>
"""
    tpl = tpl.replace("{{", "{").replace("}}", "}")
    return (
        tpl.replace("__DATA_ALLOC_DUE__", payload_alloc_due)
        .replace("__DATA_ALLOC_PAY__", payload_alloc_pay)
        .replace("__DATA_COHORT__", payload_cohort)
        .replace("__DATA_SALES_CUBE__", payload_sales_cube)
        .replace("__DATA_SALES_COHORT__", payload_sales_cohort)
        .replace("__DATA_REC_MATURED__", payload_rec_matured)
        .replace("__DATA_REC_MATURED_FACT__", payload_rec_matured_fact)
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sales, allocated = _load()
    alloc_due_dim = _prep_alloc_dim(sales, allocated)
    alloc_pay_dim = _prep_pay_dim(sales, allocated)
    sales_cube = _prep_sales_cube(sales)
    sales_cohort = _prep_sales_cohort_dim(sales)
    rec_matured_dim = _prep_recurring_matured_dim(sales, allocated)
    rec_matured_fact = _prep_recurring_matured_fact(sales, allocated)
    cohort_dim = _prep_cohort_dim(sales, allocated)

    data_alloc_due = alloc_due_dim.rename(
        columns={"product_name": "product_name", "region": "region", "payment_method": "payment_method"}
    ).to_dict(orient="records")
    data_alloc_pay = alloc_pay_dim.rename(
        columns={"product_name": "product_name", "region": "region", "payment_method": "payment_method"}
    ).to_dict(orient="records")
    data_cohort = cohort_dim.rename(
        columns={"product_name": "product_name", "region": "region", "payment_method": "payment_method"}
    ).to_dict(orient="records")
    data_sales_cube = sales_cube.to_dict(orient="records")
    data_sales_cohort = sales_cohort.to_dict(orient="records")
    data_rec_matured = rec_matured_dim.to_dict(orient="records")
    data_rec_matured_fact = rec_matured_fact.to_dict(orient="records")

    (REPORTS_DIR / "dashboard.html").write_text(
        _html(
            data_alloc_due,
            data_alloc_pay,
            data_cohort,
            data_sales_cube,
            data_sales_cohort,
            data_rec_matured,
            data_rec_matured_fact,
        ),
        encoding="utf-8",
    )
    print(str(REPORTS_DIR / "dashboard.html"))


if __name__ == "__main__":
    main()

