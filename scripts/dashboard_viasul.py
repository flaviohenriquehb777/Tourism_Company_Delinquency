from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import INTEREST_RATE_MONTHLY, PROC_DIR, REPORTS_DIR
from src.metrics import (
    expected_vs_received,
    interest_coverage,
    interest_implied,
    matured_default_rate_over_time,
    monthly_sales_kpis,
)
from src.viz import (
    add_event_markers,
    cohort_area,
    expected_received_area,
    interest_coverage_indicators,
    line_two_axes,
    ts_line,
)


st.set_page_config(
    page_title="Tourism Company Delinquency — Dashboard",
    page_icon=str(REPORTS_DIR / "favicon.ico"),
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load_sales() -> pd.DataFrame:
    p = PROC_DIR / "sales_enriched.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p, parse_dates=["purchase_date"])


@st.cache_data(show_spinner=False)
def _load_allocated() -> pd.DataFrame:
    p = PROC_DIR / "allocated.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p, parse_dates=["due_date", "payment_date_eff"])


def _events() -> dict:
    try:
        import json

        p = PROC_DIR / "events.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _apply_filters(sales: pd.DataFrame, allocated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.header("Filtros")

    min_d = pd.Timestamp(sales["purchase_date"].min()).date()
    max_d = pd.Timestamp(sales["purchase_date"].max()).date()
    d0, d1 = st.sidebar.date_input("Período de compra", (min_d, max_d), min_value=min_d, max_value=max_d)
    d0 = pd.Timestamp(d0)
    d1 = pd.Timestamp(d1) + pd.Timedelta(days=1)

    tipos = ["Recorrente", "Tradicional"]
    sel_tipos = st.sidebar.multiselect("Tipo", tipos, default=tipos)

    sales2 = sales[(sales["purchase_date"] >= d0) & (sales["purchase_date"] < d1)].copy()
    if "recurring" in sales2.columns:
        if "Recorrente" not in sel_tipos:
            sales2 = sales2[~sales2["recurring"]]
        if "Tradicional" not in sel_tipos:
            sales2 = sales2[sales2["recurring"]]

    if "premium" in sales2.columns:
        prem_sel = st.sidebar.selectbox("Premium", ["Todos", "Somente premium", "Sem premium"], index=0)
        if prem_sel == "Somente premium":
            sales2 = sales2[sales2["premium"] == True]
        elif prem_sel == "Sem premium":
            sales2 = sales2[sales2["premium"] == False]

    for col, label in [
        ("region", "Região"),
        ("gender", "Gênero"),
        ("payment_method", "Forma de pagamento"),
        ("product_code", "Código do produto"),
        ("product_name", "Produto"),
    ]:
        if col in sales2.columns and sales2[col].notna().any():
            vals = sorted(sales2[col].dropna().astype(str).unique().tolist())
            chosen = st.sidebar.multiselect(label, vals, default=vals)
            sales2 = sales2[sales2[col].astype(str).isin(chosen)]

    ids = set(sales2["contract_id"].astype(str))
    allocated2 = allocated[allocated["contract_id"].astype(str).isin(ids)].copy()
    return sales2, allocated2


def _kpis(sales: pd.DataFrame) -> pd.DataFrame:
    return monthly_sales_kpis(sales)


def _evr(allocated: pd.DataFrame) -> pd.DataFrame:
    return expected_vs_received(allocated)


def _defaults(allocated: pd.DataFrame) -> pd.DataFrame:
    return matured_default_rate_over_time(allocated)


def _cohort(allocated: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    from src.metrics import cohort_curves

    return cohort_curves(allocated, sales)


def _metric_cards(evr: pd.DataFrame, defaults_t: pd.DataFrame) -> None:
    last = evr.dropna().tail(1)
    last_def = defaults_t.dropna().tail(1)
    col1, col2, col3 = st.columns(3)
    if len(last):
        exp = float(last["expected"].iloc[0])
        rec = float(last["received"].iloc[0])
        cov = rec / exp if exp else 0.0
        col1.metric("Receita esperada (mês)", f"R$ {exp:,.0f}".replace(",", "."))
        col2.metric("Receita recebida (mês)", f"R$ {rec:,.0f}".replace(",", "."))
        col3.metric("Cobertura (recebido/esperado)", f"{cov*100:.1f}%".replace(".", ","))
    if len(last_def):
        st.caption(
            f"Inadimplência (mês): {(float(last_def['default_rate'].iloc[0])*100):.1f}% | "
            f"Recorrente: {(float(last_def['recurring_default_rate'].iloc[0])*100):.1f}%"
        )


def main() -> None:
    st.title("Tourism Company Delinquency — Dashboard")
    st.caption("Taxa de juros de referência: 2,49% a.m.")

    sales = _load_sales()
    allocated = _load_allocated()
    events = _events()
    sales_f, allocated_f = _apply_filters(sales, allocated)

    kpis = _kpis(sales_f)
    evr = _evr(allocated_f)
    defaults_t = _defaults(allocated_f)
    si = interest_implied(sales_f)
    snap = defaults_t["month"].max() if "month" in defaults_t.columns and defaults_t["month"].notna().any() else pd.Timestamp.today()
    cov = interest_coverage(allocated_f, si, pd.Timestamp(snap))

    _metric_cards(evr, defaults_t)

    tabs = st.tabs(
        [
            "Visão Geral",
            "Ticket Alto",
            "Recorrente",
            "WhatsApp",
            "Premium",
            "Juros",
            "Geo & Perfil",
            "Cohort",
            "Relatório",
        ]
    )

    with tabs[0]:
        colA, colB = st.columns(2)
        with colA:
            fig_ticket = add_event_markers(
                line_two_axes(kpis, "month", "avg_ticket", "contracts", "Ticket médio", "Contratos", "Ticket médio e volume"),
                events,
            )
            st.plotly_chart(fig_ticket, use_container_width=True)
        with colB:
            fig_evr = add_event_markers(expected_received_area(evr, "Receita esperada x recebida (área)"), events)
            st.plotly_chart(fig_evr, use_container_width=True)

        colC, colD = st.columns(2)
        with colC:
            from src.viz import default_rate_plot

            fig_def = add_event_markers(default_rate_plot(defaults_t, "Inadimplência por maturidade"), events)
            st.plotly_chart(fig_def, use_container_width=True)
        with colD:
            fig_interest = interest_coverage_indicators(cov, "Cobertura de juros x perdas (contratos maduros)")
            st.plotly_chart(fig_interest, use_container_width=True)

    with tabs[1]:
        st.subheader("Evento 1 — Ticket médio mais alto (Jan/2021)")
        st.plotly_chart(add_event_markers(expected_received_area(evr, "Receita esperada x recebida (área)"), events), use_container_width=True)
        if "product_name" in sales_f.columns:
            tmp = sales_f.copy()
            tmp["month"] = tmp["purchase_date"].values.astype("datetime64[M]")
            prod = (
                tmp.groupby(["month", "product_name"], as_index=False)
                .agg(receita=("product_price", "sum"))
                .sort_values("month")
            )
            fig = px.area(prod, x="month", y="receita", color="product_name", title="Mix de produtos (receita)")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Evento 2 — Expansão do recorrente (Jul/2021)")
        k = kpis.copy()
        if "recurring_share" in k.columns:
            k["recurring_share_pct"] = k["recurring_share"] * 100
            fig = add_event_markers(ts_line(k, "month", "recurring_share_pct", "Participação de Vendas Recorrentes (%)", "Recorrente (%)"), events)
            st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(add_event_markers(expected_received_area(evr, "Receita esperada x recebida (área)"), events), use_container_width=True)

    with tabs[3]:
        st.subheader("WhatsApp de cobrança (Jul/2021)")
        st.plotly_chart(add_event_markers(expected_received_area(evr, "Receita esperada x recebida (área)"), events), use_container_width=True)

    with tabs[4]:
        st.subheader("Oferta premium (Jul/2024)")
        if "premium" in sales_f.columns:
            tmp = sales_f.copy()
            tmp["month"] = tmp["purchase_date"].values.astype("datetime64[M]")
            prem = tmp.groupby(["month", "premium"], as_index=False).agg(vendas=("contract_id", "nunique"))
            fig = px.bar(prem, x="month", y="vendas", color="premium", barmode="group", title="Vendas (premium vs não premium)")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.subheader("Política de juros (2,49% a.m.)")
        st.plotly_chart(interest_coverage_indicators(cov, "Cobertura de juros x perdas (contratos maduros)"), use_container_width=True)
        st.caption(f"Cobertura total estimada: {cov.get('coverage_ratio') if cov.get('coverage_ratio') is not None else 'N/D'}x")

    with tabs[6]:
        st.subheader("Geo & Perfil")
        cols = st.columns(2)
        with cols[0]:
            if "region" in sales_f.columns and sales_f["region"].notna().any():
                r = sales_f.groupby("region", as_index=False).agg(vendas=("contract_id", "nunique")).sort_values("vendas", ascending=False).head(12)
                fig = px.bar(r, x="vendas", y="region", orientation="h", title="Top regiões por vendas")
                st.plotly_chart(fig, use_container_width=True)
        with cols[1]:
            if "gender" in sales_f.columns and sales_f["gender"].notna().any():
                g = sales_f.groupby("gender", as_index=False).agg(vendas=("contract_id", "nunique")).sort_values("vendas", ascending=False)
                fig = px.pie(g, names="gender", values="vendas", title="Distribuição por gênero (vendas)")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[7]:
        st.subheader("Cohort — % inadimplência e % recorrente")
        cohort_df = _cohort(allocated_f, sales_f)
        fig = cohort_area(cohort_df, "Cohort — Inadimplência vs Recorrente")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[8]:
        p = REPORTS_DIR / "respostas_empresa.md"
        if p.exists():
            st.markdown(p.read_text(encoding="utf-8"))
        else:
            st.info("Relatório não encontrado. Gere com: python scripts/build_deliverables.py")


if __name__ == "__main__":
    main()

