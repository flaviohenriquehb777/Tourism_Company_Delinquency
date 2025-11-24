import pandas as pd
from dateutil.relativedelta import relativedelta

def monthly_sales_kpis(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["month"] = s["purchase_date"].values.astype("datetime64[M]")
    s["ticket"] = s["product_price"]
    s["contracts"] = 1
    out = s.groupby("month").agg(
        contracts=("contracts","sum"),
        avg_ticket=("ticket","mean"),
        recurring_share=("recurring","mean"),
        total_sold=("product_price","sum"),
    ).reset_index()
    return out

def expected_vs_received(allocated: pd.DataFrame) -> pd.DataFrame:
    a = allocated.copy()
    a["due_month"] = a["due_date"].values.astype("datetime64[M]")
    a["pay_month"] = pd.to_datetime(a["payment_date_eff"]).values.astype("datetime64[M]")
    exp = a.groupby("due_month")["expected_amount"].sum().reset_index().rename(columns={"due_month":"month","expected_amount":"expected"})
    rec = a.groupby("pay_month")["paid_amount"].sum().reset_index().rename(columns={"pay_month":"month","paid_amount":"received"})
    out = pd.merge(exp, rec, on="month", how="outer").fillna(0).sort_values("month")
    return out

def matured_default_rate_over_time(allocated: pd.DataFrame) -> pd.DataFrame:
    timeline = pd.date_range(allocated["due_date"].min(), allocated["due_date"].max() + relativedelta(months=+1), freq="MS")
    rows = []
    for snap in timeline:
        snap_agg = matured_snapshot_overall(allocated, snap)
        rows.append({"month": snap, **snap_agg})
    return pd.DataFrame(rows)

def matured_snapshot_overall(allocated: pd.DataFrame, snap_date: pd.Timestamp) -> dict:
    a = allocated.copy()
    if "contract_id" not in a.columns:
        cols = ", ".join(map(str, a.columns.tolist()))
        raise ValueError(
            f"allocated precisa conter a coluna 'contract_id' para cálculo por contrato. Colunas encontradas: {cols}"
        )
    grp = a.groupby("contract_id").agg(
        total_expected=("expected_amount","sum"),
        total_paid=("paid_amount","sum"),
        last_due=("due_date","max"),
        recurring=("recurring","max"),
    ).reset_index()
    grp = grp[grp["recurring"] == True]
    grp["matured"] = grp["last_due"] < snap_date
    matured = grp[grp["matured"]]
    if matured.empty:
        return {"contracts_matured":0,"defaults":0,"default_rate":0.0,"losses":0.0,"recurring_default_rate":0.0,"traditional_default_rate":None}
    matured["defaulted"] = matured["total_paid"] + 1e-6 < matured["total_expected"]
    losses = (matured["total_expected"] - matured["total_paid"]).clip(lower=0).sum()
    rd = matured[matured["recurring"]==True]
    td = matured[matured["recurring"]!=True]
    return {
        "contracts_matured": int(len(matured)),
        "defaults": int(matured["defaulted"].sum()),
        "default_rate": float(matured["defaulted"].mean()),
        "losses": float(losses),
        "recurring_default_rate": float(rd["defaulted"].mean()) if not rd.empty else 0.0,
        "traditional_default_rate": None,
    }

def segment_compare(df: pd.DataFrame, date_cut: pd.Timestamp, metric: str) -> dict:
    pre = df[df["month"] < date_cut][metric].mean()
    post = df[df["month"] >= date_cut][metric].mean()
    delta = None
    delta_abs = None
    if pd.notnull(pre) and pd.notnull(post):
        delta_abs = post - pre
        delta = (post - pre) / pre if pre != 0 else None
    return {
        "pre": float(pre) if pd.notnull(pre) else None,
        "post": float(post) if pd.notnull(post) else None,
        "delta_abs": float(delta_abs) if delta_abs is not None else None,
        "delta_pct": float(delta) if delta is not None else None,
    }

def premium_flag(sales: pd.DataFrame, premium_start: pd.Timestamp) -> pd.DataFrame:
    out = sales.copy()
    thr = out["product_price"].quantile(0.75)
    out["premium"] = (out["purchase_date"] >= premium_start) & (out["product_price"] >= thr)
    return out, thr

def interest_implied(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["principal"] = s["product_price"]
    s["total_due"] = s["installment_value"] * s["installments_total"]
    s.loc[s["total_due"].isna() & s["principal"].notna(), "total_due"] = s["principal"]
    s["surcharge"] = (s["total_due"] - s["principal"]).round(2).clip(lower=0)
    s["tenor"] = s["installments_total"]
    s["implied_monthly_rate"] = ((s["total_due"]/s["principal"]).pow(1/s["tenor"]) - 1).replace([pd.NA, pd.NaT], 0).clip(lower=0)
    from src.config import INTEREST_RATE_MONTHLY
    tenor = s["tenor"].fillna(1).replace(0, 1)
    s["policy_rate_monthly"] = INTEREST_RATE_MONTHLY
    s["policy_total_due"] = s["principal"] * (1 + INTEREST_RATE_MONTHLY) ** tenor
    s.loc[tenor <= 1, "policy_total_due"] = s["principal"]
    s["surcharge_policy"] = (s["policy_total_due"] - s["principal"]).round(2).clip(lower=0)
    return s

def interest_coverage(allocated: pd.DataFrame, sales_interest: pd.DataFrame, snap_date: pd.Timestamp) -> dict:
    agg = matured_snapshot_overall(allocated, snap_date)
    a = allocated.copy()
    last_due = a.groupby("contract_id")["due_date"].max().reset_index().rename(columns={"due_date":"last_due"})
    si = sales_interest.merge(last_due, on="contract_id", how="left")
    matured_ids = set(si[si["last_due"] < snap_date]["contract_id"].astype(str))
    si_m = si[si["contract_id"].astype(str).isin(matured_ids)]
    interest_income = float(si_m["surcharge_policy"].sum()) if "surcharge_policy" in si_m.columns else float(si_m["surcharge"].sum())
    losses = agg["losses"]
    no_interest_data = interest_income < 0.01
    cov = (interest_income / losses) if (not no_interest_data) and losses and losses > 0 else None
    split = {}
    for flag in [True, False]:
        ids = set(si_m[si_m["recurring"]==flag]["contract_id"].astype(str))
        a_sub = a[a["contract_id"].astype(str).isin(ids)]
        agg_sub = matured_snapshot_overall(a_sub, snap_date)
        if "surcharge_policy" in si_m.columns:
            inc_sub = float(si_m[si_m["recurring"]==flag]["surcharge_policy"].sum())
        else:
            inc_sub = float(si_m[si_m["recurring"]==flag]["surcharge"].sum())
        cov_sub = (inc_sub / agg_sub["losses"]) if (not no_interest_data) and agg_sub["losses"] and agg_sub["losses"] > 0 else None
        split["recurring" if flag else "traditional"] = cov_sub
    if no_interest_data:
        split = {"recurring": None, "traditional": None}
    return {"coverage_ratio": cov, "split": split, "interest_income": interest_income, "losses": float(losses), "no_interest_data": no_interest_data}

def cohort_curves(allocated: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    a = allocated.copy()
    s = sales.copy()
    s["purchase_month"] = s["purchase_date"].values.astype("datetime64[M]")
    a = a.merge(s[["contract_id", "purchase_month"]], on="contract_id", how="left")
    a = a[a["purchase_month"].notna()]
    a["due_month"] = a["due_date"].values.astype("datetime64[M]")
    a["age_months"] = (
        (a["due_month"].dt.year - a["purchase_month"].dt.year) * 12
        + (a["due_month"].dt.month - a["purchase_month"].dt.month)
    ).astype(int)
    a = a[a["age_months"] >= 0]
    a["expected_rec"] = a["expected_amount"].where(a["recurring"] == True, 0.0)

    g = (
        a.groupby(["purchase_month", "age_months"], as_index=False)
        .agg(
            expected=("expected_amount", "sum"),
            received=("paid_amount", "sum"),
            expected_rec=("expected_rec", "sum"),
        )
        .sort_values(["purchase_month", "age_months"])
    )

    g["cum_expected"] = g.groupby("purchase_month")["expected"].cumsum()
    g["cum_received"] = g.groupby("purchase_month")["received"].cumsum()
    g["cum_expected_rec"] = g.groupby("purchase_month")["expected_rec"].cumsum()

    denom = g["cum_expected"].replace(0, pd.NA)
    g["delinquency_pct"] = (1 - (g["cum_received"] / denom)).clip(lower=0).fillna(0.0)
    g["recurring_pct"] = (g["cum_expected_rec"] / denom).fillna(0.0)
    return g[["purchase_month", "age_months", "delinquency_pct", "recurring_pct", "cum_expected", "cum_received"]]
