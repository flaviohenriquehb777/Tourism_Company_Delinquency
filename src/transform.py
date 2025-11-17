import numpy as np
import pandas as pd

def build_schedule(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["contract_id"] = s["contract_id"].astype(str)
    s["customer_id"] = s["customer_id"].astype(str)
    s["purchase_date"] = pd.to_datetime(s["purchase_date"], errors="coerce")
    s = s[s["purchase_date"].notna()].reset_index(drop=True)

    n = (
        pd.to_numeric(s.get("installments_total"), errors="coerce")
        .fillna(1)
        .astype(int)
        .clip(lower=1)
        .to_numpy()
    )
    total_rows = int(n.sum())
    if total_rows == 0:
        return pd.DataFrame(
            columns=[
                "contract_id",
                "customer_id",
                "due_date",
                "installment_no",
                "installments_total",
                "expected_amount",
                "recurring",
            ]
        )

    row_idx = np.repeat(np.arange(len(s), dtype=int), n)
    base = np.repeat(np.cumsum(n) - n, n)
    offset = (np.arange(total_rows, dtype=int) - base).astype(int)

    sch = s.loc[row_idx, ["contract_id", "customer_id", "purchase_date", "installments_total", "installment_value", "product_price", "recurring"]].copy()
    sch["installment_no"] = offset + 1

    purchase_day = sch["purchase_date"].dt.day.astype(int)
    due_period = sch["purchase_date"].dt.to_period("M") + offset
    due_month_start = due_period.dt.to_timestamp(how="start")
    due_month_end = (due_period + 1).dt.to_timestamp(how="start") - pd.Timedelta(days=1)
    due_day = np.minimum(purchase_day.to_numpy(), due_month_end.dt.day.to_numpy())
    sch["due_date"] = (due_month_start + pd.to_timedelta(due_day - 1, unit="D")).dt.normalize()

    iv = pd.to_numeric(sch.get("installment_value"), errors="coerce")
    total = pd.to_numeric(sch.get("product_price"), errors="coerce")
    inst_total = pd.to_numeric(sch.get("installments_total"), errors="coerce").fillna(1).astype(int)
    sch["expected_amount"] = iv.where(iv.notna(), total / inst_total.replace(0, 1))

    sch = sch[["contract_id", "customer_id", "due_date", "installment_no", "installments_total", "expected_amount", "recurring"]]
    return sch

def assign_payments(schedule: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    s = schedule.copy()
    s["contract_id"] = s["contract_id"].astype(str)
    s = s.sort_values(["contract_id", "due_date", "installment_no"])

    p = payments.copy()
    p["ref_id"] = p["ref_id"].astype(str)
    p["payment_date"] = pd.to_datetime(p["payment_date"], errors="coerce")
    p["payment_amount"] = pd.to_numeric(p["payment_amount"], errors="coerce")
    p = p.dropna(subset=["ref_id", "payment_date", "payment_amount"]).sort_values(["ref_id", "payment_date"])
    pay_agg = p.groupby("ref_id", dropna=False).agg(
        payment_date=("payment_date", list),
        payment_amount=("payment_amount", list),
    )
    def alloc(group):
        expected = pd.to_numeric(group["expected_amount"], errors="coerce").fillna(0).to_numpy(dtype=float)
        alloc_paid = np.zeros(len(group), dtype=float)
        pay_dates_eff = [pd.NaT] * len(group)
        idx = 0
        cid = str(group.name)
        if cid in pay_agg.index:
            dates = pay_agg.at[cid, "payment_date"]
            amts = pay_agg.at[cid, "payment_amount"]
            for d, a in zip(dates, amts):
                amt = float(a) if pd.notnull(a) else 0.0
                while amt > 0 and idx < len(expected):
                    remaining = expected[idx] - alloc_paid[idx]
                    if remaining <= 1e-9:
                        idx += 1
                        continue
                    take = remaining if remaining < amt else amt
                    alloc_paid[idx] += take
                    pay_dates_eff[idx] = d
                    amt -= take
                    if alloc_paid[idx] + 1e-6 >= expected[idx]:
                        idx += 1
        out = group.copy()
        out["contract_id"] = cid
        out["paid_amount"] = alloc_paid.tolist()
        out["payment_date_eff"] = pay_dates_eff
        return out
    allocated = s.groupby("contract_id", dropna=False, group_keys=False).apply(alloc)
    allocated["paid_amount"] = allocated["paid_amount"].fillna(0.0)
    return allocated

def maturity_snapshot(allocated: pd.DataFrame, snap_date: pd.Timestamp) -> pd.DataFrame:
    comp = allocated.copy()
    comp["matured"] = comp["due_date"] < snap_date - relativedelta(days=+1)
    agg = comp.groupby("contract_id").agg(
        installments_total=("installments_total","max"),
        total_expected=("expected_amount","sum"),
        total_paid=("paid_amount","sum"),
        last_due=("due_date","max"),
    ).reset_index()
    agg["matured"] = agg["last_due"] < snap_date
    agg["defaulted"] = (agg["matured"]) & (agg["total_paid"] + 1e-6 < agg["total_expected"])
    return agg
