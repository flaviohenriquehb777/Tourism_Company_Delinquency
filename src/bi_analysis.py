from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from .bi_metrics import TAXA_JUROS_MENSAL, calcular_metricas


@dataclass(frozen=True)
class Period:
    key: str
    label: str
    start: date
    end: date


def _to_ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d)


def _filter_period(fp: pd.DataFrame, fv: pd.DataFrame, p: Period) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = _to_ts(p.start)
    end = _to_ts(p.end) + pd.Timedelta(days=1)
    fp2 = fp[(fp["Data_Compra"] >= start) & (fp["Data_Compra"] < end)]
    fv2 = fv[(fv["Data_Compra"] >= start) & (fv["Data_Compra"] < end)]
    return fp2, fv2


def _as_date(x: str) -> date:
    return pd.Timestamp(x).date()


def metricas_por_periodo(fp: pd.DataFrame, fv: pd.DataFrame) -> dict[str, Any]:
    fp = fp.copy()
    fv = fv.copy()
    fp["Data_Compra"] = pd.to_datetime(fp["Data_Compra"], errors="coerce")
    fv["Data_Compra"] = pd.to_datetime(fv["Data_Compra"], errors="coerce")

    periods = [
        Period("baseline", "Baseline (2020)", _as_date("2020-01-01"), _as_date("2020-12-31")),
        Period("evento1", "Ticket Alto (Jan-Jun/21)", _as_date("2021-01-01"), _as_date("2021-06-30")),
        Period("evento2", "Recorrente+WhatsApp (Jul-Dez/21)", _as_date("2021-07-01"), _as_date("2021-12-31")),
        Period("pos_whatsapp", "Pós-WhatsApp (Jan-Jun/22)", _as_date("2022-01-01"), _as_date("2022-06-30")),
        Period("normal", "Período Normal (2022-Jun/24)", _as_date("2022-01-01"), _as_date("2024-06-30")),
        Period("pre_premium", "Pré-Premium (Jan-Jun/24)", _as_date("2024-01-01"), _as_date("2024-06-30")),
        Period("premium", "Premium (Jul-Set/24)", _as_date("2024-07-01"), _as_date("2024-09-30")),
        Period("pos_premium", "Pós-Premium (Out-Dez/24)", _as_date("2024-10-01"), _as_date("2024-12-31")),
    ]

    out: dict[str, Any] = {
        "taxa_juros_mensal": TAXA_JUROS_MENSAL,
        "total_linhas": int(len(fp)),
        "total_vendas": int(fv["ID_Compra"].nunique()),
        "receita_total": float(fp["Valor_Parcela"].sum()),
    }

    for p in periods:
        fpp, fvv = _filter_period(fp, fv, p)
        out[p.key] = calcular_metricas(fpp, fvv).to_dict()

    fp_rec = fp[fp["Tipo_Pagamento"] == "Recorrente"]
    fv_rec = fv[fv["Tipo_Pagamento"] == "Recorrente"]
    out["recorrente_total"] = calcular_metricas(fp_rec, fv_rec).to_dict()

    fp_trad = fp[fp["Tipo_Pagamento"] == "Tradicional"]
    fv_trad = fv[fv["Tipo_Pagamento"] == "Tradicional"]
    out["tradicional_total"] = calcular_metricas(fp_trad, fv_trad).to_dict()

    return out

