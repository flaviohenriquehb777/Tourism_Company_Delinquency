from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


TAXA_JUROS_MENSAL = 0.0249


@dataclass(frozen=True)
class MetricPack:
    receita_esperada: float
    receita_recebida: float
    receita_perdida: float
    pct_cobertura: float
    ticket_medio: float
    qtd_vendas: int
    qtd_clientes: int
    qtd_parc_inad: int
    qtd_vendas_inad: int
    taxa_inad_parcelas: float
    taxa_inad_financeira: float
    taxa_vendas_inad: float
    receita_juros: float
    saldo_juros_inad: float
    break_even_taxa: float
    taxa_adequada: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "receita_esperada": self.receita_esperada,
            "receita_recebida": self.receita_recebida,
            "receita_perdida": self.receita_perdida,
            "pct_cobertura": self.pct_cobertura,
            "ticket_medio": self.ticket_medio,
            "qtd_vendas": self.qtd_vendas,
            "qtd_clientes": self.qtd_clientes,
            "qtd_parc_inad": self.qtd_parc_inad,
            "qtd_vendas_inad": self.qtd_vendas_inad,
            "taxa_inad_parcelas": self.taxa_inad_parcelas,
            "taxa_inad_financeira": self.taxa_inad_financeira,
            "taxa_vendas_inad": self.taxa_vendas_inad,
            "receita_juros": self.receita_juros,
            "saldo_juros_inad": self.saldo_juros_inad,
            "break_even_taxa": self.break_even_taxa,
            "taxa_adequada": self.taxa_adequada,
        }


def calcular_metricas(fp: pd.DataFrame, fv: pd.DataFrame, taxa_juros_mensal: float = TAXA_JUROS_MENSAL) -> MetricPack:
    receita_esperada = float(fp["Valor_Parcela"].sum())
    receita_recebida = float(fp.loc[fp["Flag_Inadimplente"] == 0, "Valor_Parcela"].sum())
    receita_perdida = float(receita_esperada - receita_recebida)
    pct_cobertura = float(receita_recebida / receita_esperada) if receita_esperada else 0.0
    ticket_medio = float(fv["Valor_Total_BRL"].mean()) if len(fv) else 0.0

    qtd_vendas = int(fv["ID_Compra"].nunique())
    qtd_clientes = int(fv["ID_Cliente"].nunique())

    parc_recorrentes = fp[fp["Tipo_Pagamento"] == "Recorrente"]
    qtd_parc_inad = int(fp["Flag_Inadimplente"].sum())
    qtd_parc_rec = int(len(parc_recorrentes))
    taxa_inad_parcelas = float(qtd_parc_inad / qtd_parc_rec) if qtd_parc_rec else 0.0
    taxa_inad_financeira = float(receita_perdida / receita_esperada) if receita_esperada else 0.0
    qtd_vendas_inad = int(fv["Flag_Venda_Inad"].sum()) if "Flag_Venda_Inad" in fv.columns else 0
    taxa_vendas_inad = float(qtd_vendas_inad / qtd_vendas) if qtd_vendas else 0.0

    fp_rec_pagas = fp[(fp["Tipo_Pagamento"] == "Recorrente") & (fp["Flag_Inadimplente"] == 0)].copy()
    base_juros = float((fp_rec_pagas["Valor_Parcela"] * fp_rec_pagas["Num_Parcela"]).sum())
    receita_juros = float(base_juros * taxa_juros_mensal)
    saldo_juros_inad = float(receita_juros - receita_perdida)
    break_even_taxa = float(receita_perdida / base_juros) if base_juros else 0.0
    taxa_adequada = bool(taxa_juros_mensal >= break_even_taxa) if base_juros else False

    return MetricPack(
        receita_esperada=receita_esperada,
        receita_recebida=receita_recebida,
        receita_perdida=receita_perdida,
        pct_cobertura=pct_cobertura,
        ticket_medio=ticket_medio,
        qtd_vendas=qtd_vendas,
        qtd_clientes=qtd_clientes,
        qtd_parc_inad=qtd_parc_inad,
        qtd_vendas_inad=qtd_vendas_inad,
        taxa_inad_parcelas=taxa_inad_parcelas,
        taxa_inad_financeira=taxa_inad_financeira,
        taxa_vendas_inad=taxa_vendas_inad,
        receita_juros=receita_juros,
        saldo_juros_inad=saldo_juros_inad,
        break_even_taxa=break_even_taxa,
        taxa_adequada=taxa_adequada,
    )

