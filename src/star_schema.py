from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROC_DIR


@dataclass(frozen=True)
class StarSchemaPaths:
    dim_tempo: Path
    dim_cliente: Path
    dim_produto: Path
    dim_forma: Path
    dim_regiao: Path
    fato_pagamentos: Path
    fato_vendas: Path


def _marco_estrategico(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "Desconhecido"
    if dt < pd.Timestamp("2021-01-01"):
        return "Baseline (2020)"
    if dt < pd.Timestamp("2021-07-01"):
        return "Ticket Alto (Jan-Jun/21)"
    if dt < pd.Timestamp("2022-01-01"):
        return "Recorrente+WhatsApp (Jul-Dez/21)"
    if dt < pd.Timestamp("2024-07-01"):
        return "Período Normal (2022-Jun/24)"
    if dt < pd.Timestamp("2024-10-01"):
        return "Premium (Jul-Set/24)"
    return "Pós-Premium (Out/24+)"


def build_star_schema(xlsx_path: Path, out_dir: Path | None = None) -> StarSchemaPaths:
    out_dir = out_dir or PROC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(str(xlsx_path), sheet_name="Planilha1", usecols=list(range(15)))
    df.columns = [
        "Nome_Cliente",
        "Email_Cliente",
        "ID_Compra",
        "Data_Compra",
        "Produto",
        "Codigo_Produto",
        "Total_Parcelas_Recorrentes",
        "Valor",
        "Conversao",
        "Moeda_Compra",
        "Qtd_Parcelas",
        "Forma_Pagamento",
        "Num_Parcela",
        "Regiao",
        "Genero",
    ]

    df["ID_Compra"] = df["ID_Compra"].astype(str)
    df["Data_Compra"] = pd.to_datetime(df["Data_Compra"], errors="coerce")
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
    df["Conversao"] = pd.to_numeric(df["Conversao"], errors="coerce").fillna(1.0)
    df["Qtd_Parcelas"] = pd.to_numeric(df["Qtd_Parcelas"], errors="coerce").fillna(1).astype(int)
    df["Num_Parcela"] = pd.to_numeric(df["Num_Parcela"], errors="coerce").fillna(0).astype(int)
    df["Total_Parcelas_Recorrentes"] = (
        pd.to_numeric(df["Total_Parcelas_Recorrentes"], errors="coerce").fillna(0).astype(int)
    )

    df = df.dropna(subset=["Valor", "Data_Compra"]).reset_index(drop=True)

    df["Valor_BRL"] = df["Valor"] * df["Conversao"]
    df["Valor_Parcela"] = df["Valor_BRL"] / df["Qtd_Parcelas"].replace(0, 1)

    df["Tipo_Pagamento"] = np.where(
        df["Total_Parcelas_Recorrentes"] > 0,
        "Recorrente",
        np.where(df["Qtd_Parcelas"] == 1, "À Vista", "Tradicional"),
    )
    df["Flag_Inadimplente"] = np.where(
        (df["Total_Parcelas_Recorrentes"] > 0) & (df["Num_Parcela"] == 0),
        1,
        0,
    ).astype(int)

    df["Ano"] = df["Data_Compra"].dt.year
    df["Mes"] = df["Data_Compra"].dt.month
    df["AnoMes"] = df["Data_Compra"].dt.to_period("M").astype(str)
    df["Marco_Estrategico"] = df["Data_Compra"].apply(_marco_estrategico)

    dim_tempo = pd.DataFrame({"Data": pd.date_range("2020-01-01", "2025-12-31", freq="D")})
    dim_tempo["Ano"] = dim_tempo["Data"].dt.year
    dim_tempo["Mes"] = dim_tempo["Data"].dt.month
    dim_tempo["Dia"] = dim_tempo["Data"].dt.day
    dim_tempo["Trimestre"] = dim_tempo["Data"].dt.quarter.map(lambda q: f"T{q}")
    dim_tempo["Semestre"] = dim_tempo["Mes"].apply(lambda m: "S1" if m <= 6 else "S2")
    dim_tempo["Nome_Mes"] = dim_tempo["Data"].dt.strftime("%b")
    dim_tempo["AnoMes"] = dim_tempo["Data"].dt.to_period("M").astype(str)
    dim_tempo["Marco_Estrategico"] = dim_tempo["Data"].apply(_marco_estrategico)

    dim_cliente = (
        df[["Nome_Cliente", "Email_Cliente", "Genero", "Regiao"]]
        .drop_duplicates(subset=["Email_Cliente"])
        .reset_index(drop=True)
    )
    dim_cliente["ID_Cliente"] = dim_cliente.index + 1

    ticket_medio_por_produto = df.groupby("Codigo_Produto")["Valor_BRL"].mean().reset_index()
    ticket_medio_por_produto.columns = ["Codigo_Produto", "Ticket_Medio_Produto"]
    ticket_geral = float(ticket_medio_por_produto["Ticket_Medio_Produto"].median())

    def faixa_ticket(val: float) -> str:
        if val < ticket_geral * 0.7:
            return "Baixo"
        if val < ticket_geral * 1.3:
            return "Médio"
        if val < ticket_geral * 2.0:
            return "Alto"
        return "Premium"

    ticket_medio_por_produto["Faixa_Ticket"] = ticket_medio_por_produto["Ticket_Medio_Produto"].apply(
        faixa_ticket
    )

    dim_produto = (
        df[["Codigo_Produto", "Produto"]]
        .drop_duplicates()
        .merge(ticket_medio_por_produto, on="Codigo_Produto", how="left")
    )
    dim_produto["Era_Premium"] = dim_produto["Codigo_Produto"].isin(["PR8"])

    dim_forma = pd.DataFrame(
        {
            "Forma_Pagamento": ["Cartão de Crédito", "Boleto", "PIX"],
            "Risco_Inadimplencia": ["Médio", "Médio", "Baixo"],
            "Categoria": ["Parcelável", "À Vista/Parcelável", "À Vista"],
        }
    )

    mapa_macro = {
        "São Paulo": "Sudeste",
        "Rio de Janeiro": "Sudeste",
        "Minas Gerais": "Sudeste",
        "Espírito Santo": "Sudeste",
        "Paraná": "Sul",
        "Santa Catarina": "Sul",
        "Rio Grande do Sul": "Sul",
        "Bahia": "Nordeste",
        "Pernambuco": "Nordeste",
        "Ceará": "Nordeste",
        "Maranhão": "Nordeste",
        "Paraíba": "Nordeste",
        "Rio Grande do Norte": "Nordeste",
        "Alagoas": "Nordeste",
        "Sergipe": "Nordeste",
        "Piauí": "Nordeste",
        "Distrito Federal/Goiás": "Centro-Oeste",
        "Mato Grosso": "Centro-Oeste",
        "Mato Grosso do Sul": "Centro-Oeste",
        "Pará": "Norte",
        "Amazonas": "Norte",
        "Rondônia": "Norte",
        "Estrangeiro": "Internacional",
    }
    dim_regiao = pd.DataFrame({"Regiao": pd.Series(df["Regiao"].dropna().unique(), dtype="object")})
    dim_regiao["Macro_Regiao"] = dim_regiao["Regiao"].map(mapa_macro).fillna("Outros")

    df_fk = df.merge(dim_cliente[["Email_Cliente", "ID_Cliente"]], on="Email_Cliente", how="left")

    fato_pagamentos = df_fk[
        [
            "ID_Compra",
            "Data_Compra",
            "AnoMes",
            "Ano",
            "Mes",
            "Codigo_Produto",
            "Forma_Pagamento",
            "Regiao",
            "ID_Cliente",
            "Num_Parcela",
            "Qtd_Parcelas",
            "Total_Parcelas_Recorrentes",
            "Valor_BRL",
            "Valor_Parcela",
            "Moeda_Compra",
            "Tipo_Pagamento",
            "Flag_Inadimplente",
            "Marco_Estrategico",
        ]
    ].copy()

    fato_vendas = (
        fato_pagamentos.groupby("ID_Compra", dropna=False)
        .agg(
            Data_Compra=("Data_Compra", "first"),
            AnoMes=("AnoMes", "first"),
            Codigo_Produto=("Codigo_Produto", "first"),
            Forma_Pagamento=("Forma_Pagamento", "first"),
            Regiao=("Regiao", "first"),
            ID_Cliente=("ID_Cliente", "first"),
            Tipo_Pagamento=("Tipo_Pagamento", "first"),
            Marco_Estrategico=("Marco_Estrategico", "first"),
            Valor_Total_BRL=("Valor_BRL", "first"),
            Qtd_Parcelas=("Qtd_Parcelas", "first"),
            Total_Parcelas_Recorrentes=("Total_Parcelas_Recorrentes", "first"),
            Qtd_Parcelas_Inad=("Flag_Inadimplente", "sum"),
        )
        .reset_index()
    )
    fato_vendas["Flag_Venda_Inad"] = (fato_vendas["Qtd_Parcelas_Inad"] > 0).astype(int)
    fato_vendas["Ano"] = pd.to_datetime(fato_vendas["Data_Compra"]).dt.year
    fato_vendas["Mes"] = pd.to_datetime(fato_vendas["Data_Compra"]).dt.month

    p_dim_tempo = out_dir / "dim_tempo.csv"
    p_dim_cliente = out_dir / "dim_cliente.csv"
    p_dim_produto = out_dir / "dim_produto.csv"
    p_dim_forma = out_dir / "dim_forma.csv"
    p_dim_regiao = out_dir / "dim_regiao.csv"
    p_fp = out_dir / "fato_pagamentos.csv"
    p_fv = out_dir / "fato_vendas.csv"

    dim_tempo.to_csv(p_dim_tempo, index=False)
    dim_cliente.to_csv(p_dim_cliente, index=False)
    dim_produto.to_csv(p_dim_produto, index=False)
    dim_forma.to_csv(p_dim_forma, index=False)
    dim_regiao.to_csv(p_dim_regiao, index=False)
    fato_pagamentos.to_csv(p_fp, index=False)
    fato_vendas.to_csv(p_fv, index=False)

    return StarSchemaPaths(
        dim_tempo=p_dim_tempo,
        dim_cliente=p_dim_cliente,
        dim_produto=p_dim_produto,
        dim_forma=p_dim_forma,
        dim_regiao=p_dim_regiao,
        fato_pagamentos=p_fp,
        fato_vendas=p_fv,
    )

