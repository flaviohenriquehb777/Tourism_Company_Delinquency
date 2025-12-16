from pathlib import Path
from jinja2 import Template
import pandas as pd

TEMPLATE = """
# Tourism Company Delinquency – Relatório Analítico

## 1. Mudança de estratégia em janeiro de 2021
Esta seção avalia a mudança de portfólio (priorização de ticket médio mais alto) e o efeito sobre vendas e risco.

- Ticket médio (R$): {{ fmt_money(high_ticket.avg_ticket.pre) }} → {{ fmt_money(high_ticket.avg_ticket.post) }} (Δ {{ fmt_pct(high_ticket.avg_ticket.delta_pct) }})
- Inadimplência por maturidade (recorrente): {{ fmt_pct(high_ticket.default.pre) }} → {{ fmt_pct(high_ticket.default.post) }} (Δ {{ fmt_pp(high_ticket.default.delta_abs) }})
- Cobertura de recebimento (recebido/esperado): {{ fmt_pct(high_ticket.receipt.pre) }} → {{ fmt_pct(high_ticket.receipt.post) }} (Δ {{ fmt_pp(high_ticket.receipt.delta_abs) }})
- Leitura executiva: ticket cresceu, mas a inadimplência recorrente e a cobertura de recebimento indicam maior risco e pior conversão de caixa no pós-mudança.

## 2. Ampliação do parcelamento recorrente em julho de 2021
Esta seção mede se a ampliação do recorrente aumentou volume e se o risco subiu de forma desproporcional.

- Volume de contratos/mês: {{ fmt_int(recurring.contracts.pre) }} → {{ fmt_int(recurring.contracts.post) }} (Δ {{ fmt_pct(recurring.contracts.delta_pct) }})
- Inadimplência por maturidade (recorrente): {{ fmt_pct(recurring.default_rec.pre) }} → {{ fmt_pct(recurring.default_rec.post) }} (Δ {{ fmt_pp(recurring.default_rec.delta_abs) }})
- Cobertura de recebimento (recebido/esperado): {{ fmt_pct(recurring.receipt.pre) }} → {{ fmt_pct(recurring.receipt.post) }} (Δ {{ fmt_pp(recurring.receipt.delta_abs) }})
- Leitura executiva: houve forte expansão de volume; o indicador-chave é se a inadimplência e a cobertura de recebimento acompanharam de forma saudável.

## 3. WhatsApp Cobrança
Objetivo: avaliar se a plataforma melhorou a conversão de recebimento sem degradar vendas.

- Início (referência): {{ whatsapp_start or 'não detectado' }}
- Inadimplência por maturidade (recorrente): {{ fmt_pct(whatsapp.default.pre) }} → {{ fmt_pct(whatsapp.default.post) }}
- Observação: quando "antes" aparece como N/D, não há histórico suficiente anterior ao marco para comparação estatística.

## 4. Produtos premium em agosto de 2024
Aqui avaliamos se a oferta premium pressionou o fluxo de caixa e a inadimplência.

- Limiar de premium (P75 do preço, R$): {{ fmt_money(premium_threshold) }}
- Inadimplência por maturidade (recorrente): {{ fmt_pct(premium.default.pre) }} → {{ fmt_pct(premium.default.post) }} (Δ {{ fmt_pp(premium.default.delta_abs) }})
- Cobertura de recebimento (recebido/esperado): {{ fmt_pct(premium.receipt.pre) }} → {{ fmt_pct(premium.receipt.post) }} (Δ {{ fmt_pp(premium.receipt.delta_abs) }})
- Leitura executiva: se a inadimplência e/ou a cobertura piorarem, a estratégia premium tende a aumentar risco e pressionar caixa.

## 5. Política de juros
Análise de suficiência dos juros para cobrir perdas por inadimplência (recorrente).

- Cobertura (juros ÷ perdas): {{ interest.coverage_note }}
- Comparativo por modalidade: {{ interest.split_note }}
- Nota: quando a base não traz juros explícitos/embutidos, a cobertura tende a ficar subestimada.

## Resumo executivo
{{ executive_summary }}
"""

def render_report(context: dict, out_path: Path):
    def metric_blank():
        return {"pre": None, "post": None, "delta_abs": None, "delta_pct": None}

    def as_dict(x):
        return x if isinstance(x, dict) else {}

    c = as_dict(context).copy()

    high_ticket = as_dict(c.get("high_ticket")).copy()
    high_ticket.setdefault("avg_ticket", metric_blank())
    high_ticket.setdefault("default", metric_blank())
    high_ticket.setdefault("receipt", metric_blank())
    c["high_ticket"] = high_ticket

    recurring = as_dict(c.get("recurring")).copy()
    recurring.setdefault("contracts", metric_blank())
    recurring.setdefault("default_rec", metric_blank())
    recurring.setdefault("receipt", metric_blank())
    c["recurring"] = recurring

    whatsapp = as_dict(c.get("whatsapp")).copy()
    whatsapp.setdefault("default", metric_blank())
    c["whatsapp"] = whatsapp

    premium = as_dict(c.get("premium")).copy()
    premium.setdefault("default", metric_blank())
    premium.setdefault("receipt", metric_blank())
    c["premium"] = premium

    interest = as_dict(c.get("interest")).copy()
    interest.setdefault("coverage_note", "N/D")
    interest.setdefault("split_note", "N/D")
    c["interest"] = interest

    c.setdefault("whatsapp_start", None)
    c.setdefault("premium_threshold", None)
    c.setdefault("executive_summary", "")

    def fmt_money(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "N/D"
        return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def fmt_pct(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "N/D"
        return f"{float(x)*100:.1f}%"

    def fmt_pp(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "N/D"
        return f"{float(x)*100:+.1f} p.p."

    def fmt_int(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "N/D"
        return f"{float(x):,.0f}".replace(",", ".")

    t = Template(TEMPLATE)
    txt = t.render(**c, fmt_money=fmt_money, fmt_pct=fmt_pct, fmt_pp=fmt_pp, fmt_int=fmt_int)
    out_path.write_text(txt, encoding="utf-8")
    return out_path
