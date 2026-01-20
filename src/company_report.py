from __future__ import annotations

from pathlib import Path

from jinja2 import Template


TEMPLATE = """# Respostas às Perguntas do Negócio

Este documento resume as respostas para as perguntas levantadas pela empresa, embasadas nos dados e com referência direta aos gráficos do dashboard.

## 1) Priorização de Ticket Médio Mais Alto (a partir de Jan/2021)

**Resposta**
- O ticket médio aumentou de forma relevante no pós-evento, mas houve piora em inadimplência e redução na cobertura de recebimento, indicando maior risco/custo para o caixa.

**Evidências (pré vs pós)**
- Ticket médio: {{ fmt_money(high_ticket.avg_ticket.pre) }} → {{ fmt_money(high_ticket.avg_ticket.post) }} (Δ {{ fmt_pct(high_ticket.avg_ticket.delta_pct) }})
- Inadimplência (maturidade): {{ fmt_pct(high_ticket.default.pre) }} → {{ fmt_pct(high_ticket.default.post) }} (Δ {{ fmt_pp(high_ticket.default.delta_abs) }})
- Cobertura de recebimento: {{ fmt_pct(high_ticket.receipt.pre) }} → {{ fmt_pct(high_ticket.receipt.post) }} (Δ {{ fmt_pp(high_ticket.receipt.delta_abs) }})

**Onde olhar no dashboard**
- Aba **Informações Gerais**: *% Inadimplência por Produto* e *% Inadimplência por Região*.
- Aba **Fluxo de Caixa**: *Receita Esperada vs Receita Real (área)* (gap/hover de inadimplência).

## 2) Expansão do Parcelamento Recorrente (a partir de Jul/2021)

**Resposta**
- A expansão do recorrente aumentou o volume de contratos e a participação do recorrente no mix. Em paralelo, a inadimplência do recorrente aumentou, o que exige atenção ao desenho de cobrança/limites.

**Evidências (pré vs pós)**
- Contratos: {{ fmt_int(recurring.contracts.pre) }} → {{ fmt_int(recurring.contracts.post) }} (Δ {{ fmt_pct(recurring.contracts.delta_pct) }})
- Inadimplência do recorrente: {{ fmt_pct(recurring.default_rec.pre) }} → {{ fmt_pct(recurring.default_rec.post) }} (Δ {{ fmt_pp(recurring.default_rec.delta_abs) }})
- Cobertura de recebimento: {{ fmt_pct(recurring.receipt.pre) }} → {{ fmt_pct(recurring.receipt.post) }} (Δ {{ fmt_pp(recurring.receipt.delta_abs) }})

**Onde olhar no dashboard**
- Aba **Informações Gerais**: *% Inadimplência e Share Recorrente por Forma de Pagamento*.
- Aba **Análise Cohort**: evolução de *% Recorrentes (share)* por cohort.

## 3) WhatsApp de Cobrança (a partir de Jul/2021)

**Resposta**
- O WhatsApp não evidenciou uma melhora clara e sustentada de cobertura sem custo; a inadimplência e a cobertura precisam ser analisadas junto à expansão do recorrente no mesmo período.

**Evidências (pré vs pós)**
- Inadimplência (maturidade): {{ fmt_pct(whatsapp.default.pre) }} → {{ fmt_pct(whatsapp.default.post) }} (Δ {{ fmt_pp(whatsapp.default.delta_abs) }})

**Onde olhar no dashboard**
- Aba **Fluxo de Caixa**: *Receita Esperada vs Receita Real (área)* (avaliar cobertura no período).
- Aba **Análise Cohort**: *% Inadimplência* por cohort (mudança estrutural vs sazonalidade).

## 4) Oferta Premium (a partir de Jul/2024)

**Resposta**
- A entrada de produtos premium deve ser acompanhada com foco em caixa e inadimplência por maturidade. O indicador principal é o gap entre esperado e recebido e a trajetória da inadimplência após o marco.

**Evidências (pré vs pós)**
- Inadimplência (maturidade): {{ fmt_pct(premium.default.pre) }} → {{ fmt_pct(premium.default.post) }} (Δ {{ fmt_pp(premium.default.delta_abs) }})
- Cobertura de recebimento: {{ fmt_pct(premium.receipt.pre) }} → {{ fmt_pct(premium.receipt.post) }} (Δ {{ fmt_pp(premium.receipt.delta_abs) }})
- Limiar “premium” estimado (p75 do ticket no marco): {{ fmt_money(premium_threshold) if premium_threshold else "N/D" }}

**Onde olhar no dashboard**
- Aba **Fluxo de Caixa**: *Receita Esperada vs Receita Real (área)* (antes/depois).
- Aba **Cohort Detalhada**: curva da cohort do período premium (maturidade).

## 5) Política de Juros cobre as perdas por inadimplência?

**Resposta**
- A taxa de juros de referência do case é **{{ interest_rate }} a.m.**. A cobertura depende do volume de perdas (contratos maduros). O dashboard traz o indicador de cobertura estimada e o split recorrente vs tradicional.

**Evidências**
- {{ interest.coverage_note }}
- {{ interest.split_note }}

**Onde olhar no dashboard**
- Aba **Respostas**: ver o bloco “Política de Juros” no relatório embutido.

## Observações de Apresentação

- Use a aba **Visão Geral** para contar a história macro (mix, receita, cobertura, inadimplência) e depois aprofunde em **Cohort** para mostrar maturidade de recebimento por cohort.
"""


def render_company_answers(context: dict, out_path: Path, interest_rate: str) -> Path:
    def fmt_money(x):
        if x is None:
            return "N/D"
        return f"R$ {float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def fmt_int(x):
        if x is None:
            return "N/D"
        return f"{int(round(float(x))):,}".replace(",", ".")

    def fmt_pct(x):
        if x is None:
            return "N/D"
        return f"{float(x)*100:.1f}%".replace(".", ",")

    def fmt_pp(x):
        if x is None:
            return "N/D"
        n = f"{float(x)*100:.1f}".replace(".", ",")
        return f"{n} p.p."

    t = Template(TEMPLATE)
    txt = t.render(
        **context,
        interest_rate=interest_rate,
        fmt_money=fmt_money,
        fmt_int=fmt_int,
        fmt_pct=fmt_pct,
        fmt_pp=fmt_pp,
    )
    out_path.write_text(txt, encoding="utf-8")
    return out_path

