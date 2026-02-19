# Respostas às Perguntas do Negócio

Este documento resume as respostas para as perguntas levantadas pela empresa, embasadas nos dados e com referência direta aos gráficos do dashboard.

## 1) Priorização de Ticket Médio Mais Alto (a partir de Jan/2021)

**Resposta**
- O ticket médio aumentou de forma relevante no pós-evento, mas houve piora em inadimplência e redução na cobertura de recebimento, indicando maior risco/custo para o caixa.

**Evidências (pré vs pós)**
- Ticket médio: R$ 3.638,33 → R$ 6.046,16 (Δ 66,2%)
- Inadimplência (maturidade): 14,6% → 41,2% (Δ 26,7 p.p.)
- Cobertura de recebimento: 32,0% → 13,5% (Δ -18,5 p.p.)

**Onde olhar no dashboard**
- Aba **Informações Gerais**: *% Inadimplência por Produto* e *% Inadimplência por Região*.
- Aba **Fluxo de Caixa**: *Receita Esperada vs Receita Real (área)* (gap/hover de inadimplência).

## 2) Expansão do Parcelamento Recorrente (a partir de Jul/2021)

**Resposta**
- A expansão do recorrente aumentou o volume de contratos e a participação do recorrente no mix. Em paralelo, a inadimplência do recorrente aumentou, o que exige atenção ao desenho de cobrança/limites.

**Evidências (pré vs pós)**
- Contratos: 897 → 2.842 (Δ 216,9%)
- Inadimplência do recorrente: 18,1% → 42,5% (Δ 24,4 p.p.)
- Cobertura de recebimento: 28,3% → 13,0% (Δ -15,3 p.p.)

**Onde olhar no dashboard**
- Aba **Informações Gerais**: *% Inadimplência e Share Recorrente por Forma de Pagamento*.
- Aba **Análise Cohort**: evolução de *% Recorrentes (share)* por cohort.

## 3) WhatsApp de Cobrança (a partir de Jul/2021)

**Resposta**
- O WhatsApp não evidenciou uma melhora clara e sustentada de cobertura sem custo; a inadimplência e a cobertura precisam ser analisadas junto à expansão do recorrente no mesmo período.

**Evidências (pré vs pós)**
- Inadimplência (maturidade): N/D → 38,1% (Δ N/D)

**Onde olhar no dashboard**
- Aba **Fluxo de Caixa**: *Receita Esperada vs Receita Real (área)* (avaliar cobertura no período).
- Aba **Análise Cohort**: *% Inadimplência* por cohort (mudança estrutural vs sazonalidade).

## 4) Oferta Premium (a partir de Jul/2024)

**Resposta**
- A entrada de produtos premium deve ser acompanhada com foco em caixa e inadimplência por maturidade. O indicador principal é o gap entre esperado e recebido e a trajetória da inadimplência após o marco.

**Evidências (pré vs pós)**
- Inadimplência (maturidade): 27,3% → 52,7% (Δ 25,4 p.p.)
- Cobertura de recebimento: 21,4% → 8,1% (Δ -13,3 p.p.)
- Limiar “premium” estimado (p75 do ticket no marco): R$ 8.496,00

**Onde olhar no dashboard**
- Aba **Fluxo de Caixa**: *Receita Esperada vs Receita Real (área)* (antes/depois).
- Aba **Cohort Detalhada**: curva da cohort do período premium (maturidade).

## 5) Política de Juros cobre as perdas por inadimplência?

**Resposta**
- A taxa de juros de referência do case é **2,49% a.m.**. A cobertura depende do volume de perdas (contratos maduros). O dashboard traz o indicador de cobertura estimada e o split recorrente vs tradicional.

**Evidências**
- Cobertura estimada: 9.85x
- Recorrente: 0.84x; Tradicional: N/D

**Onde olhar no dashboard**
- Aba **Respostas**: ver o bloco “Política de Juros” no relatório embutido.

## Observações de Apresentação

- Use a aba **Visão Geral** para contar a história macro (mix, receita, cobertura, inadimplência) e depois aprofunde em **Cohort** para mostrar maturidade de recebimento por cohort.