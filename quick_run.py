#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script rápido para executar a análise passo a passo
"""

import sys
import os
from pathlib import Path
import pandas as pd
import plotly.io as pio

# Adiciona o diretório raiz ao path
root = Path(__file__).parent
sys.path.insert(0, str(root))

def step_01_eda():
    """Executa o notebook 01 - EDA"""
    print("\n" + "="*60)
    print("📊 EXECUTANDO: 01 - Análise Exploratória e Montagem de Base")
    print("="*60)
    
    try:
        # Importa e executa o código do notebook 01
        from src.config import ensure_dirs, RAW_DIR, PROC_DIR, DEFAULT_FILES, get_events
        from src.io_utils import copy_raw_files
        from src.loaders import load_base
        from src.transform import build_schedule, assign_payments
        
        # Cria diretórios
        ensure_dirs()
        print("✓ Diretórios criados")
        
        # Copia arquivos
        paths = copy_raw_files()
        print("✓ Arquivos copiados")
        
        # Carrega base
        xlsx = paths.get('xlsx', DEFAULT_FILES['xlsx'])
        sales, payments = load_base(xlsx)
        print(f"✓ Base carregada: {len(sales)} vendas, {len(payments)} pagamentos")
        
        # Salva dados limpos
        sales.to_csv(PROC_DIR / 'sales_clean.csv', index=False)
        payments.to_csv(PROC_DIR / 'payments_clean.csv', index=False)
        print("✓ Dados limpos salvos")
        
        # Cria cronograma e aloca pagamentos
        schedule = build_schedule(sales)
        allocated = assign_payments(schedule, payments)
        print(f"✓ Cronograma criado: {len(schedule)} parcelas")
        print(f"✓ Pagamentos alocados: {len(allocated)} registros")
        
        # Salva resultados
        schedule.to_csv(PROC_DIR / 'schedule.csv', index=False)
        allocated.to_csv(PROC_DIR / 'allocated.csv', index=False)
        
        # Detecta eventos
        events = get_events()
        print(f"✓ Eventos detectados: {events}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na etapa 01: {e}")
        import traceback
        traceback.print_exc()
        return False

def step_02_cleaning():
    """Executa limpeza e engenharia de atributos"""
    print("\n" + "="*60)
    print("🧹 EXECUTANDO: 02 - Limpeza e Engenharia de Atributos")
    print("="*60)
    
    try:
        from src.config import PROC_DIR, get_events
        from src.metrics import premium_flag
        
        # Carrega dados
        sales = pd.read_csv(PROC_DIR / 'sales_clean.csv', parse_dates=['purchase_date'])
        payments = pd.read_csv(PROC_DIR / 'payments_clean.csv', parse_dates=['payment_date'])
        
        print(f"✓ Dados carregados: {len(sales)} vendas, {len(payments)} pagamentos")
        
        # Limpa duplicatas
        sales = sales.drop_duplicates(subset=['contract_id']).reset_index(drop=True)
        sales = sales[sales['purchase_date'].notna()]
        print(f"✓ Dados limpos: {len(sales)} vendas")
        
        # Adiciona flag premium
        events = get_events()
        sales2, thr = premium_flag(sales, pd.Timestamp(events['premium_offer']))
        print(f"✓ Flag premium adicionada (limiar: R$ {thr:,.2f})")
        
        # Salva resultados
        sales2.to_csv(PROC_DIR / 'sales_enriched.csv', index=False)
        pd.Series({'premium_threshold': thr}).to_json(PROC_DIR / 'premium_threshold.json')
        payments.to_csv(PROC_DIR / 'payments_enriched.csv', index=False)
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na etapa 02: {e}")
        import traceback
        traceback.print_exc()
        return False

def step_03_analysis():
    """Executa análise de negócio e gera dashboard"""
    print("\n" + "="*60)
    print("📈 EXECUTANDO: 03 - Análise de Negócio e Dashboard")
    print("="*60)
    
    try:
        from src.config import PROC_DIR, REPORTS_DIR, get_events
        from src.metrics import monthly_sales_kpis, expected_vs_received, matured_default_rate_over_time, segment_compare, interest_implied, interest_coverage
        from src.viz import ts_line, line_two_axes, expected_received_area, default_rate_plot
        
        # Carrega dados
        sales = pd.read_csv(PROC_DIR / 'sales_enriched.csv', parse_dates=['purchase_date'])
        allocated = pd.read_csv(PROC_DIR / 'allocated.csv', parse_dates=['due_date','payment_date_eff'])
        
        print(f"✓ Dados carregados: {len(sales)} vendas, {len(allocated)} registros alocados")
        
        # Calcula KPIs
        kpis = monthly_sales_kpis(sales)
        evr = expected_vs_received(allocated)
        defaults_t = matured_default_rate_over_time(allocated)
        
        print(f"✓ KPIs calculados: {len(kpis)} meses")
        print(f"✓ Inadimplência: {len(defaults_t)} períodos")
        
        # Salva resultados intermediários
        kpis.to_csv(PROC_DIR / 'kpis_monthly.csv', index=False)
        evr.to_csv(PROC_DIR / 'expected_vs_received.csv', index=False)
        defaults_t.to_csv(PROC_DIR / 'default_rates.csv', index=False)
        
        # Análises por evento
        events = get_events()
        
        # 1. Mudança de estratégia jan/2021
        high_ticket = segment_compare(kpis, pd.Timestamp(events['high_ticket_start']), 'avg_ticket')
        print(f"\n📊 ANÁLISE 1 - Mudança Estratégica (Jan/2021):")
        print(f"   Ticket médio: R$ {high_ticket['pre']:,.2f} → R$ {high_ticket['post']:,.2f}")
        print(f"   Variação: {high_ticket['delta_pct']*100:+.1f}%")
        
        # 2. Ampliação recorrente jul/2021
        recurring = segment_compare(kpis, pd.Timestamp(events['recurring_expansion']), 'contracts')
        print(f"\n📊 ANÁLISE 2 - Expansão Recorrente (Jul/2021):")
        print(f"   Contratos: {recurring['pre']:.0f} → {recurring['post']:.0f}")
        print(f"   Variação: {recurring['delta_pct']*100:+.1f}%")
        
        # 3. WhatsApp (se detectado)
        if events['whatsapp_start']:
            whatsapp = segment_compare(defaults_t.rename(columns={'month':'month','default_rate':'default'}), pd.Timestamp(events['whatsapp_start']), 'default')
            print(f"\n📊 ANÁLISE 3 - WhatsApp Cobrança:")
            pre = whatsapp.get("pre")
            post = whatsapp.get("post")
            pre_s = f"{pre:.1%}" if pre is not None else "N/D"
            post_s = f"{post:.1%}" if post is not None else "N/D"
            print(f"   Inadimplência: {pre_s} → {post_s}")
        else:
            print(f"\n⚠️  WhatsApp: data não detectada no PDF")
        
        # 4. Premium
        premium = segment_compare(defaults_t.rename(columns={'month':'month','default_rate':'default'}), pd.Timestamp(events['premium_offer']), 'default')
        print(f"\n📊 ANÁLISE 4 - Produtos Premium (Ago/2024):")
        pre = premium.get("pre")
        post = premium.get("post")
        pre_s = f"{pre:.1%}" if pre is not None else "N/D"
        post_s = f"{post:.1%}" if post is not None else "N/D"
        print(f"   Inadimplência: {pre_s} → {post_s}")
        
        # 5. Juros
        si = interest_implied(sales)
        cov = interest_coverage(allocated, si, defaults_t['month'].max())
        print(f"\n📊 ANÁLISE 5 - Política de Juros:")
        print(f"   Cobertura: {cov['coverage_ratio']:.2f}x")
        
        # Gera gráficos
        fig_ticket = line_two_axes(kpis, 'month', 'avg_ticket', 'contracts', 'Ticket médio', 'Contratos', 'Evolução do Ticket Médio e Volume de Contratos')
        fig_evr = expected_received_area(evr, 'Receita Esperada x Recebida (área)')
        fig_def = default_rate_plot(defaults_t, 'Taxa de Inadimplência por Maturidade ao Longo do Tempo')
        kpis2 = kpis.copy()
        kpis2["recurring_share_pct"] = kpis2["recurring_share"] * 100
        fig_rec_share = ts_line(kpis2, "month", "recurring_share_pct", "Participação de Vendas Recorrentes (%)", "Recorrente (%)")
        evr2_plot = evr.copy()
        evr2_plot["coverage_pct"] = (evr2_plot["received"] / evr2_plot["expected"].replace(0, pd.NA)) * 100
        fig_cov = ts_line(evr2_plot, "month", "coverage_pct", "Cobertura de Recebimento (Recebido/Esperado %)", "Cobertura (%)")
        
        # Cria dashboard HTML
        html_parts = []
        html_parts.append(pio.to_html(fig_ticket, include_plotlyjs='cdn', full_html=False))
        html_parts.append(pio.to_html(fig_evr, include_plotlyjs=False, full_html=False))
        html_parts.append(pio.to_html(fig_def, include_plotlyjs=False, full_html=False))
        html_parts.append(pio.to_html(fig_rec_share, include_plotlyjs=False, full_html=False))
        html_parts.append(pio.to_html(fig_cov, include_plotlyjs=False, full_html=False))
        
        dashboard_html = """<!doctype html>
<html lang='pt-br'>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Dashboard - Tourism Company Delinquency</title>
<style>
body{{background:#111;color:#eee;font-family:Inter,Arial,sans-serif;margin:0}}
header{{padding:24px;border-bottom:1px solid #333}}
main{{padding:24px;max-width:1200px;margin:0 auto}}
.card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:16px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:1fr;gap:16px}}
.note{{color:#bbb}}
a{{color:#7ad7ff}}
@media(min-width:1000px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<header>
<h1>Dashboard - Tourism Company Delinquency</h1>
<p class="note">Análise completa da inadimplência em pagamentos recorrentes</p>
</header>
<main>
<div class='grid'>
{plots}
</div>
</main>
</body>
</html>"""
        
        # Salva dashboard
        dashboard_path = REPORTS_DIR / 'dashboard.html'
        dashboard_path.write_text(dashboard_html.format(plots=''.join(f"<div class='card'>{p}</div>" for p in html_parts)), encoding='utf-8')
        
        high_ticket_default = segment_compare(defaults_t, pd.Timestamp(events["high_ticket_start"]), "default_rate")
        recurring_default_rec = segment_compare(defaults_t, pd.Timestamp(events["recurring_expansion"]), "recurring_default_rate")
        whatsapp_default = (
            segment_compare(defaults_t, pd.Timestamp(events["whatsapp_start"]), "default_rate")
            if events.get("whatsapp_start")
            else {"pre": None, "post": None, "delta_pct": None}
        )

        evr2 = evr.copy()
        evr2["coverage"] = evr2["received"] / evr2["expected"].replace(0, pd.NA)
        high_ticket_receipt = segment_compare(evr2, pd.Timestamp(events["high_ticket_start"]), "coverage")
        recurring_receipt = segment_compare(evr2, pd.Timestamp(events["recurring_expansion"]), "coverage")
        premium_receipt = segment_compare(evr2, pd.Timestamp(events["premium_offer"]), "coverage")

        premium_threshold = None
        try:
            premium_threshold = float(pd.read_json(PROC_DIR / "premium_threshold.json", typ="series").get("premium_threshold"))
        except Exception:
            premium_threshold = None

        cov_ratio = cov.get("coverage_ratio") if isinstance(cov, dict) else None
        split = cov.get("split") if isinstance(cov, dict) else {}
        rec_cov = split.get("recurring") if isinstance(split, dict) else None
        trad_cov = split.get("traditional") if isinstance(split, dict) else None
        cov_note = "N/D" if cov_ratio is None else f"{float(cov_ratio):.2f}x"
        rec_note = "N/D" if rec_cov is None else f"{float(rec_cov):.2f}x"
        trad_note = "N/D" if trad_cov is None else f"{float(trad_cov):.2f}x"

        context = {
            "high_ticket": {"avg_ticket": high_ticket, "default": high_ticket_default, "receipt": high_ticket_receipt},
            "recurring": {"contracts": recurring, "default_rec": recurring_default_rec, "receipt": recurring_receipt},
            "whatsapp_start": str(events.get("whatsapp_start")) if events.get("whatsapp_start") else None,
            "whatsapp": {"default": whatsapp_default},
            "premium_threshold": premium_threshold,
            "premium": {"default": premium, "receipt": premium_receipt},
            "interest": {"coverage_note": cov_note, "split_note": f"recorrente {rec_note} | tradicional {trad_note}"},
            "executive_summary": "Resumo executivo disponível no dashboard; relatório detalha os achados por iniciativa.",
        }

        import json
        with open(PROC_DIR / "analysis_context.json", "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✓ Dashboard gerado: {dashboard_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na etapa 03: {e}")
        import traceback
        traceback.print_exc()
        return False

def step_04_report():
    """Gera relatório final"""
    print("\n" + "="*60)
    print("📝 EXECUTANDO: 04 - Geração do Relatório Final")
    print("="*60)
    
    try:
        from src.config import PROC_DIR, REPORTS_DIR
        from src.report import render_report
        import json
        
        # Carrega contexto da análise
        with open(PROC_DIR / 'analysis_context.json', 'r', encoding='utf-8') as f:
            context = json.load(f)
        
        # Prepara contexto para template
        report_context = {
            'high_ticket': context.get('high_ticket', {}),
            'recurring': context.get('recurring', {}),
            'whatsapp_start': context.get('whatsapp_start'),
            'whatsapp': context.get('whatsapp', {}),
            'premium_threshold': context.get('premium_threshold'),
            'premium': context.get('premium', {}),
            'interest': context.get('interest', {}),
            'executive_summary': context.get('executive_summary', '')
        }
        
        # Gera relatório
        report_path = REPORTS_DIR / 'business_report.md'
        render_report(report_context, report_path)
        
        print(f"\n✓ Relatório gerado: {report_path}")
        
        # Imprime resumo executivo
        print("\n📋 RESUMO EXECUTIVO:")
        print("="*50)
        
        if 'high_ticket' in context and 'avg_ticket' in context['high_ticket']:
            ht = context['high_ticket']['avg_ticket']
            print(f"1. Mudança Estratégica (Jan/2021):")
            print(f"   Ticket médio: R$ {float(ht.get('pre', 0)):,.2f} → R$ {float(ht.get('post', 0)):,.2f}")
            print(f"   Variação: {float(ht.get('delta_pct', 0))*100:+.1f}%")
        
        if 'recurring' in context and 'contracts' in context['recurring']:
            rec = context['recurring']['contracts']
            print(f"\n2. Expansão Recorrente (Jul/2021):")
            print(f"   Contratos: {float(rec.get('pre', 0)):.0f} → {float(rec.get('post', 0)):.0f}")
            print(f"   Variação: {float(rec.get('delta_pct', 0))*100:+.1f}%")
        
        if 'premium' in context and 'default' in context['premium']:
            prem = context['premium']['default']
            print(f"\n3. Produtos Premium (Ago/2024):")
            print(f"   Inadimplência: {float(prem.get('pre', 0))*100:.1f}% → {float(prem.get('post', 0))*100:.1f}%")
        
        if 'interest_coverage' in context:
            cov = context['interest_coverage']
            if isinstance(cov, dict) and 'coverage_ratio' in cov:
                print(f"\n4. Política de Juros:")
                print(f"   Cobertura: {float(cov['coverage_ratio']):.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na etapa 04: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa todo o pipeline de análise"""
    print("🚀 INICIANDO ANÁLISE COMPLETA - TOURISM COMPANY DELINQUENCY")
    print("="*80)
    
    # Executa etapas
    steps = [
        ("Análise Exploratória", step_01_eda),
        ("Limpeza e Engenharia", step_02_cleaning),
        ("Análise de Negócio", step_03_analysis),
        ("Relatório", step_04_report),
    ]
    
    results = []
    for name, func in steps:
        print(f"\n🔍 Executando: {name}")
        success = func()
        results.append((name, success))
        if not success:
            print(f"❌ Falha na etapa: {name}")
            break
    
    print("\n" + "="*80)
    print("📊 RESUMO DA EXECUÇÃO")
    print("="*80)
    for name, success in results:
        status = "✅ SUCESSO" if success else "❌ FALHA"
        print(f"{status} - {name}")
    
    if all(success for _, success in results):
        print("\n🎉 ANÁLISE CONCLUÍDA COM SUCESSO!")
        print(f"📊 Dashboard: file:///{REPORTS_DIR / 'dashboard.html'}")
        print(f"📋 Relatório: file:///{REPORTS_DIR / 'business_report.md'}")
        return True
    else:
        print("\n❌ ANÁLISE FALHOU EM ALGUMAS ETAPAS")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
