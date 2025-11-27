import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

def dark_layout(fig, title):
    fig.update_layout(template="plotly_dark", title=title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=40,r=20,t=60,b=40))
    fig.update_xaxes(showgrid=True, gridcolor="#333")
    fig.update_yaxes(showgrid=True, gridcolor="#333")
    return fig

def line_two_axes(df: pd.DataFrame, x: str, y_left: str, y_right: str, name_left: str, name_right: str, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y_left], name=name_left, mode="lines+markers", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df[x], y=df[y_right], name=name_right, mode="lines+markers", yaxis="y2"))
    fig.update_layout(yaxis=dict(title=name_left), yaxis2=dict(title=name_right, overlaying="y", side="right"))
    return dark_layout(fig, title)

def ts_line(df: pd.DataFrame, x: str, y: str, title: str, name: str):
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    fig.update_traces(name=name)
    return dark_layout(fig, title)

def stacked_bars(df: pd.DataFrame, x: str, y_cols: list[str], names: list[str], title: str):
    fig = go.Figure()
    for c, n in zip(y_cols, names):
        fig.add_trace(go.Bar(x=df[x], y=df[c], name=n))
    fig.update_layout(barmode="stack")
    return dark_layout(fig, title)

def expected_received_area(df: pd.DataFrame, title: str):
    d = df.copy()
    d["expected"] = pd.to_numeric(d["expected"], errors="coerce").fillna(0.0)
    d["received"] = pd.to_numeric(d["received"], errors="coerce").fillna(0.0)
    denom = d["expected"].replace(0, pd.NA)
    d["delinq_pct"] = (1 - (d["received"] / denom)).clip(lower=0).fillna(0.0)
    custom = d[["received", "expected", "delinq_pct"]].to_numpy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["month"],
            y=d["received"],
            name="Recebido",
            mode="lines",
            fill="tozeroy",
            line=dict(color="rgba(99, 179, 237, 1)", width=2),
            fillcolor="rgba(99, 179, 237, 0.35)",
            customdata=custom,
            hovertemplate=(
                "Mês: %{x|%Y-%m}<br>"
                "Recebido: R$ %{customdata[0]:,.2f}<br>"
                "Esperado: R$ %{customdata[1]:,.2f}<br>"
                "Inadimplência: %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["month"],
            y=d["expected"],
            name="Gap (inadimplência)",
            mode="lines",
            fill="tonexty",
            line=dict(color="rgba(0,0,0,0)", width=0),
            fillcolor="rgba(251, 191, 36, 0.25)",
            customdata=custom,
            hovertemplate=(
                "Mês: %{x|%Y-%m}<br>"
                "Recebido: R$ %{customdata[0]:,.2f}<br>"
                "Esperado: R$ %{customdata[1]:,.2f}<br>"
                "Inadimplência: %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["month"],
            y=d["expected"],
            name="Esperado",
            mode="lines",
            line=dict(color="rgba(224, 231, 255, 0.9)", width=2),
            customdata=custom,
            hovertemplate=(
                "Mês: %{x|%Y-%m}<br>"
                "Recebido: R$ %{customdata[0]:,.2f}<br>"
                "Esperado: R$ %{customdata[1]:,.2f}<br>"
                "Inadimplência: %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )

    fig.update_yaxes(title="R$", rangemode="tozero")
    return dark_layout(fig, title)

def interest_coverage_indicators(cov: dict, title: str):
    cov = cov if isinstance(cov, dict) else {}
    split = cov.get("split") if isinstance(cov.get("split"), dict) else {}

    total = cov.get("coverage_ratio")
    rec = split.get("recurring")
    trad = split.get("traditional")
    no_interest = bool(cov.get("no_interest_data"))

    def val(x):
        if x is None:
            return float("nan")
        return float(x)

    def label(name: str, x):
        if no_interest:
            return f"{name} (sem juros)"
        return f"{name} (N/D)" if x is None else name

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        column_widths=[0.34, 0.33, 0.33],
    )
    fig.add_trace(
        go.Indicator(mode="number", value=val(total), number={"suffix": "x", "valueformat": ".2f"}, title={"text": label("Cobertura", total)}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Indicator(mode="number", value=val(rec), number={"suffix": "x", "valueformat": ".2f"}, title={"text": label("Recorrente", rec)}),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Indicator(mode="number", value=val(trad), number={"suffix": "x", "valueformat": ".2f"}, title={"text": label("Tradicional", trad)}),
        row=1,
        col=3,
    )
    if no_interest:
        fig.add_annotation(
            x=0.5,
            y=-0.25,
            xref="paper",
            yref="paper",
            showarrow=False,
            text="A base não traz juros mensuráveis (surcharge ≈ 0); cobertura não é inferível com estes dados.",
            font=dict(size=11, color="#bbb"),
        )
    fig.update_layout(height=240)
    return dark_layout(fig, title)

def add_event_markers(fig, events: dict):
    events = events if isinstance(events, dict) else {}
    labels = {
        "high_ticket_start": "Ticket alto",
        "recurring_expansion": "Expansão recorrente",
        "whatsapp_start": "WhatsApp cobrança",
        "premium_offer": "Oferta premium",
    }
    for k, lab in labels.items():
        dt = events.get(k)
        if dt is None:
            continue
        x = pd.Timestamp(dt)
        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="#888", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=x,
            y=1,
            xref="x",
            yref="paper",
            text=lab,
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10, color="#bbb"),
        )
    return fig

def cohort_area(df: pd.DataFrame, title: str):
    d = df.copy()
    d["purchase_month"] = pd.to_datetime(d["purchase_month"])
    d = d.sort_values(["purchase_month", "age_months"])
    cohorts = d["purchase_month"].dropna().sort_values().unique().tolist()
    if not cohorts:
        return dark_layout(go.Figure(), title)

    def slice_cohort(c):
        s = d[d["purchase_month"] == c]
        x = s["age_months"]
        delinq = s["delinquency_pct"] * 100
        rec = s["recurring_pct"] * 100
        custom = s[["cum_received", "cum_expected"]].to_numpy()
        return x, delinq, rec, custom

    c0 = cohorts[-1]
    x0, delinq0, rec0, custom0 = slice_cohort(c0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x0,
            y=delinq0,
            name="Inadimplência (%)",
            mode="lines",
            fill="tozeroy",
            line=dict(color="rgba(239, 68, 68, 1)", width=2),
            fillcolor="rgba(239, 68, 68, 0.25)",
            customdata=custom0,
            hovertemplate=(
                "Cohort: %{meta|%Y-%m}<br>"
                "Mês após compra: %{x}<br>"
                "Inadimplência: %{y:.1f}%<br>"
                "Recebido (acum.): R$ %{customdata[0]:,.2f}<br>"
                "Esperado (acum.): R$ %{customdata[1]:,.2f}<extra></extra>"
            ),
            meta=c0,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x0,
            y=rec0,
            name="Recorrente (share %)",
            mode="lines",
            fill="tozeroy",
            line=dict(color="rgba(99, 179, 237, 1)", width=2),
            fillcolor="rgba(99, 179, 237, 0.18)",
            customdata=custom0,
            hovertemplate=(
                "Cohort: %{meta|%Y-%m}<br>"
                "Mês após compra: %{x}<br>"
                "Recorrente: %{y:.1f}%<extra></extra>"
            ),
            meta=c0,
        )
    )

    buttons = []
    for c in cohorts:
        x, delinq, rec, custom = slice_cohort(c)
        buttons.append(
            dict(
                label=pd.Timestamp(c).strftime("%Y-%m"),
                method="update",
                args=[
                    {"x": [x, x], "y": [delinq, rec], "customdata": [custom, custom], "meta": [c, c]},
                    {"title": f"{title} — cohort {pd.Timestamp(c).strftime('%Y-%m')}"},
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                x=1,
                y=1.15,
                xanchor="right",
                yanchor="top",
                buttons=buttons,
                showactive=True,
            )
        ]
    )
    fig.update_xaxes(title="Mês após compra", dtick=1)
    fig.update_yaxes(title="%", rangemode="tozero")
    return dark_layout(fig, f"{title} — cohort {pd.Timestamp(c0).strftime('%Y-%m')}")

def default_rate_plot(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["month"], y=(df["default_rate"]*100), name="Inadimplência (%)", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df["month"], y=(df["recurring_default_rate"]*100), name="Recorrente (%)", mode="lines"))
    if "traditional_default_rate" in df.columns and df["traditional_default_rate"].notna().any():
        fig.add_trace(go.Scatter(x=df["month"], y=(df["traditional_default_rate"]*100), name="Tradicional (%)", mode="lines"))
    fig.update_yaxes(title="%", rangemode="tozero")
    return dark_layout(fig, title)
