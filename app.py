# Guaranteed-COE buffer calculator.
# pip install -r requirements.txt ; streamlit run app.py
# Reads the tables exported by notebooks 06 and 07 in data/processed/; never re-runs the models.
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROC = Path(__file__).resolve().parent / "data" / "processed"
NAVY, TEAL, RED, MUTED, LIGHT = "#1F2A44", "#1A7F8E", "#C0392B", "#5B6576", "#C9D3DE"
CONDITION = {"Low": "Calm", "Medium": "Moderate", "High": "Turbulent"}
K_LABEL = {
    1: "Price-war dealer: a lost sale hurts as much as a shortfall",
    1.5: "Competitive dealer: overpricing is costly",
    2.3: "Close to what a flat 5% implies (k ≈ 2.6)",
    4: "Typical: a shortfall is 4× worse than a lost sale",
    9: "Thin margins: a shortfall is 9× worse",
    19: "Very cautious: shortfalls are almost unacceptable",
}
RULES = {"R0 No buffer": "No buffer", "R1 Fixed 5%": "Fixed 5%", "R7 Vol-scaled": "This model"}

st.set_page_config(page_title="COE Buffer Calculator", page_icon="🚗", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric, button, input, select, textarea {{
    font-family: 'Inter', sans-serif !important; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
h1, h2, h3 {{ color: {NAVY}; font-weight: 700; letter-spacing: -0.01em; }}
.app-header {{ border-left: 5px solid {TEAL}; padding: 0.2rem 0 0.2rem 1rem; margin-bottom: 0.5rem; }}
.app-header h1 {{ font-size: 1.9rem; margin: 0; padding: 0; }}
.app-header p {{ color: {MUTED}; margin: 0.3rem 0 0 0; font-size: 0.98rem; }}
.pill {{ display: inline-block; background: #E6F2F4; color: {TEAL}; border-radius: 999px;
        padding: 0.1rem 0.65rem; font-size: 0.8rem; font-weight: 600; margin-right: 0.4rem; }}
.pill.warn {{ background: #FBEAEA; color: {RED}; }}
div[data-testid="stMetric"] {{ background: #F4F6F8; border-radius: 10px; padding: 0.9rem 1rem; }}
div[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.85rem; font-weight: 500; }}
div[data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 700; }}
section[data-testid="stSidebar"] {{ background: #F4F6F8; }}
.small {{ color: {MUTED}; font-size: 0.85rem; }}
.card {{ background: #F4F6F8; border-radius: 10px; padding: 0.9rem 1rem; height: 100%; }}
.card.accent {{ background: #E6F2F4; border-left: 4px solid {TEAL}; }}
.card .lab {{ color: {MUTED}; font-size: 0.85rem; font-weight: 500; }}
.card .val {{ color: {NAVY}; font-size: 1.85rem; font-weight: 700; line-height: 1.25; margin-top: 0.2rem; }}
.card .sub {{ color: {MUTED}; font-size: 0.82rem; margin-top: 0.15rem; }}
.tip {{ display: inline-block; width: 1.05em; height: 1.05em; line-height: 1.05em;
       text-align: center; border: 1px solid {MUTED}; border-radius: 50%; font-size: 0.7rem; color: {MUTED};
       cursor: help; margin-left: 0.3rem; font-style: normal; font-weight: 600; }}
.tip:hover {{ border-color: {TEAL}; color: {TEAL}; }}
.tip .tiptext {{ visibility: hidden; opacity: 0; transition: opacity 0.12s; position: absolute; z-index: 1000;
       left: -0.2rem; right: -0.2rem; top: 1.7em; background: {NAVY}; color: #FFFFFF;
       text-align: left; font-size: 0.78rem; font-weight: 400; line-height: 1.4; padding: 0.55rem 0.7rem;
       border-radius: 6px; box-shadow: 0 4px 14px rgba(31,42,68,0.25); white-space: normal; }}
.tip:hover .tiptext {{ visibility: visible; opacity: 1; }}
.card .lab {{ position: relative; }}
.key {{ color: {RED}; font-weight: 600; font-size: 0.85rem; margin-left: 0.6rem; }}
.formula {{ font-size: 1.05rem; color: {NAVY}; background: #E6F2F4; border-radius: 8px;
           padding: 0.6rem 0.9rem; display: inline-block; }}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    bt = pd.read_csv(f"{PROC}/buffer_table.csv")
    bt["signal_range"] = bt.signal_range.str.replace("<=", "≤")
    today = pd.read_csv(f"{PROC}/buffer_today.csv").set_index("category")
    form = pd.read_csv(f"{PROC}/buffer_formula.csv")
    res = pd.read_csv(f"{PROC}/backtest_results.csv")
    by_model = pd.read_csv(f"{PROC}/buffer_today_by_model.csv")
    return bt, today, form, res, by_model


bt, today, form, res, by_model = load()


def card(col, label, value, sub="", tip="", accent=False):
    """A KPI tile; hovering the ⓘ shows `tip`."""
    i = f'<span class="tip">i<span class="tiptext">{tip}</span></span>' if tip else ""
    col.markdown(f'<div class="card{" accent" if accent else ""}"><div class="lab">{label}{i}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)
CHART_CONFIG = {"font": "Inter", "view": {"stroke": None},
                "axis": {"labelColor": MUTED, "titleColor": MUTED, "gridColor": "#EEF1F4", "domain": False,
                         "tickColor": "#EEF1F4", "labelFontSize": 12, "titleFontSize": 12,
                         "titleFontWeight": 500}}

# ----------------------------------------------------------------------------- header
head, about = st.columns([6, 1])
with head:
    st.markdown("""<div class="app-header"><h1>Guaranteed-COE Buffer Calculator</h1>
    <p>How much should a dealer add to a guaranteed-COE car price so it rarely loses money,
    without overpricing the car? Pick a category, the market condition and your cost ratio.</p></div>""",
                unsafe_allow_html=True)
with about:
    with st.popover("ⓘ About", use_container_width=True):
        st.markdown("""
**What this is.** A dealer selling a *guaranteed-COE* package fixes the car price at today's
premium and secures the COE about two bidding rounds (one month) later. Any rise in between comes
out of the dealer's margin. The **buffer** is the extra added to the price to cover that risk.

**How the buffer is set.** It is a newsvendor decision: charge the **τ = k / (1 + k)** percentile
of the forecast 2-round rise, where **k** is how many times worse a shortfall is than losing a sale
by overpricing. The forecast comes from a volatility-scaled model tested out of sample, 2014–2026.

**How to read it.**
- *Recommended buffer*: the typical buffer the model charged in this market condition, applied to
  the premium you enter.
- *Too small historically*: share of past sales where the rise beat the buffer. At k = 4 the
  target is about 20%.
- *Formula buffer*: the same model using **today's** volatility (z × volatility × premium).

**Data.** LTA COE Bidding Results (data.gov.sg), Jan 2010 – Oct 2026 round 2. IS5126 Group 4.

*Hypothetical backtest performance, not a guarantee. The cost of overpricing is private to each
dealer, so you choose k.*
""")

# ----------------------------------------------------------------------------- inputs
with st.sidebar:
    st.markdown("### Quote settings")
    cat = st.selectbox("COE category", ["A", "B", "E"], index=1,
                       format_func=lambda c: {"A": "Cat A · smaller cars", "B": "Cat B · larger cars",
                                              "E": "Cat E · open category"}[c],
                       help="The COE category of the car being sold.")
    now = today.loc[cat]
    premium = st.number_input("Premium quoted (S$)", min_value=10_000, max_value=300_000, step=1_000,
                              value=int(round(now.premium_now)),
                              help="Defaults to the latest closing premium (Oct 2026, round 2). "
                                   "Change it to price a quote at a different premium.")
    conds = ["Low", "Medium", "High"]
    cond = st.radio("Market condition", conds, index=conds.index(now.condition), horizontal=True,
                    format_func=lambda c: CONDITION[c] + (" (today)" if c == now.condition else ""),
                    help="Tercile of recent volatility: the std. dev. of 1-round premium changes "
                         "over the last 12 rounds. Defaults to today's condition.")
    k = st.select_slider("Cost ratio k", options=[1, 1.5, 2.3, 4, 9, 19], value=4,
                         help="How many times worse it is to be S$1 short (the dealer pays) than to "
                              "overprice by S$1 (lost or discounted sales). Only the dealer knows this.")
    st.caption(K_LABEL[k])
    st.divider()
    period = st.selectbox("Backtest period", ["2014-26", "2014-19", "2020-22", "2023-26"],
                          format_func=lambda p: {"2014-26": "Full 2014–2026", "2014-19": "2014–2019 (calm)",
                                                 "2020-22": "2020–2022 (turbulent)",
                                                 "2023-26": "2023–2026 (recent)"}[p],
                          help="Period used for the comparison with today's rules.")

tau = min(bt.tau.unique(), key=lambda x: abs(x - k / (1 + k)))
row = bt[(bt.category == cat) & (bt.condition == cond) & (bt.tau == tau)].iloc[0]
buffer_sgd = row.buffer_pct / 100 * premium
z = form.loc[form.tau == tau, "z"].iloc[0]
formula_pct = max(0.0, z * now.vol_ewma)

# ----------------------------------------------------------------------------- headline
st.markdown(f'<span class="pill">Cat {cat}</span><span class="pill">{CONDITION[cond]} market · '
            f'volatility {row.signal_range}</span><span class="pill">k = {k} → {tau * 100:.0f}th percentile</span>',
            unsafe_allow_html=True)
st.write("")
m1, m2, m3, m4 = st.columns(4)
card(m1, "Recommended buffer", f"S${buffer_sgd:,.0f}", f"{row.buffer_pct:.1f}% of the premium",
     "Median out-of-sample buffer of the volatility-scaled model in this market condition (2014-2026), "
     "as a % of the premium, applied to the premium you entered.", accent=True)
card(m2, "COE cost to build in", f"S${premium + buffer_sgd:,.0f}", f"premium S${premium:,.0f} + buffer",
     "The COE part of the guaranteed price: the premium plus the buffer. The rest of the car price comes on top.")
card(m3, "Too small historically", f"{row.underwater_pct:.0f}%", f"of {row.n_rounds} past sales · target ≈ {1 - tau:.0%}",
     "Share of past sales in this condition where the 2-round rise exceeded the buffer.")
card(m4, "Average shortfall when short", f"S${row.avg_shortfall:,.0f}", "paid by the dealer",
     "When the buffer was too small, the average amount the dealer still had to pay.")

if cond != now.condition:
    st.info(f"Cat {cat} is **{CONDITION[now.condition].lower()}** today; you are pricing a "
            f"**{CONDITION[cond].lower()}** market.", icon="ℹ️")

st.write("")

# ----------------------------------------------------------------------------- tabs
t1, t2, t3, t4 = st.tabs(["Buffer by cost ratio", "Today's market", "Compare with today's rules",
                          "Method & limitations"])


def bars(df, x, x_title, selected, sort):
    df = df.assign(sel=df[x] == selected)
    base = alt.Chart(df).encode(x=alt.X(f"{x}:N", sort=sort, title=x_title, axis=alt.Axis(labelAngle=0)))
    b = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=42).encode(
        y=alt.Y("buffer_pct:Q", title="Buffer, % of premium"),
        color=alt.condition("datum.sel", alt.value(TEAL), alt.value(LIGHT)),
        tooltip=[alt.Tooltip(f"{x}:N", title=x_title),
                 alt.Tooltip("buffer_pct:Q", title="Buffer %", format=".1f"),
                 alt.Tooltip("buffer_now:Q", title="Buffer S$", format=",.0f"),
                 alt.Tooltip("underwater_pct:Q", title="Too small, % of sales", format=".0f")])
    label = base.mark_text(dy=-8, fontSize=12, fontWeight=600, color=NAVY).encode(
        y="buffer_pct:Q", text=alt.Text("buffer_pct:Q", format=".1f"))
    rule = alt.Chart(pd.DataFrame({"y": [5]})).mark_rule(color=RED, strokeDash=[5, 4], strokeWidth=2).encode(y="y:Q")
    return (b + label + rule).properties(height=320).configure(**CHART_CONFIG)


KEY = '<span class="key">- - - fixed 5%</span>'
with t1:
    left, right = st.columns([3, 2])
    d = bt[(bt.category == cat) & (bt.condition == cond)].copy()
    d["k_lab"] = d.k.map(lambda v: f"k = {v:g}")
    d["buffer_now"] = d.buffer_pct / 100 * premium
    with left:
        st.markdown(f"**Cat {cat}, {CONDITION[cond].lower()} market: buffer for each cost ratio**{KEY}",
                    unsafe_allow_html=True)
        st.altair_chart(bars(d, "k_lab", "Cost ratio", f"k = {k:g}", list(d.k_lab)), use_container_width=True, theme=None)
    with right:
        c = bt[(bt.category == cat) & (bt.tau == tau)].copy()
        c["cond_lab"] = c.condition.map(CONDITION)
        c["buffer_now"] = c.buffer_pct / 100 * premium
        st.markdown(f"**Cat {cat}, k = {k}: buffer by market condition**{KEY}", unsafe_allow_html=True)
        st.altair_chart(bars(c, "cond_lab", "Market condition", CONDITION[cond], list(CONDITION.values())),
                        use_container_width=True, theme=None)
    st.markdown('<p class="small">The cost ratio moves the buffer most; recent volatility comes next. '
                'Bid pressure barely matters, so it is not an input. Hover over a bar for details.</p>',
                unsafe_allow_html=True)

with t2:
    a, b = st.columns([1, 1])
    with a:
        with st.container(border=True):
            st.markdown(f"**Cat {cat} at the latest round (Oct 2026, round 2)**")
            x1, x2 = st.columns(2)
            card(x1, "Closing premium", f"S${now.premium_now:,.0f}", "Oct 2026, round 2")
            band = bt[(bt.category == cat) & (bt.condition == now.condition)].signal_range.iloc[0]
            card(x2, "Market condition", CONDITION[now.condition], f"volatility band {band}",
                 "Tercile of the 12-round volatility (2014-2026 cut-offs).")
            st.write("")
            y1, y2 = st.columns(2)
            card(y1, "12-round volatility", f"{now.vol_12:.1%}", "sets the market condition",
                 "Std. dev. of 1-round % changes over the last 12 rounds.")
            card(y2, "EWMA volatility", f"{now.vol_ewma:.1%}", "used by the formula",
                 "Exponentially weighted volatility (lambda = 0.94).")
            st.caption(f"Bid ratio: {now.bid_ratio:.2f} bids per COE. It is shown for context; it does "
                       "not predict the next rise.")
    with b:
        with st.container(border=True):
            st.markdown("**One-line formula, using today's volatility**")
            st.markdown('<span class="formula">buffer = z × recent volatility × premium</span>',
                        unsafe_allow_html=True)
            st.write("")
            card(st, "Formula buffer today", f"S${formula_pct * premium:,.0f}",
                 f"{formula_pct:.1%} of the premium = {z:.2f} × {now.vol_ewma:.1%}",
                 "z is the k/(1+k) percentile of past rises divided by volatility, pooled across categories.",
                 accent=True)
            st.write("")
            st.dataframe(form[["k", "tau", "z"]].rename(columns={"tau": "percentile"}).style.format(
                {"k": "{:g}", "percentile": "{:.0%}", "z": "{:.2f}"}), hide_index=True, use_container_width=True)
    st.markdown(f'<p class="small">Why two numbers? The <b>recommended buffer</b> is the typical buffer in a '
                f'{CONDITION[cond].lower()} market; the <b>formula buffer</b> uses today\'s exact volatility, '
                f'which can sit anywhere inside that band.</p>', unsafe_allow_html=True)
    m = by_model[(by_model.category == cat) & (by_model.k == k)]
    if len(m):
        vals = m.drop(columns=["category", "k"]).iloc[0] / now.premium_now * premium
        st.markdown(f'<p class="small">Model uncertainty: our forecast models ({", ".join(vals.index)}) '
                    f'give S${vals.min():,.0f} ({vals.idxmin()}) to S${vals.max():,.0f} ({vals.idxmax()}) '
                    f'for Cat {cat} at k = {k} today.</p>', unsafe_allow_html=True)

def backtest(cat, period, tau):
    """Backtest summary for one category. Per-category results are stored by sub-period, so the full
    period is the sales-weighted combination of 2014-19, 2020-22 and 2023-26."""
    r = res[(res.category == cat) & res.rule.isin(RULES) & ((res.tau.round(2) == tau) | res.tau.isna())]
    if period != "2014-26":
        r = r[r.period == period]
    costs = [c for c in r if c.startswith("cost_k")]

    def combine(g):
        w, miss = g.n / g.n.sum(), g.n * g.underwater_pct
        out = {"underwater_pct": (w * g.underwater_pct).sum(), "avg_buffer": (w * g.avg_buffer).sum(),
               "avg_shortfall": (miss * g.avg_shortfall).sum() / miss.sum(),
               "worst_shortfall": g.worst_shortfall.max()}
        return pd.Series(out | {c: (w * g[c]).sum() for c in costs})

    out = r.groupby("rule").apply(combine, include_groups=False)
    return out.rename(index=RULES).loc[list(RULES.values())].rename_axis("Rule")


with t3:
    r = backtest(cat, period, tau)
    cost_col = f"cost_k{k:g}"
    cols = {"underwater_pct": "Sales underwater", "avg_buffer": "Average buffer",
            "avg_shortfall": "Avg shortfall when short", "worst_shortfall": "Worst shortfall"}
    if cost_col in r:
        cols[cost_col] = f"Cost per car at k = {k:g}"
    table = r[list(cols)].rename(columns=cols)
    st.markdown(f"**Cat {cat}, one simulated sale per round, {period.replace('-', '–20')}: "
                f"this model at the {tau * 100:.0f}th percentile vs what dealers do today**")
    sty = table.style.format({c: "{:.1f}%" if c == "Sales underwater" else "S${:,.0f}" for c in table})
    if cost_col in r:
        sty = sty.highlight_min(subset=[cols[cost_col]], color="#E6F2F4")
    st.dataframe(sty, use_container_width=True)
    st.markdown('<p class="small"><b>Cost per car</b> = shortfall + excess buffer ÷ k, the dealer\'s total cost '
                'when a shortfall is k times worse than overpricing (shown for k = 1, 2.3, 4 and 9). '
                'Across all categories, fixed 5% is about as good as the model for a dealer with k ≈ 2–4, '
                'the k it implies; for other cost ratios the model rule is cheaper. All rules sit on the '
                'same protection-vs-cost frontier: more protection always costs more buffer.</p>',
                unsafe_allow_html=True)

with t4:
    st.markdown("""
**Method.**
1. *Target*: the % rise in the premium over the next 2 rounds, from LTA data Jan 2010 – Oct 2026.
2. *Forecast*: a volatility-scaled quantile model, tested in strict time order (rolling origin from
   2014; it only sees rises already known at each quote). Chosen on 2014–19, checked on 2020–26.
3. *Decision*: newsvendor rule, buffer = the k/(1+k) percentile of the forecast rise.
4. *Table*: median out-of-sample buffer by volatility tercile, as a % of the premium.

**Limitations.**
- Hypothetical backtest for one price-taking dealer; if every dealer changed rule at once,
  premiums themselves could move.
- The cost of overpricing is unknown, so k is your choice, not an estimate.
- Policy shocks (quota injections, rule changes) cannot be predicted; no buffer covers every spike.
- About 400 rounds per category, with a category redefinition in May 2022.

Full analysis and code: notebooks 01–09 in this repository.
""")

st.divider()
st.caption("IS5126 Group 4 · Source: LTA COE Bidding Results (data.gov.sg), Jan 2010 – Oct 2026 round 2 · "
           "Hypothetical backtest performance, not proven savings.")
