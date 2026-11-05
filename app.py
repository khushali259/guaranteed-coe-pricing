# Guaranteed-COE buffer calculator (optional live demo).
# pip install streamlit ; streamlit run app.py
# Reads data/processed/buffer_table.csv exported by notebooks/06_newsvendor.ipynb; never re-runs the models.
import pandas as pd
import streamlit as st

bt = pd.read_csv("data/processed/buffer_table.csv")
today = pd.read_csv("data/processed/buffer_today.csv")

st.title("Guaranteed-COE buffer calculator")
st.caption("Newsvendor buffer for a dealer quoting today's premium and securing the COE two rounds later. "
           "Backtest: Cat A/B/E, quotes Jan 2014 - Sep 2026, out-of-sample forecasts only.")

cat = st.selectbox("COE category", ["A", "B", "E"])
cond = st.selectbox("Recent volatility (std. dev. of 1-round premium changes over the last 12 rounds)",
                    ["Low", "Medium", "High"])
k = st.select_slider("Underpricing costs k times as much as overpricing",
                     options=[1, 1.5, 2.3, 4, 9, 19], value=4)
tau = round(k / (1 + k), 2)
tau = min(bt.tau.unique(), key=lambda x: abs(x - tau))
row = bt[(bt.category == cat) & (bt.condition == cond) & (bt.tau == tau)].iloc[0]

st.metric("Recommended buffer", f"S${row.buffer_sgd:,.0f}", f"{row.buffer_pct:.1f}% of premium",
          delta_color="off")
st.write(f"Critical ratio tau = {tau}. Volatility range for '{cond}' in Cat {cat}: {row.signal_range}. "
         f"Today's volatility in Cat {cat}: {today.loc[today.category == cat, 'vol_12'].iloc[0]:.1%}.")
st.write(f"In the 2014-2026 backtest this buffer was too small on **{row.underwater_pct:.0f}%** "
         f"of sales, with an average shortfall of **S${row.avg_shortfall:,.0f}** when it was.")

st.subheader("One-line formula (volatility-scaled rule)")
form = pd.read_csv("data/processed/buffer_formula.csv")
z = form.loc[form.tau == tau, "z"].iloc[0]
vol_now = today.loc[today.category == cat, "vol_ewma"].iloc[0]
st.write(f"buffer = z x recent volatility x premium, with z = **{z:.2f}** at k = {k}. "
         f"Cat {cat}'s EWMA volatility today is {vol_now:.1%}, so the formula gives "
         f"**{max(0, z * vol_now):.1%}** of the premium.")

st.subheader("Forecast made at the latest round (Oct 2026, round 2)")
st.dataframe(today[today.category == cat].round(0), hide_index=True)
st.caption("Hypothetical backtest performance, not proven savings. c_o (the cost of overpricing) is not public, "
           "so the dealer chooses k.")
