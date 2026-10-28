"""Newsvendor buffer (Phase 6): b* = F^{-1}(c_u / (c_u + c_o)) of the forecast rise."""
import numpy as np
import pandas as pd


def tau_to_k(tau):
    """Critical ratio tau = k / (1 + k)  <=>  k = c_u / c_o = tau / (1 - tau)."""
    return tau / (1 - tau)


def k_to_tau(k):
    return k / (1 + k)


def buffer_sgd(q_pct, premium_now):
    """Newsvendor buffer in S$ from a forecast quantile of the % rise."""
    return np.maximum(0, q_pct) * premium_now


def coverage(actual_sgd, buffer):
    """Share of sales where the buffer covered the rise (actual <= buffer)."""
    return float(np.mean(np.asarray(actual_sgd) <= np.asarray(buffer)))


def implied_cost_ratios(base: pd.DataFrame) -> pd.DataFrame:
    """Back out the tau (and k) that the no-buffer and fixed-5% rules imply:
    the coverage a rule achieves is the critical ratio it behaves as if it used."""
    rows = []
    groups = [("All", base)] + list(base.groupby("category"))
    for cat, g in groups:
        for rule, b in [("R0 No buffer", np.zeros(len(g))),
                        ("R1 Fixed 5%", 0.05 * g["premium_now"].to_numpy())]:
            tau = coverage(g["actual_sgd"], b)
            rows.append({"category": cat, "rule": rule, "implied_tau": tau,
                         "implied_k": tau_to_k(tau)})
    return pd.DataFrame(rows)


SIGNAL_LABELS = {"vol_12": "recent volatility (std. dev. of 1-round changes, last 12 rounds)",
                 "bid_ratio": "bid ratio (bids / quota)"}


def condition_cutoffs(feat: pd.DataFrame, signal: str = "vol_12") -> dict:
    """Market condition = tercile of `signal` per category: recent volatility (headline)
    or the bid ratio (bids / quota)."""
    return {c: tuple(g[signal].quantile([1 / 3, 2 / 3]))
            for c, g in feat.groupby("category")}


def assign_condition(df: pd.DataFrame, cuts: dict, signal: str = "vol_12") -> pd.Series:
    lo = df["category"].map(lambda c: cuts[c][0])
    hi = df["category"].map(lambda c: cuts[c][1])
    return pd.Series(np.select([df[signal] <= lo, df[signal] <= hi],
                               ["Low", "Medium"], "High"), index=df.index)


def buffer_table(fc: pd.DataFrame, cuts: dict, latest_premium: dict,
                 signal: str = "vol_12") -> pd.DataFrame:
    """Dealer buffer table: category x market condition x tau. `fc` must carry the
    `signal` column (vol_12 or bid_ratio) at the quote round.

    buffer_pct is the median newsvendor buffer (as % of the premium at quote time) over
    all backtest rounds in that cell; buffer_sgd applies it to the latest premium.
    underwater_pct / avg_shortfall come from charging the actual per-round buffers.
    """
    d = fc.dropna(subset=["actual_sgd"]).copy()
    d["condition"] = assign_condition(d, cuts, signal)
    d["buffer"] = buffer_sgd(d["q_pct"], d["premium_now"])
    d["buffer_pct_row"] = np.maximum(0, d["q_pct"]) * 100
    d["short"] = (d["actual_sgd"] - d["buffer"]).clip(lower=0)
    rows = []
    for (cat, cond, tau), g in d.groupby(["category", "condition", "tau"]):
        pct = g["buffer_pct_row"].median()
        s = g["short"]
        rows.append({
            "category": cat, "condition": cond, "tau": tau, "k": round(tau_to_k(tau), 1),
            "signal": signal, "signal_range": _range_label(cond, cuts[cat], signal),
            "buffer_pct": pct, "buffer_sgd": pct / 100 * latest_premium[cat],
            "underwater_pct": (s > 0).mean() * 100,
            "avg_shortfall": s[s > 0].mean() if (s > 0).any() else 0.0,
            "n_rounds": len(g),
        })
    out = pd.DataFrame(rows)
    order = {"Low": 0, "Medium": 1, "High": 2}
    return out.sort_values(["category", "condition", "tau"],
                           key=lambda s: s.map(order) if s.name == "condition" else s
                           ).reset_index(drop=True)


def _range_label(cond, cut, signal="vol_12"):
    fmt = (lambda x: f"{x * 100:.1f}%") if signal == "vol_12" else (lambda x: f"{x:.2f}")
    lo, hi = map(fmt, cut)
    return {"Low": f"<= {lo}", "Medium": f"{lo}-{hi}", "High": f"> {hi}"}[cond]
