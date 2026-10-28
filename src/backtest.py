"""Backtest of dealer pricing rules (Phase 7).

A simulated dealer sells one guaranteed car per category in every round from Jan 2014,
quotes the latest premium plus a buffer b_t, and secures the COE h rounds later at
P_{t+h}. Base case: the dealer keeps any gain if the premium falls.
"""
import numpy as np
import pandas as pd

from .config import K_COST
from .newsvendor import buffer_sgd, k_to_tau

RULE_MODELS = {
    "R2 Rolling percentile": "M1",
    "R3 Newsvendor SARIMAX": "M2",
    "R4 Newsvendor QR": "M3",
}
FIXED_RULES = ["R0 No buffer", "R1 Fixed 5%"]


def rule_buffers(fc: pd.DataFrame, h: int, extra: dict | None = None) -> pd.DataFrame:
    """Long table: one row per category x round x rule x tau with buffer and actual rise."""
    rule_models = {**RULE_MODELS, **(extra or {})}
    d = fc[(fc["h"] == h)].dropna(subset=["actual_sgd"])
    keep = ["category", "t", "round_id", "month", "bidding_no", "period", "premium_now",
            "bid_ratio", "actual_sgd", "actual_pct"]
    base = d[(d["model"] == "M0") & (d["tau"] == 0.5)][keep]
    parts = [base.assign(rule="R0 No buffer", tau=np.nan, buffer=0.0),
             base.assign(rule="R1 Fixed 5%", tau=np.nan, buffer=0.05 * base["premium_now"])]
    for rule, model in rule_models.items():
        m = d[d["model"] == model]
        parts.append(m[keep + ["tau"]].assign(
            rule=rule, buffer=buffer_sgd(m["q_pct"], m["premium_now"])))
    return pd.concat(parts, ignore_index=True)


def score(df: pd.DataFrame, ks=K_COST) -> dict:
    """Plan metrics, plus the newsvendor cost for each k (c_u = 1, c_o = 1/k)
    and the buyer's net cost in the refund case (unused buffer returned)."""
    short = (df["actual_sgd"] - df["buffer"]).clip(lower=0)
    excess = (df["buffer"] - df["actual_sgd"]).clip(lower=0)
    out = {
        "n": len(df),
        "underwater_pct": (short > 0).mean() * 100,
        "avg_shortfall": short[short > 0].mean() if (short > 0).any() else 0.0,
        "worst_shortfall": short.max(),
        "avg_buffer": df["buffer"].mean(),
        "avg_buffer_pct": (df["buffer"] / df["premium_now"]).mean() * 100,
        "avg_loss_per_car": short.mean(),
        "avg_buyer_cost_refund": np.minimum(df["buffer"], df["actual_sgd"].clip(lower=0)).mean(),
    }
    for k in ks:
        out[f"cost_k{k}"] = (short + excess / k).mean()
    return out


def score_table(bt: pd.DataFrame, by=("rule", "tau")) -> pd.DataFrame:
    rows = []
    for key, g in bt.groupby(list(by), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **score(g)})
    return pd.DataFrame(rows)


def matched_comparison(bt: pd.DataFrame, rule: str, ks=K_COST, against="R1 Fixed 5%",
                       taus_available=None) -> pd.DataFrame:
    """Newsvendor cost per car of `rule` at tau* = k/(1+k) vs a fixed rule, for each k."""
    taus_available = taus_available or sorted(bt["tau"].dropna().unique())
    rows = []
    for k in ks:
        tau = min(taus_available, key=lambda x: abs(x - k_to_tau(k)))
        a = bt[(bt["rule"] == rule) & (bt["tau"] == tau)]
        b = bt[bt["rule"] == against]
        rows.append({"k": k, "tau_star": round(k_to_tau(k), 3), "tau_used": tau,
                     f"cost_{rule.split()[0]}": score(a)[f"cost_k{k}"],
                     f"cost_{against.split()[0]}": score(b)[f"cost_k{k}"]})
    return pd.DataFrame(rows)


def per_round_cost(bt, rule, tau, k):
    """Average newsvendor cost per car in each round (averaged over categories)."""
    d = bt[(bt["rule"] == rule) & ((bt["tau"] == tau) if tau is not None else True)].copy()
    short = (d["actual_sgd"] - d["buffer"]).clip(lower=0)
    excess = (d["buffer"] - d["actual_sgd"]).clip(lower=0)
    d["cost"] = short + excess / k
    return d.groupby("round_id")["cost"].mean()


def block_bootstrap_diff(bt, rule_a, tau_a, rule_b, tau_b, k, block=8, n_boot=1000, seed=0):
    """Moving-block bootstrap CI for mean cost(rule_a) - cost(rule_b), resampling blocks
    of `block` consecutive rounds so that the overlap of h-round targets is kept."""
    a = per_round_cost(bt, rule_a, tau_a, k)
    b = per_round_cost(bt, rule_b, tau_b, k)
    diff = (a - b).dropna().to_numpy()
    n = len(diff)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"k": k, "mean_diff": diff.mean(), "ci_low": lo, "ci_high": hi,
            "share_boot_below_0": (boots < 0).mean(), "n_rounds": n}
