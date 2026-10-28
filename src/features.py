"""Target and feature construction. Every feature at round t uses only rounds <= t
(plus quota already announced by LTA for the current quota quarter)."""
import numpy as np
import pandas as pd

from .config import HORIZONS, POLICY_EVENTS

FEATURES = ["bid_ratio", "bid_ratio_lag1", "unsuccessful_share", "mom_1", "mom_2",
            "mom_4", "vol_12", "quota_chg_6", "e_minus_b", "round_no", "month_num",
            "post_2022"]


def load_policy_events(path=POLICY_EVENTS) -> pd.DataFrame:
    ev = pd.read_csv(path, parse_dates=["date", "announced"])
    return ev.sort_values("date").reset_index(drop=True)


def _event_flags(dates_t, dates_end, category, events, direction=None):
    """1 if an event already announced at t takes effect in (t, t+h].
    direction: None = any event; "down" / "up" = only events expected to lower / raise
    the premium once they take effect (column `direction` in policy_events.csv)."""
    out = np.zeros(len(dates_t), dtype=int)
    for _, e in events.iterrows():
        applies = (e["categories"] == "ALL") or (category in e["categories"].split(";"))
        if not applies or (direction is not None and e["direction"] != direction):
            continue
        hit = (e["announced"] <= dates_t) & (e["date"] > dates_t) & (e["date"] <= dates_end)
        out |= hit.to_numpy().astype(int)
    return out


def build_features(panel: pd.DataFrame, events: pd.DataFrame | None = None,
                   horizons=HORIZONS) -> pd.DataFrame:
    """Return one row per category x round with targets r_h, R_h and all features."""
    if events is None:
        events = load_policy_events()
    # Cat E vs Cat B premium gap, the same value for every category in a round
    piv = panel.pivot_table(index="round_id", columns="category", values="premium")
    e_minus_b = ((piv["E"] - piv["B"]) / piv["B"]).rename("e_minus_b")

    parts = []
    for cat, g in panel.groupby("category", sort=True):
        g = g.sort_values("t").copy()
        P, Q = g["premium"], g["quota"]
        g["bid_ratio"] = g["bids_received"] / Q
        g["bid_ratio_lag1"] = g["bid_ratio"].shift(1)
        g["unsuccessful_share"] = (g["bids_received"] - g["bids_success"]) / g["bids_received"]
        g["ret_1"] = P.pct_change()
        for k in (1, 2, 4):
            g[f"mom_{k}"] = P / P.shift(k) - 1
        g["vol_12"] = g["ret_1"].rolling(12).std()
        g["quota_chg_6"] = Q / Q.shift(6) - 1          # change vs one quarter (6 rounds) ago
        g["log_premium"] = np.log(P)
        g["round_no"] = g["bidding_no"]

        last_q_in_quarter = g.groupby("quota_q")["quota"].transform("last")
        for h in horizons:
            fut = P.shift(-h)
            # a target spans the suspension if any of rounds t+1..t+h follows the gap
            spans = sum(g["after_gap"].shift(-j).fillna(0) for j in range(1, h + 1)) > 0
            g[f"r_{h}"] = (fut / P - 1).where(~spans)
            g[f"R_{h}"] = (fut - P).where(~spans)
            # quota change to t+h, known only while t+h is in the announced quarter;
            # otherwise use the last announced quota (end of the current quarter)
            same_q = g["quota_q"].shift(-h) == g["quota_q"]
            g[f"quota_change_next_{h}"] = np.where(same_q, Q.shift(-h) / Q - 1,
                                                   last_q_in_quarter / Q - 1)
            end_date = g["date"].shift(-h).fillna(g["date"] + pd.Timedelta(days=14 * h))
            g[f"event_next_{h}"] = _event_flags(g["date"], end_date, cat, events)
            g[f"event_down_next_{h}"] = _event_flags(g["date"], end_date, cat, events, "down")
            g[f"event_up_next_{h}"] = _event_flags(g["date"], end_date, cat, events, "up")
        parts.append(g)

    df = pd.concat(parts).merge(e_minus_b, left_on="round_id", right_index=True, how="left")
    df["quota_change_next"] = df["quota_change_next_2"]   # headline-horizon aliases
    df["event_next"] = df["event_next_2"]
    df["event_down_next"] = df["event_down_next_2"]
    df["event_up_next"] = df["event_up_next_2"]
    return df.sort_values(["category", "t"]).reset_index(drop=True)
