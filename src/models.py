"""Rolling-origin quantile forecasts of the h-round premium rise (Phase 5).

All models forecast the percentage rise r_{t,h} = P_{t+h}/P_t - 1 at quantile levels
TAUS, using only information available at origin t. Models are refit every
REFIT_EVERY rounds; between refits the latest fitted model is reused.

M0 naive (rise = 0), M1 rolling empirical quantiles, M2 SARIMAX, M3 pooled quantile
regression (main), M4 gradient boosting with quantile loss (stretch), M5 = M1+M3 average,
M6 volatility-scaled quantiles, M7 = M1 with adaptive (online) calibration.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
                                             IterationLimitWarning)
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import (ACI_ETAS, REFIT_EVERY, ROLL_WINDOW, TAUS, TEST_START, VALID_PERIODS,
                     period_of)
from .metrics import evaluate

BASE_COLS = ["category", "t", "round_id", "month", "bidding_no", "premium", "bid_ratio"]


def _to_long(base: pd.DataFrame, h: int, model: str, q: np.ndarray, taus) -> pd.DataFrame:
    """Stack an (n_rows x n_taus) matrix of quantile forecasts into the long format."""
    out = []
    for j, tau in enumerate(taus):
        d = base[BASE_COLS].copy()
        d["h"], d["model"], d["tau"] = h, model, tau
        d["q_pct"] = q[:, j]
        d["actual_pct"] = base[f"r_{h}"].to_numpy()
        out.append(d)
    out = pd.concat(out, ignore_index=True)
    out["q_sgd"] = out["q_pct"] * out["premium"]
    out["actual_sgd"] = out["actual_pct"] * out["premium"]
    out["period"] = out["month"].map(period_of)
    return out.rename(columns={"premium": "premium_now"})


def _origins(feat, start=TEST_START):
    return feat[feat["month"] >= start].sort_values(["round_id", "category"])


# ----------------------------------------------------------------------------- M0
def m0_naive(feat, h, taus=TAUS):
    base = _origins(feat)
    return _to_long(base, h, "M0", np.zeros((len(base), len(taus))), taus)


# ----------------------------------------------------------------------------- M1
def m1_rolling(feat, h, taus=TAUS, window=ROLL_WINDOW):
    """tau-quantile of the last `window` h-round rises that were already realised at t."""
    parts = []
    for _, g in feat.groupby("category"):
        g = g.sort_values("t").copy()
        realised = g[f"r_{h}"].shift(h)          # r_{t-h,h} = P_t / P_{t-h} - 1, known at t
        for tau in taus:
            g[f"q{tau}"] = realised.rolling(window, min_periods=24).quantile(tau)
        parts.append(g)
    base = _origins(pd.concat(parts))
    return _to_long(base, h, "M1", base[[f"q{tau}" for tau in taus]].to_numpy(), taus)


# ----------------------------------------------------------------------------- M3
def qr_formula(h, train=None, min_post=24):
    """Phase 4/5 specification. post_2022 is dropped while the training window
    has too few post-May-2022 rows to estimate it."""
    terms = ["bid_ratio", f"quota_change_next_{h}", "mom_2", "vol_12", "post_2022",
             "C(category)"]
    if train is not None and train["post_2022"].sum() < min_post:
        terms.remove("post_2022")
    return f"r_{h} ~ " + " + ".join(terms)


def _rolling_blocks(feat, h, refit_every, start):
    """Yield (train, test) pairs: train holds rows whose target is realised by the
    refit origin r0 (round_id <= r0 - h); test holds origins r0 .. r0+refit_every-1."""
    base = _origins(feat, start)
    rounds = np.sort(base["round_id"].unique())
    for r0 in rounds[::refit_every]:
        train = feat[(feat["round_id"] <= r0 - h)].dropna(subset=[f"r_{h}"])
        test = base[(base["round_id"] >= r0) & (base["round_id"] < r0 + refit_every)]
        yield train, test


def m3_quantreg(feat, h, taus=TAUS, refit_every=REFIT_EVERY, start=TEST_START):
    preds = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IterationLimitWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        for train, test in _rolling_blocks(feat, h, refit_every, start):
            f = qr_formula(h, train)
            cols = ["bid_ratio", f"quota_change_next_{h}", "mom_2", "vol_12"]
            train = train.dropna(subset=cols)
            q = np.column_stack([smf.quantreg(f, train).fit(q=tau, max_iter=5000)
                                 .predict(test).to_numpy() for tau in taus])
            q.sort(axis=1)                       # fix any quantile crossing per row
            preds.append(_to_long(test, h, "M3", q, taus))
    return pd.concat(preds, ignore_index=True)


# ----------------------------------------------------------------------------- M4
def gbm_features(h):
    return ["bid_ratio", "bid_ratio_lag1", "unsuccessful_share", f"quota_change_next_{h}",
            "quota_chg_6", "mom_1", "mom_2", "mom_4", "vol_12", "e_minus_b", "round_no",
            "post_2022", f"event_down_next_{h}", f"event_up_next_{h}", "is_A", "is_B"]


def m4_gbm(feat, h, taus=TAUS, refit_every=REFIT_EVERY, start=TEST_START):
    feat = feat.assign(is_A=(feat["category"] == "A").astype(int),
                       is_B=(feat["category"] == "B").astype(int))
    X = gbm_features(h)
    preds = []
    for train, test in _rolling_blocks(feat, h, refit_every, start):
        train = train.dropna(subset=X)
        q = np.column_stack([
            GradientBoostingRegressor(loss="quantile", alpha=tau, n_estimators=150,
                                      max_depth=2, learning_rate=0.05, min_samples_leaf=20,
                                      subsample=0.8, random_state=0)
            .fit(train[X], train[f"r_{h}"]).predict(test[X]) for tau in taus])
        q.sort(axis=1)
        preds.append(_to_long(test, h, "M4", q, taus))
    return pd.concat(preds, ignore_index=True)


# ----------------------------------------------------------------------------- M2
def _sarimax_inputs(g):
    """y_t = 100 * dlog P_t; exog = bid ratio at t-1 and 100 * dlog quota_t."""
    y = 100 * np.log(g["premium"]).diff()
    y[g["after_gap"] == 1] = np.nan          # the 4-month jump across the suspension
    dq = 100 * np.log(g["quota"]).diff()
    dq[g["after_gap"] == 1] = 0
    exog = pd.DataFrame({"bid_lag": g["bid_ratio"].shift(1).bfill(),
                         "dlog_quota": dq.fillna(0)})
    return y.to_numpy(), exog.to_numpy()


def select_arma_order(y, exog, max_p=2, max_q=2):
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
            try:
                res = SARIMAX(y, exog, order=(p, 0, q), trend="c").fit(disp=False)
            except Exception:
                continue
            if best is None or res.aic < best[0]:
                best = (res.aic, (p, 0, q))
    return best[1]


def m2_sarimax(feat, horizons, taus=TAUS, refit_every=REFIT_EVERY, start=TEST_START,
               n_sims=2000, seed=0):
    """ARMA on log-premium changes with exogenous bid ratio and quota change, per
    category. Forecast quantiles of the cumulative h-round rise come from simulating
    2,000 future paths. Future exog: the bid ratio is held at its value at t; the quota
    change is used only while the round lies in the announced quota quarter, else 0."""
    rng = np.random.default_rng(seed)
    hmax = max(horizons)
    out, orders = [], {}
    for cat, g in feat.groupby("category"):
        g = g.sort_values("t").reset_index(drop=True)
        y, exog = _sarimax_inputs(g)
        pre = (g["month"] < start).to_numpy()
        order = select_arma_order(y[1:pre.sum()], exog[1:pre.sum()])
        orders[cat] = order
        positions = np.where(~pre)[0]
        params, qs = None, {h: [] for h in horizons}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, pos in enumerate(positions):
                mod = SARIMAX(y[1:pos + 1], exog[1:pos + 1], order=order, trend="c")
                if i % refit_every == 0:
                    res = mod.fit(disp=False, start_params=params)
                    params = res.params
                else:
                    res = mod.filter(params)
                fx = np.zeros((hmax, 2))
                fx[:, 0] = g.loc[pos, "bid_ratio"]
                for s in range(1, hmax + 1):
                    nxt = pos + s
                    if nxt < len(g) and g.loc[nxt, "quota_q"] == g.loc[pos, "quota_q"]:
                        fx[s - 1, 1] = exog[nxt, 1]
                sims = np.asarray(res.simulate(hmax, repetitions=n_sims, anchor="end",
                                               exog=fx, random_state=rng)).reshape(hmax, -1)
                cum = np.exp(np.cumsum(sims, axis=0) / 100) - 1
                for h in horizons:
                    qs[h].append(np.quantile(cum[h - 1], taus))
        base = g.iloc[positions]
        for h in horizons:
            out.append(_to_long(base, h, "M2", np.vstack(qs[h]), taus))
    return pd.concat(out, ignore_index=True), orders


# ----------------------------------------------------------------------------- M5
def m5_combo(fc):
    """Equal-weight average of the M1 and M3 quantiles (simple forecast combination)."""
    keys = ["category", "t", "h", "tau"]
    m1 = fc[fc["model"] == "M1"].set_index(keys)
    m3 = fc[fc["model"] == "M3"].set_index(keys)
    out = m3.copy()
    out["q_pct"] = (m1["q_pct"].reindex(m3.index) + m3["q_pct"]) / 2
    out["q_sgd"] = out["q_pct"] * out["premium_now"]
    out["model"] = "M5"
    return out.dropna(subset=["q_pct"]).reset_index()


# ----------------------------------------------------------------------------- M6
def m6_volscaled(feat, h, taus=TAUS, vol="vol_ewma", beta=1.0, centre=False, window=None,
                 start=TEST_START):
    """Volatility-scaled quantiles: forecast q_tau = s_t * z_tau, with s_t = vol_t ** beta.

    z_tau is the tau-quantile of the standardised rises r_{s,h} / s_s over every round s
    whose target is realised at t (s <= t - h), pooled across categories (expanding
    window, or the last `window` rounds). The dealer version is one line:
    buffer % = z_tau x recent volatility. centre=True standardises around the rolling
    48-round median rise instead of 0 (a location-scale variant, tried in 08)."""
    d = feat.sort_values(["round_id", "category"]).copy()
    d["s"] = d[vol] ** beta
    d["m"] = 0.0
    if centre:
        realised = d.groupby("category")[f"r_{h}"].shift(h)
        d["m"] = realised.groupby(d["category"]).transform(
            lambda x: x.rolling(ROLL_WINDOW, min_periods=24).median())
    d["z"] = (d[f"r_{h}"] - d["m"]) / d["s"]
    base = _origins(d, start)
    zq = {}
    for r in base["round_id"].unique():
        lo = -np.inf if window is None else r - h - window + 1
        z = d.loc[(d["round_id"] <= r - h) & (d["round_id"] >= lo), "z"]
        zq[r] = np.nanquantile(z, taus)
    q = (np.vstack([zq[r] for r in base["round_id"]]) * base[["s"]].to_numpy()
         + base[["m"]].to_numpy())
    return _to_long(base, h, "M6", q, taus)


def z_multipliers(feat, h, taus=TAUS, vol="vol_ewma"):
    """The z_tau of M6 using every realised rise in the data (for the dealer's formula)."""
    z = (feat[f"r_{h}"] / feat[vol]).dropna()
    return pd.Series(np.quantile(z, taus), index=taus, name="z")


# ----------------------------------------------------------------------------- M7
def adaptive_calibration(f, eta, name=None):
    """Online quantile tracking (adaptive conformal calibration) on top of any model.

    For each category and tau, the forecast is shifted by theta_t (in % points of the
    premium). Each time a past forecast is resolved, theta moves up by eta * tau if the
    rise beat the quantile (a miss) and down by eta * (1 - tau) if it did not, so the
    long-run miss rate is pushed to 1 - tau. The outcome of origin t - h is only known at
    t, so the update is delayed by h rounds. eta = 0 returns the base model unchanged."""
    out = []
    for (_, h, _), g in f.groupby(["category", "h", "tau"]):
        g = g.sort_values("t").copy()
        tau = g["tau"].iloc[0]
        q0, y = g["q_pct"].to_numpy(), g["actual_pct"].to_numpy()
        q, theta = q0.copy(), 0.0
        for i in range(len(g)):
            s = i - h
            if s >= 0 and not np.isnan(y[s]):
                theta += eta * (float(y[s] > q[s]) - (1 - tau))
            q[i] = q0[i] + theta
        g["q_pct"] = q
        out.append(g)
    out = pd.concat(out)
    out["q_sgd"] = out["q_pct"] * out["premium_now"]
    if name:
        out["model"] = name
    return out


def choose_eta(f, etas=ACI_ETAS, periods=VALID_PERIODS):
    """Step size with the lowest mean pinball loss (S$) on the validation period."""
    scores = {}
    for eta in etas:
        a = adaptive_calibration(f, eta)
        e = evaluate(a[a["period"].isin(periods)], by=("tau",))
        scores[eta] = e["pinball_sgd"].mean()
    return min(scores, key=scores.get), scores


def run_all(feat, horizons, taus=TAUS, gbm_horizons=(2,), verbose=True, h_select=2):
    """Run every model and return (forecasts, sarimax_orders, extras)."""
    parts = []
    for h in horizons:
        for name, fn in [("M0", m0_naive), ("M1", m1_rolling), ("M3", m3_quantreg)]:
            if verbose:
                print(f"h={h} {name}")
            parts.append(fn(feat, h, taus))
        if h in gbm_horizons:
            if verbose:
                print(f"h={h} M4")
            parts.append(m4_gbm(feat, h, taus))
    if verbose:
        print("M2 SARIMAX (all horizons)")
    m2, orders = m2_sarimax(feat, horizons, taus)
    parts.append(m2)
    for h in horizons:
        if verbose:
            print(f"h={h} M6")
        parts.append(m6_volscaled(feat, h, taus))
    fc = pd.concat(parts, ignore_index=True)
    fc = pd.concat([fc, m5_combo(fc)], ignore_index=True)
    # M7: one step size for all horizons, chosen on the validation period at h_select
    eta, eta_scores = choose_eta(fc[(fc["model"] == "M1") & (fc["h"] == h_select)])
    if verbose:
        print(f"M7 adaptive calibration of M1, eta = {eta}")
    fc = pd.concat([fc, adaptive_calibration(fc[fc["model"] == "M1"], eta, "M7")],
                   ignore_index=True)
    extras = {"m7_eta": eta, "m7_eta_validation_pinball": eta_scores}
    return (fc.sort_values(["model", "h", "category", "t", "tau"]).reset_index(drop=True),
            orders, extras)
