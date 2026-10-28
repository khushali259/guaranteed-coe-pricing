"""Forecast scoring: pinball loss and calibration (coverage)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def pinball(y, q, tau):
    """Pinball (quantile) loss: tau*(y-q) if y >= q else (1-tau)*(q-y).

    This is the newsvendor cost with c_u = tau and c_o = 1 - tau, i.e. the newsvendor
    cost scaled by 1/(c_u + c_o), so a lower pinball loss means a cheaper buffer.
    """
    d = np.asarray(y) - np.asarray(q)
    return np.where(d >= 0, tau * d, (tau - 1) * d)


def evaluate(fc: pd.DataFrame, by=("model", "h", "tau")) -> pd.DataFrame:
    """Mean pinball loss (in % points and S$) and coverage = share of actuals <= quantile."""
    fc = fc.dropna(subset=["actual_pct", "q_pct"]).copy()
    fc["pinball_pct"] = pinball(fc["actual_pct"], fc["q_pct"], fc["tau"]) * 100
    fc["pinball_sgd"] = pinball(fc["actual_sgd"], fc["q_sgd"], fc["tau"])
    fc["covered"] = (fc["actual_pct"] <= fc["q_pct"]).astype(float)
    out = fc.groupby(list(by)).agg(n=("covered", "size"),
                                   pinball_pct=("pinball_pct", "mean"),
                                   pinball_sgd=("pinball_sgd", "mean"),
                                   coverage=("covered", "mean")).reset_index()
    if "tau" in by:
        out["coverage_gap"] = out["coverage"] - out["tau"]
    return out


def round_losses(fc: pd.DataFrame, model: str, h: int, taus=None) -> pd.Series:
    """Mean pinball loss (S$) per quote round for one model, averaged over categories and tau."""
    d = fc[(fc["model"] == model) & (fc["h"] == h)].dropna(subset=["actual_sgd", "q_sgd"])
    if taus is not None:
        d = d[d["tau"].isin(taus)]
    loss = pinball(d["actual_sgd"], d["q_sgd"], d["tau"])
    return pd.Series(loss, index=d.index).groupby(d["round_id"]).mean()


def diebold_mariano(fc: pd.DataFrame, model_a: str, model_b: str, h: int, taus=None,
                    rounds=None) -> dict:
    """Diebold-Mariano test of equal forecast loss: d_t = loss_a - loss_b per quote round.
    The mean of d_t is tested against 0 with HAC (Newey-West) standard errors, maxlags = h,
    because h-round targets from neighbouring rounds overlap. Negative = A is better."""
    la, lb = round_losses(fc, model_a, h, taus), round_losses(fc, model_b, h, taus)
    d = (la - lb).dropna()
    if rounds is not None:
        d = d[d.index.isin(rounds)]
    res = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": h})
    return {"model_a": model_a, "model_b": model_b, "n_rounds": len(d),
            "loss_a": la.reindex(d.index).mean(), "loss_b": lb.reindex(d.index).mean(),
            "mean_diff": float(res.params[0]), "pct_diff": float(res.params[0] / lb.reindex(d.index).mean() * 100),
            "dm_stat": float(res.tvalues[0]), "p_value": float(res.pvalues[0])}
