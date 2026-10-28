"""Download, load and clean the LTA COE bidding results."""
import time

import numpy as np
import pandas as pd
import requests

from .config import (API_URL, CATEGORIES, CUTOFF_MONTH, CUTOFF_ROUND,
                     POST_2022_FROM, RAW_COE, RESOURCE_ID)

NUM_COLS = ["quota", "bids_success", "bids_received", "premium"]


def download_coe(path=RAW_COE, force=False, pause=2.0):
    """Download the full dataset from the data.gov.sg API into data/raw/.

    The snapshot is frozen: if the file exists it is kept unless force=True.
    """
    if path.exists() and not force:
        print(f"Using frozen snapshot {path.name} (pass force=True to re-download)")
        return pd.read_csv(path, dtype=str)
    rows, offset = [], 0
    while True:
        r = requests.get(API_URL, params={"resource_id": RESOURCE_ID,
                                          "limit": 1000, "offset": offset}, timeout=30)
        r.raise_for_status()
        res = r.json()["result"]
        rows += res["records"]
        offset += 1000
        if offset >= res["total"]:
            break
        time.sleep(pause)  # stay under the API rate limit
    df = pd.DataFrame(rows).drop(columns="_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_raw(path=RAW_COE):
    """Read the raw snapshot and convert text columns to numbers."""
    df = pd.read_csv(path, dtype=str)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c].str.replace(",", "", regex=False))
    df["bidding_no"] = df["bidding_no"].astype(int)
    df["category"] = df["vehicle_class"].str[-1]
    return df


def quota_quarter(month: pd.Series) -> pd.Series:
    """LTA quota quarters run Feb-Apr, May-Jul, Aug-Oct, Nov-Jan.

    Shift each month back by one so that Feb->Q1, May->Q2, Aug->Q3, Nov->Q4,
    and January belongs to the previous year's Nov-Jan quarter.
    """
    p = pd.PeriodIndex(month, freq="M") - 1
    return pd.Series(p.year * 10 + p.quarter, index=month.index)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per car category and round, numbered t = 0, 1, 2, ... in time order."""
    df = raw[raw["category"].isin(CATEGORIES)].copy()
    cut = (df["month"] < CUTOFF_MONTH) | ((df["month"] == CUTOFF_MONTH) &
                                          (df["bidding_no"] <= CUTOFF_ROUND))
    df = df[cut]
    df = df.sort_values(["category", "month", "bidding_no"]).reset_index(drop=True)
    df["t"] = df.groupby("category").cumcount()
    # global round index, identical across categories (all have the same rounds)
    rounds = df[["month", "bidding_no"]].drop_duplicates().sort_values(["month", "bidding_no"])
    rounds["round_id"] = np.arange(len(rounds))
    df = df.merge(rounds, on=["month", "bidding_no"], how="left")

    # approximate close date: round 1 in the first week, round 2 in the third week
    month_start = pd.to_datetime(df["month"] + "-01")
    df["date"] = month_start + pd.to_timedelta(np.where(df["bidding_no"] == 1, 6, 20), unit="D")
    df["month_num"] = month_start.dt.month
    df["year"] = month_start.dt.year
    df["quota_q"] = quota_quarter(df["month"])

    # suspension: flag the first round after a gap of more than one month
    month_idx = df["year"] * 12 + df["month_num"]
    gap = month_idx - month_idx.groupby(df["category"]).shift()
    df["after_gap"] = (gap > 1).astype(int)
    df["post_2022"] = (df["month"] >= POST_2022_FROM).astype(int)
    return df.sort_values(["category", "t"]).reset_index(drop=True)


def check_raw(raw: pd.DataFrame) -> dict:
    """Phase 1 checks: equal rows per category, no duplicates, only Apr-Jun 2020 missing."""
    counts = raw["vehicle_class"].value_counts()
    dups = raw.duplicated(["month", "bidding_no", "vehicle_class"]).sum()
    have = set(raw["month"].unique())
    all_months = pd.period_range(raw["month"].min(), raw["month"].max(), freq="M").astype(str)
    missing = sorted(set(all_months) - have)
    rounds = raw.groupby("month")["bidding_no"].nunique()
    return {
        "rows": len(raw),
        "rows_per_category": counts.to_dict(),
        "equal_rows": counts.nunique() == 1,
        "duplicates": int(dups),
        "missing_months": missing,
        "months_without_2_rounds": rounds[rounds != 2].index.tolist(),
        "first_month": raw["month"].min(),
        "last_month": raw["month"].max(),
    }
