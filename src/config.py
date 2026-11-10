"""Project-wide constants: paths, categories, horizons, quantile levels, periods."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
REPORT = PROC                         # text summaries (EDA findings, buffer guidance) sit with the other outputs

# data.gov.sg dataset: LTA "COE Bidding Results / Prices"
API_URL = "https://data.gov.sg/api/action/datastore_search"
RESOURCE_ID = "d_69b3380ad7e51aff3a7dcc84eba52b8a"
RAW_COE = RAW / "coe_bidding_results.csv"
POLICY_EVENTS = RAW / "policy_events.csv"

# Frozen cut-off: the snapshot ends with the 2nd round of October 2026.
# Never re-download after this; all results use the same data.
CUTOFF_MONTH, CUTOFF_ROUND = "2026-10", 2

CATEGORIES = ["A", "B", "E"]          # car categories only
HORIZONS = [1, 2, 3, 4]               # rounds between quote and purchase
H = 2                                 # headline horizon (about one month)
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
K_COST = [1, 2.3, 4, 9]               # cost ratios c_u / c_o used in the backtest

TEST_START = "2014-01"                # first forecast origin (train window 2010-2013)
ROLL_WINDOW = 48                      # M1 look-back (rounds)
REFIT_EVERY = 4                       # rounds between model refits
POST_2022_FROM = "2022-05"            # May 2022 category redefinition
EWMA_LAMBDA = 0.94                    # decay of the EWMA volatility used by M6 (RiskMetrics value)
ACI_ETAS = [0.0025, 0.005, 0.01, 0.02]  # step sizes tried for adaptive calibration (M7), chosen on 2014-19

# Model selection uses the validation period only; 2020-2026 is the untouched test period.
VALID_PERIODS = ["2014-19"]
TEST_PERIODS = ["2020-22", "2023-26"]

PERIODS = {
    "2014-19": ("2014-01", "2019-12"),
    "2020-22": ("2020-01", "2022-12"),
    "2023-26": ("2023-01", "2026-12"),
}


def period_of(month: str) -> str:
    for name, (lo, hi) in PERIODS.items():
        if lo <= month <= hi:
            return name
    return "pre-2014"
