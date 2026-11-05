# How much buffer should a car dealer build into a guaranteed-COE package?

IS5126 Group 4. We forecast how far the COE premium can rise between a dealer's quote and the purchase two rounds later, turn the forecast distribution into a buffer with the **newsvendor model** (b* = the c_u/(c_u+c_o) quantile of the forecast), and backtest that buffer against what dealers do today (no buffer, fixed 5%) on Jan 2014 – Oct 2026 data.

## Data and cut-off

| Source | File |
|---|---|
| LTA *COE Bidding Results / Prices*, data.gov.sg (resource `d_69b3380ad7e51aff3a7dcc84eba52b8a`) | `data/raw/coe_bidding_results.csv` |
| Policy events from LTA / MOT / MAS pages, entered by hand with source links | `data/raw/policy_events.csv` |

**Frozen cut-off: 2nd bidding round of October 2026.** The snapshot has 1,990 rows (398 rounds × 5 categories, Jan 2010 – Oct 2026; only Apr–Jun 2020 is missing, the COVID-19 suspension). Do not re-download; `01_download.ipynb` keeps the existing file unless `FORCE = True`.

## How to run

```bash
python3.11 -m venv .venv && source .venv/bin/activate     # or use conda
pip install -r requirements.txt
cd notebooks
for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb" --ExecutePreprocessor.timeout=1800; done
```

Run the notebooks in order; each reads only the files written by the ones before it. Total run time is about 11–12 minutes on a laptop (05 and 08 take about 4 and 6 minutes). `data/processed/forecasts.csv` is about 21 MB. Results are deterministic: no random train/test splits anywhere, and fixed random seeds for SARIMAX simulation, gradient boosting and the bootstrap.

Optional demo: `pip install streamlit && streamlit run app.py` (from the project root).

## Layout

```
data/raw/            untouched downloads + policy_events.csv
data/processed/      everything the notebooks write (features, forecasts, tables)
notebooks/           01_download … 09_risk_diagnostics, run in order
src/                 shared code: config, data, features, models, metrics, newsvendor, backtest, plots
figures/             charts for the report
report/              report_draft.md/.pdf (4-page report), report_full.md (all results), slides outline, EDA findings
app.py               Streamlit buffer calculator (optional)
```

| Notebook | Phase | Main outputs |
|---|---|---|
| `01_download` | 1 Data | raw snapshot + checks (equal rows, no duplicates, only Apr–Jun 2020 missing) |
| `02_clean_features` | 2 Cleaning & features | `coe_features.csv`; target spot check; hand recomputation; automated look-ahead test |
| `03_eda` | 3 EDA | 6 charts, ADF tests, rise summary stats, `report/eda_findings.md` |
| `04_regression` | 4 Driver regression | HAC OLS, VIF, post-2022 stability, per-category fits, residuals, `regression_effects.csv` |
| `05_forecast` | 5 Forecast distribution | `forecasts.csv` (M0–M7, h = 1–4, 6 quantiles), pinball/calibration tables, model choice on 2014–19 (`best_model.json`), Diebold–Mariano tests (`dm_tests.csv`) |
| `06_newsvendor` | 6 Newsvendor buffer | `buffer_table.csv` (headline, by volatility), `buffer_table_by_bidratio.csv`, implied cost ratios, today's buffer (`buffer_today.csv`, `buffer_today_by_model.csv`), the one-line formula (`buffer_formula.csv`), c_o scenarios |
| `07_backtest` | 7 Backtest | `backtest_results.csv`, frontier charts, bootstrap CIs (full and test period), best-fixed-% benchmark, sensitivity |
| `08_sensitivity` | Robustness | M1 window, refit frequency, the 16 M6 variants, adaptive calibration on M1/M3/M6 (`sensitivity_*.csv`, `figures/08_*.png`) |
| `09_risk_diagnostics` | Risk diagnostics | cumulative shortfall and cost over time vs fixed 5%; Kupiec coverage and Christoffersen clustering tests for R2, R4, R7 (`kupiec_coverage_tests.csv`, `christoffersen_independence_tests.csv`, `figures/09_*.png`) |

## Conventions and design decisions

- **Time order only.** Rolling-origin evaluation from Jan 2014; at each origin the models see only targets already realised (round ≤ t − h). Refit every 4 rounds.
- **No look-ahead in features.** `02` rebuilds the features on a truncated panel and checks that nothing changes. `quota_change_next` uses a future quota only while that round is inside the quota quarter LTA has already announced (Feb–Apr, May–Jul, Aug–Oct, Nov–Jan). `event_next` uses an event only if it was announced by round t.
- **Suspension.** Any target that spans Apr–Jun 2020 is dropped.
- **Target spot check.** Cat B Jan 2024 round 1 → round 2 gives R = S$26,990 (S$85,010 → S$112,000). That is the 1-round rise (h = 1). The 2-round rise from the same quote is S$17,328.
- **Choosing the buffer model.** The rule was fixed before the results were seen: the lowest mean pinball loss at h = 2 **on the validation period 2014–19**. 2020–26 is an untouched test period. The winner is M6 (volatility-scaled; before M6 and M7 were added it was M1, the 48-round rolling percentile). Diebold–Mariano tests (`dm_tests.csv`) find M1, M3, M4, M5 and M6 statistically tied; SARIMAX (M2) and M7 (M1 + adaptive calibration) are significantly worse. The plan's main rule R4 (M3) and the rolling percentile R2 are still reported throughout.
- **Buffer table condition.** Market condition = tercile of recent volatility (`vol_12`), because the bid ratio barely predicts rises. The bid-ratio version is `buffer_table_by_bidratio.csv`.
- **Policy events** carry a `direction` column (down / up / mixed); the regression and M4 use separate flags for price-lowering and price-raising events.
- **Additions beyond the plan, each flagged in its notebook:**
  - `quota_chg_6`, the quota change already in force;
  - M5, the M1+M3 forecast combination;
  - a family of fixed-% rules (0–20%) and the best fixed % chosen with hindsight, as a tougher benchmark;
  - the buffer table cut by recent volatility (now the headline), Diebold–Mariano tests, and directional event flags;
  - M6, the volatility-scaled model (`vol_ewma` feature, one-line dealer formula buffer = z × volatility), and M7, M1 with adaptive (online) calibration, each with backtest rules R7 / R8;
  - notebook 08 robustness checks.
- **Cut, per the plan's 10-day scope:** the pre-2010 PDF extension (so `pdfplumber` is not needed) and the New Registrations data. M4 (gradient boosting) is included, at h = 2 only, as an appendix result.
