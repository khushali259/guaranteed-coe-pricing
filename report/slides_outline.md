# Presentation outline: 17 slides, 20 minutes (recorded)

The speaker script for every slide is in `STUDY_GUIDE.md`, Part B. Figures are in `figures/`. Times add up to 19:45, leaving 15 seconds of slack. Suggested presenters follow the project phases; one person can present everything.

| # | Time | Slide title | On the slide | Visual | Presenter |
|---|---|---|---|---|---|
| 1 | 0:15 | **How much buffer should a dealer charge for a guaranteed COE?** | Group 4 · IS5126 · data to October 2026 round 2 | photo or COE bidding screenshot | Report lead |
| 2 | 1:00 | **S$26,990 in one round** | Cat B, January 2024: S$85,010 → S$112,000. A dealer who guaranteed the old price paid the difference. | `03_1_premium_events.png`, Jan 2024 circled | Report lead |
| 3 | 1:15 | **The dealer's problem** | Since 2023, no buffer = underwater on **64%** of Cat A/B sales, **S$4.9k–6.2k** lost each time. Today: no buffer or a flat 5%. **Question:** how big a buffer, so the dealer rarely loses money without overpricing? | 3 big numbers | Report lead |
| 4 | 1:30 | **The newsvendor idea** | buffer = order quantity; premium rise = uncertain demand. c_u = shortfall, c_o = lost sales. **Charge the τ = k/(1+k) percentile of the forecast rise.** k = 4 → τ = 0.8. | mapping table + formula; k → τ strip (1→0.5, 2.3→0.7, 4→0.8, 9→0.9) | Decision lead |
| 5 | 1:15 | **Data, and no peeking at the future** | LTA, 398 rounds, 2010–2026; Cat A/B/E; 14 policy events with announcement dates. Target = % rise over 2 rounds. Rules: announced events only, published quotas only, COVID gap dropped, automated look-ahead test. | timeline graphic; "only what the dealer knew that day" | EDA lead |
| 6 | 1:15 | **What the data says** | Direction is unpredictable (bids vs rise ≈ 0). **Volatility clusters** (calm and stormy spells persist). Dollar risk hasn't fallen: 90th-percentile rise S$8.6k → S$9.2k. | `03_3_volatility.png` + `03_4_bidratio_vs_rise.png` | EDA lead |
| 7 | 1:00 | **What drives a rise?** | R² = **0.018**. Announced COE injections / rebate cuts: −4.3 points (≈ S$5.7k, p = 0.003). Bid ratio: not significant. → Forecast the *size* of swings, not the direction. | small coefficient table | Regression lead |
| 8 | 1:30 | **Forecasting the range, fairly** | 8 models: naive → rolling percentile → SARIMAX → quantile regression → gradient boosting → combo → **volatility-scaled** → adaptive calibration. Rolling origin from 2014, past data only. Score = pinball loss (= newsvendor cost). Choose on **2014–19**, test on **2020–26**. | model ladder graphic + rolling-origin timeline | Forecasting lead |
| 9 | 1:30 | **Simple is enough** | Validation winner: volatility-scaled M6 (S$1,188 vs M1 S$1,208). On test, all good models are **statistically tied** (Diebold–Mariano). SARIMAX 10% worse. Adaptive calibration fixes coverage but **costs 1.4% more** (p = 0.001). | `05_pinball_by_period.png` + `08_calibration_tradeoff.png` (small) | Forecasting lead |
| 10 | 1:00 | **What today's rules secretly assume** | No buffer covers 46% → acts as if **k = 0.84** (overpricing feared more than losses). Fixed 5% covers 72% → **k = 2.6**, but only 59% in 2020–22. | two-row table | Decision lead |
| 11 | 1:30 | **The dealer's buffer table** | Cat B at k = 4: **6.4% calm, 7.8% turbulent**. Cost ratio matters most, volatility next, bid pressure least. **One line: buffer = z × volatility × premium** (z = 1.2 at k = 4). Today (calm): A S$6.1k, B S$7.3k, E S$4.8k. | buffer table (Cat B) + z table | Decision lead |
| 12 | 1:30 | **No free lunch** | Every sensible rule sits on the same protection-vs-cost frontier. To match fixed 5%'s protection (S$3.6k average buffer), the model rules need about the same or a little more (S$3.7k–4.2k). | `07_frontier.png` | Decision lead |
| 13 | 1:15 | **Who gains from the model?** | Cost per car vs fixed 5% (95% bootstrap CI). k = 1: −S$716 · k = 2.3: tie · k = 4: tie · k = 9: −S$229. Holds on the 2020–26 test period. **Fixed 5% is the right rule only for a k ≈ 2–4 dealer.** | bar chart or table by k with CIs | Decision lead |
| 14 | 1:00 | **When it matters: turbulent spells** | 2020–22: fixed 5% underwater on ~40% of sales vs ~22–25% for the rolling percentile. Jan 2024 jump (+S$17.3k): the volatility rule's k = 9 buffer was S$17.4k; fixed 5% charged S$4.3k. | `07_frontier_by_period.png` or `05_fan_catB.png` | Forecasting lead |
| 15 | 1:15 | **Is it robust? Do losses bunch?** | Same result for any window of 36–72 rounds and any refit frequency. Buffers miss ~20% as intended (Kupiec), and misses don't cluster (Christoffersen); only quantile regression clusters, in Cat E. | `09_cumulative_saving_vs_fixed5.png` + `08_sensitivity_window.png` | Risk-diagnostics lead |
| 16 | 0:45 | **Limitations** | Hypothetical one-dealer backtest; c_o unknown; dealers' bids move prices; regime change after 2022; volatility and calibration models designed after first results. | bullets | Report lead |
| 17 | 1:00 | **Recommendations** | 1. Always buffer. 2. Pick k first. 3. Scale with volatility, not bids (z × volatility, or a rolling percentile). 4. Fixed 5% only if k ≈ 2–4. Short demo: `streamlit run app.py`. | calculator screenshot or 20-second screen recording | All |

## Design notes for the deck

- **One message per slide.** Make the slide title the takeaway ("Simple is enough", not "Forecast results").
- **Text.** At most 3 bullets or 3 numbers per slide; the script carries the rest.
- **Charts.** Use the figures as they are (same palette throughout). Crop titles that repeat the slide title.
- **Numbers.** Keep the S$ format consistent: S$7.3k on slides, S$7,284 in the report.
- **Backup slides** (after slide 17, not timed), for questions:
  - the full buffer table for A/B/E;
  - the Diebold–Mariano p-value table;
  - the Kupiec/Christoffersen results;
  - the M6 design grid.

## Prepared answers

- **"Why not a fancier model?"** We tried six alternatives to the rolling percentile. Diebold–Mariano tests find none significantly better, and little predicts direction (R² = 0.02).
- **"Why is the volatility model the buffer model if it's only tied?"** Our selection rule (lowest 2014–19 loss) was fixed in advance, and we don't change rules after seeing the test. It barely matters: Cat B today is S$7,284 vs S$7,617 with the rolling percentile.
- **"You tried 16 versions of the volatility model. Isn't that overfitting?"** We chose on 2014–19 only and still show the test: the winner ranks 3rd of 16 there. So we claim it is equally good, not better.
- **"Adaptive calibration is state of the art. Why didn't it help?"** It made coverage almost exact (gap 2.6 → 0.4 points), but cost depends on how big the misses are, not just how many. The nudging made buffers jumpier: +1.4% cost.
- **"How do you know c_o?"** We don't. We sweep k = 1–19, show the k current rules imply (0.84 and 2.6), and give labelled scenarios.
- **"Isn't fixed 5% just as good?"** For k ≈ 2–4, yes, and we say so. Not for other dealers, and its coverage drops to 59% in turbulent periods.
- **"Do your buffers fail in bunches?"** No, for the recommended rules (Christoffersen tests on non-overlapping paths, Bonferroni-corrected). Only quantile regression clusters, in Cat E: a 42% chance of a miss after a miss vs 12% otherwise.
- **"Would dealers' own bids change the result?"** Yes, if the whole market switched at once. For one price-taking dealer, the result holds.
