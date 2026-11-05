# Presentation outline (11 slides, about 12 minutes)

Each phase lead presents their part. Figures are in `figures/`.

| # | Slide | Content | Visual | Presenter |
|---|---|---|---|---|
| 1 | **The January 2024 jump** | Cat B went from S$85,010 to S$112,000 in one round (+S$26,990). A dealer that guaranteed the COE at S$85k absorbed it. | `03_1_premium_events.png` | Report lead |
| 2 | **The dealer's problem** | Since 2023, a no-buffer dealer was underwater on 64% of Cat A and B sales: S$4,920 / S$6,180 per underwater car, worst S$20,216. MOT does not track these packages. Research question. | big numbers | Report lead |
| 3 | **The newsvendor idea** (the prof's suggestion) | Buffer = order quantity; premium rise = demand; c_u = shortfall, c_o = lost or discounted sales. b* = the τ = k/(1+k) quantile of the forecast. | mapping table + formula | Decision lead |
| 4 | **Data and what drives rises** | 398 rounds, 2010–2026; policy events. Little predicts the direction (R² 0.02); announced COE injections / loan curbs / rebate cuts → 4.3 points lower; mean reversion. Risk clusters in volatility. | `03_3_volatility.png` | EDA/regression lead |
| 5 | **Forecasting the range, not the point** | M0–M7, rolling origin; model chosen on 2014–19, tested on 2020–26. The volatility-scaled model M6 wins validation (−1.7% vs M1) but not test. Diebold–Mariano: rolling percentile, quantile regression, GBM and volatility scaling are statistically tied; SARIMAX 10% worse (p < 0.001); adaptive calibration (M7) significantly worse than M1 (p = 0.001) despite better coverage. | `05_calibration.png`, `05_fan_catB.png`, `08_calibration_tradeoff.png` | Forecasting lead |
| 6 | **What today's rules imply** | No buffer ⇒ k = 0.84 (acts as if overpricing were worse). Fixed 5% ⇒ k = 2.6, but its coverage drops to 59% in 2020–22. | small table | Decision lead |
| 7 | **The buffer table (headline)** | Rows = recent volatility. k = 4: about 5–9% of premium; k = 9: 8–14.5%. The cost ratio matters most; volatility next (2.1-point spread); bid pressure least (0.5). One-line formula: buffer = z × volatility (z = 1.2 at k = 4). Today (calm market) at k = 4: A S$6.1k, B S$7.3k, E S$4.8k. | buffer table + formula | Decision lead |
| 8 | **The frontier** | Model rules (including the volatility-scaled R7) and fixed-% rules sit on nearly the same frontier. A fixed 5% is near-optimal only if k ≈ 2–4; at k = 1 or 9 the newsvendor rule is cheaper (bootstrap CIs exclude 0); at k = 9 also on the 2020–26 test period alone. | `07_frontier.png` | Decision lead |
| 9 | **When it matters: turbulent spells, and robustness** | 2020–22: fixed 5% underwater on about 40% of sales vs about 22–25% for the rolling percentile. Nov 2023 / Jan 2024 jumps: the model buffers widened (S$11k–17k vs S$5.5k); only the volatility-scaled rule fully covered Jan 2024. Results unchanged for windows of 36–72 rounds and any refit frequency. | `07_frontier_by_period.png`, `08_sensitivity_window.png` | Forecasting lead |
| 10 | **Limitations** | Hypothetical backtest; c_o unknown; dealers' bids move the premium; regime shift after 2022; directional event flags rest on few events. | bullets | Report lead |
| 11 | **Recommendation** | 1. Always buffer. 2. Pick k first. 3. Let the buffer follow volatility and horizon (z × volatility, or a rolling quantile; no recalibration needed). 4. Fixed 5% only if your k ≈ 2–4. Live demo: `streamlit run app.py` (screenshot as backup). | calculator screenshot | All |

## Prepared answers

- **"Why not a fancier model?"** We tried several. Gradient boosting (M4) and quantile regression (M3) are within 1% of the 48-round rolling quantile, and Diebold–Mariano tests find no significant difference (p = 0.41–0.89). Little predicts the direction of the next move (R² = 0.02), so the extra features only fine-tune the spread. More techniques are not better.
- **"How do you know c_o?"** We don't. So we sweep k from 1 to 19, show the k that current rules imply (0.84 and 2.6), and give labelled scenarios. For example, a 5% chance of losing a S$8,000-margin sale per extra S$1,000 gives k ≈ 2.5, τ ≈ 0.7.
- **"Would dealers' own bids change the result?"** Yes, if the whole market changed its rule at once: dealers are the main bidders. For one dealer acting as a price-taker the result holds. We state this as a limitation.
- **"Isn't fixed 5% just as good?"** For a dealer with k ≈ 2–4, yes, and we say so. It fails for other cost ratios, and in turbulent periods its coverage drops to 59%.
- **"You tried 16 versions of the volatility model. Isn't that overfitting?"** We chose on 2014–19 only, and we show the test result anyway: the winner ranks 3rd of 16 on 2020–26, and only 2 of 16 beat the plain rolling percentile there. So we do not claim volatility scaling is better, only that it is equally good and easy to use.
- **"Adaptive calibration is state of the art. Why didn't it help?"** It made coverage almost exact in 2020–22 (gap 2.6 → 0.4 points), but the dealer's cost depends on how far the misses are, not only how many. Moving the quantile after every miss made the buffer jumpier and raised the cost by 2–8%.
