# Pricing Guaranteed COE Packages: How Much Buffer Should a Car Dealer Charge?

IS5126 Group 4 · Final report (draft) · Data cut-off: October 2026, round 2

> Numbers in this draft come from the executed notebooks (`notebooks/01–08`). If the notebooks are re-run on the same frozen snapshot, every number reproduces exactly.

## 1. Problem

Singapore car dealers sell "guaranteed COE" packages. The dealer quotes a car price based on the latest Certificate of Entitlement (COE) premium and must then secure the buyer's COE in later bidding rounds, whatever the premium turns out to be. Any rise in between comes out of the dealer's margin. The Ministry of Transport has said it does not track these packages, so the risk rests entirely with dealers.

The risk is large. A dealer who quoted at the latest premium and secured the COE two rounds (about one month) later was **underwater on 64% of Category A and 64% of Category B sales since 2023**. The average loss was **S$4,920 (A) and S$6,180 (B)** per underwater car, and the worst 2-round rise was **S$20,216** (Cat A, September 2025). The single largest 1-round jump was S$26,990, in Cat B in January 2024 (S$85,010 → S$112,000).

**Research question.** How large a COE buffer should a dealer build into a guaranteed package, by category, so that it rarely loses money without overpricing the car?

**Framing: the newsvendor model.** Following the professor's suggestion, we treat the buffer *b* as the newsvendor's order quantity and the premium rise *R* between quote and purchase as uncertain demand:

| Newsvendor | Guaranteed-COE package |
|---|---|
| Underage cost *c_u* | *R* > *b*: the dealer absorbs the shortfall, S$1 per S$1 short |
| Overage cost *c_o* | *b* > *R*: the car is overpriced, so sales are lost or discounted |
| Optimal order | $b^* = F_R^{-1}\bigl(c_u/(c_u+c_o)\bigr)$, a quantile of the forecast distribution of *R* |

Writing *k* = *c_u*/*c_o*, the critical ratio is τ* = *k*/(1+*k*). Because *c_o* is private and unknown, we report buffers for τ = 0.5 to 0.95 (k = 1 to 19).

## 2. Data

**Source.** LTA *COE Bidding Results / Prices* (data.gov.sg): 1,990 rows covering 398 bidding rounds × 5 categories, January 2010 to October 2026. Each row has the quota, bids received, successful bids and the closing premium. There are no duplicates; the only gap is April–June 2020, when bidding was suspended during the circuit breaker. We use the car categories A, B and E (1,194 rows). We added 14 policy events (loan rules, the 2014 and 2022 Cat A redefinitions, zero vehicle growth from 2018, the suspension, quota injections in 2020, 2023 and 2025, and EV-rebate changes), each with its LTA, MOT or MAS source, its announcement date and its expected direction once in force: *down* (more supply or less demand: injections, loan curbs, rebate cuts; 6 events), *up* (zero vehicle growth, loan easing, a new rebate; 3) or *mixed* (category redefinitions, VES, the suspension; 5).

**Cleaning and target.** Rounds are numbered in time order within each category. Any target that spans the 2020 suspension is dropped, and `post_2022` marks the May 2022 redefinition. The target is the 2-round rise $r_{t,2} = P_{t+2}/P_t - 1$, modelled in % because premiums rose about 7× since 2010, and converted back to S$ with $P_t$.

**Features** use only information available at round *t*: bid ratio, unsuccessful share, momentum, 12-round and exponentially weighted (EWMA, λ = 0.94) volatility, the Cat E–Cat B gap, the announced quota change, round, month and announced policy events, flagged separately for price-lowering and price-raising events. An automated test rebuilds the features from a truncated panel and confirms that none of them changes, i.e. none uses later rounds.

**Exploratory findings** (`report/eda_findings.md`, figures `03_*`):

1. Premiums rose 6.9× (Cat B: S$19,190 → S$131,901) and 7.1× (Cat A) between January 2010 and October 2026.
2. In % terms the 2-round rise is narrower than in 2010–14 (90th percentile +15.5% → +9.0%; std. dev. 13.5% → 7.4%). Premiums are now much higher, though, so the dollar risk has not fallen: the 90th percentile rose from S$8,631 to S$9,161.
3. For Cat B since 2023, the 90th percentile of the 2-round rise is S$10,036 and the 95th S$12,220. The 2-round rise exceeded 10% in 8% of rounds (7 of 90).
4. Volatility clusters. Cat B's 12-round volatility ranged from 3.6% to 11.4% (10th to 90th percentile), and absolute 1-round changes have a lag-1 autocorrelation of 0.33. Calm and turbulent spells persist.
5. Demand pressure alone is a weak signal. The correlation between the bid ratio and the next 2-round rise is about 0 in Cat B.
6. There is little momentum. The lag-1 autocorrelation of 1-round changes is −0.13 (A), −0.01 (B) and +0.32 (E).
7. Log premiums are non-stationary (ADF p = 0.12–0.76) and their changes are stationary (p < 0.001), so we model rises rather than levels.

## 3. Methods

**Driver regression (Phase 4).** Pooled OLS of $r_{t,2}$ on the features with category and round dummies. We use Newey–West (HAC) standard errors with 2 lags, because consecutive 2-round rises overlap. We drop features with VIF > 5 (the unsuccessful share, VIF 11, which duplicates the bid ratio), test stability across May 2022 with interactions, and fit per category as a robustness check.

**Forecast distribution (Phase 5).** We compare eight models, all producing quantiles of the h-round rise at τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}:

| Model | Description |
|---|---|
| M0 | Naive: rise = 0 (the no-buffer dealer) |
| M1 | Rolling empirical quantile of the last 48 already-realised h-round rises |
| M2 | SARIMAX per category on 100·Δlog premium, with exogenous lagged bid ratio and quota change; ARMA order by AIC on 2010–13; quantiles from 2,000 simulated paths |
| M3 | Pooled linear quantile regression on bid ratio, announced quota change, 2-round momentum, 12-round volatility, `post_2022` and category: a feature-based newsvendor (Ban & Rudin) |
| M4 | Gradient boosting with quantile loss and depth-2 trees (stretch goal, h = 2 only) |
| M5 | Equal-weight average of the M1 and M3 quantiles |
| M6 | Volatility-scaled: forecast = *z*_τ × EWMA volatility, where *z*_τ is the τ-quantile of all past rises divided by the volatility at their quote (pooled across categories, expanding window). The dealer version is one line: buffer % = *z*_k × recent volatility |
| M7 | M1 with adaptive calibration (online quantile tracking): the quantile is shifted up after each resolved miss and down after each hit, with the h-round delay before an outcome is known; step size η chosen on 2014–19 |

M6 and M7 were added after the first round of results. M1's buffer reacted slowly to new turbulent spells, and coverage drifted in 2020–22. M6's design was chosen on the validation period only, as the best of 16 variants: volatility measure × scaling exponent × centring (notebook 08).

Evaluation uses a rolling origin from January 2014 with a refit every 4 rounds. At each origin the training set holds only targets already realised. We score with the pinball loss, which equals the newsvendor cost scaled by 1/(c_u + c_o), and check calibration (the share of actual rises below the τ-quantile). Results are reported overall and for 2014–19, 2020–22 and 2023–26. **The buffer model is chosen on 2014–19 only (validation); 2020–26 is an untouched test period.** Diebold–Mariano tests (per-round loss differences, HAC standard errors) check whether differences between models exceed chance.

**Newsvendor buffer (Phase 6).** $b^*_t = \max(0, \hat q_\tau)\cdot P_t$ from the model with the lowest pinball loss on the validation period. We also back out the τ that each current rule implies (its coverage). The headline table cuts the buffers by category × market condition × *k*, where the condition is the tercile of **recent volatility** (std. dev. of 1-round changes over the last 12 rounds). The bid-ratio version is kept for comparison.

**Backtest (Phase 7).** A simulated dealer sells one guaranteed car per category per round. It quotes 894 times between January 2014 and September 2026, each time charging the latest premium plus a buffer and buying two rounds later. If the premium falls, the dealer keeps the gain. We compare rules R0 (no buffer), R1 (fixed 5%), R2 (M1), R3 (M2), R4 (M3) and R7 (M6, the buffer model), plus R5 (M4), R6 (M5) and R8 (M7) in the appendix. Metrics are the share underwater, the average and worst shortfall, the average buffer and the newsvendor cost (shortfall + excess/*k*). Confidence intervals come from a moving-block bootstrap (8-round blocks, 1,000 resamples). Two further checks: a tougher benchmark, the best fixed % chosen with hindsight, and sensitivity tests for h = 1, 3, 4 and a refund case.

**Robustness (notebook 08).** We re-run the out-of-sample forecasts with:
- M1 windows of 24–96 rounds;
- M3 refit every 1, 2, 4 or 8 rounds, and M4 every 2, 4 or 8;
- all 16 M6 variants;
- adaptive calibration at η = 0–0.02 on M1, M3 and M6.

## 4. Results

### 4.1 What signals a rise (driver regression)

The features explain little of the 2-round rise: **R² = 0.018**. Two effects are significant at 5% (HAC):

- **Announced price-lowering events.** When an announced COE injection, loan curb or rebate cut takes effect in the next two rounds, the following 2-round rise is on average **4.3 percentage points lower**: about **S$5,680 less** on today's Cat B premium of S$131,901 (p = 0.003). Only 12 rows carry this flag, and the effect comes from the post-2022 events (its coefficient on 2010–22 alone is 0.000), so its size is uncertain.
- **Mean reversion.** After a 2-round rise one standard deviation larger than usual, the next 2-round rise is **0.8 points lower** (about S$1,010; p = 0.030).

**Announced price-raising events** (zero vehicle growth, loan easing, a new rebate; 12 rows) show no effect (−2.1 points, p = 0.52): with only three such events, the data cannot tell. Splitting the old single event flag by direction leaves the conclusion unchanged: the effect was the supply injections all along. The bid ratio is **not** significant: a move from 1.2 to 1.4 comes with +0.3 points (S$440; p = 0.19). Per category, the Cat E–B gap matters for Cat B (p = 0.003), the bid ratio for Cat E (p = 0.025), and price-lowering events for Cat A (p = 0.001). The effects shift after May 2022: a joint Wald test on the post-2022 interactions gives p = 0.028.

*So what for the dealer:* nothing a dealer can observe predicts the *direction* of the next move reliably, so a buffer cannot be tuned on bid pressure. These are associations, not causes, because dealers' own bids are part of the bid count.

### 4.2 Forecast quality

Mean pinball loss in S$ at h = 2, averaged over τ (lower is cheaper):

| Period | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 |
|---|---|---|---|---|---|---|---|---|
| Validation 2014–19 (used to choose) | 1,474 | 1,208 | 1,421 | 1,235 | 1,228 | 1,210 | **1,188** | 1,218 |
| Test 2020–26 (untouched) | 3,029 | 1,917 | 2,063 | 1,927 | 1,908 | **1,901** | 1,929 | 1,951 |
| All 2014–26 | 2,277 | 1,574 | 1,753 | 1,593 | 1,579 | 1,567 | 1,571 | 1,597 |

By the rule fixed in advance (lowest validation loss), **the buffer now uses M6, the volatility-scaled model**. Before M6 and M7 were added, the rule chose M1. M6's validation lead over M1 is 1.7%, but it does not carry over: on the test period M6 is 1.4% behind the best model (M5) and 0.6% behind M1.

**Diebold–Mariano tests** (p-values for equal loss, 2014–26 / test 2020–26):

| Comparison | Loss difference | p, 2014–26 | p, test 2020–26 |
|---|---|---|---|
| M6 vs M1 | −0.2% | 0.91 | 0.83 |
| M7 vs M1 | +1.4% | **0.001** | **0.004** |
| M1 vs M3 | −1.2% | 0.41 | 0.79 |
| M1 vs M4 | −0.3% | 0.89 | 0.88 |
| M1 vs M5 | +0.5% | 0.54 | 0.41 |
| M2 vs M1 | +10.2% | < 0.001 | 0.003 |
| every model vs M0 | −23% to −31% | < 0.001 | — |

Every model beats "no buffer" by about 30%. **M1, M3, M4, M5 and M6 are statistically indistinguishable**: features, gradient boosting and volatility scaling add no detectable accuracy over a rolling percentile. **Adaptive calibration (M7) is significantly worse than plain M1**, and it is the only one of the new ideas with a clear verdict. It does what it is designed to do: the mean coverage error in 2020–22 falls from 2.6 to 0.4 percentage points, and coverage at τ = 0.8 rises from 77% to 81%. But it buys that by moving the quantiles after every miss, and the extra movement costs more pinball loss than the better coverage saves. The same trade-off appears when it is applied to M3 and M6 (§4.6). SARIMAX is significantly worse (10% higher loss) because its Gaussian ranges are too wide in calm periods (89% coverage at τ = 0.8). At τ = 0.8 alone, M1 even beats M3 (−3.7%, p = 0.03).

Over 2014–26, calibration is good for:
- M1 and M7: coverage within 1 point of τ at every level;
- M6 and M5: within 2 points (M6 slightly under-covers);
- M3: it slightly over-covers, by 1–4 points.

M6's coverage is less stable by period than M1's. Its mean gap is 8.8 points in 2020–22, against 2.6 for M1. The ranking is the same at h = 1, 3 and 4. By period:
- **2014–19:** M6 has the lowest loss.
- **2023–26:** M4 is lowest, by under 0.5%. M6 is 1.8% behind M1 here: after the 2023–24 jumps its volatility-proportional buffers were too wide.
- **2020–22:** SARIMAX's wide ranges pay off and it is lowest (S$1,564 against S$1,605–1,666 for the others). In that turbulent spell, being cautious was right.

*So what:* a dealer does not need a complex model. A well-calibrated estimate of recent rise risk is what drives the buffer. A rolling percentile of past rises provides it, and so does a volatility-scaled rule, equally well.

### 4.3 What today's rules imply

Over 2014–2026, no buffer covered only **46%** of sales: it behaves as if overpricing were the bigger risk (implied k = 0.84). The fixed 5% buffer covered **72%** (implied τ = 0.72, k = 2.6), but its coverage swings with the market: 78% in 2014–19, **59% in 2020–22** and 72% in 2023–26.

### 4.4 The dealer's buffer table (headline)

The buffer as % of the premium is the median out-of-sample newsvendor buffer from the buffer model (M6), applied to the October 2026 premium (A S$132,004, B S$131,901, E S$135,790). The condition is the tercile of recent 12-round volatility per category. Full table: `data/processed/buffer_table.csv`.

| Category, recent volatility | k = 1 | k = 2.3 | k = 4 | k = 9 | k = 19 |
|---|---|---|---|---|---|
| A, Low (≤ 4.7%) | S$677 (0.5%) | S$4,915 (3.7%) | S$7,984 (6.0%) | S$11,927 (9.0%) | S$15,097 (11.4%) |
| A, High (> 7.1%) | S$322 (0.2%) | S$7,140 (5.4%) | S$11,878 (9.0%) | S$19,159 (14.5%) | S$24,380 (18.5%) |
| B, Low (≤ 4.9%) | S$761 (0.6%) | S$5,179 (3.9%) | S$8,395 (6.4%) | S$12,465 (9.5%) | S$16,146 (12.2%) |
| B, High (> 7.1%) | S$810 (0.6%) | S$6,464 (4.9%) | S$10,314 (7.8%) | S$16,616 (12.6%) | S$21,682 (16.4%) |
| E, Low (≤ 4.1%) | S$577 (0.4%) | S$4,361 (3.2%) | S$7,055 (5.2%) | S$10,695 (7.9%) | S$13,813 (10.2%) |
| E, High (> 5.5%) | S$339 (0.2%) | S$5,508 (4.1%) | S$9,222 (6.8%) | S$14,585 (10.7%) | S$19,124 (14.1%) |

*Guidance:* a dealer who rates underpricing four times as costly as overpricing (k = 4, τ = 0.8) should add:
- **about 7.8% (S$10,314) to a Cat B package after a turbulent spell**, meaning 12-round volatility above 7.1%;
- **6.4% (S$8,395) in a calm market**, meaning volatility below 4.9%.

The **cost ratio matters more than the market condition**. Moving from k = 2.3 to k = 9 more than doubles the buffer.

With M6 the market condition now matters clearly, which it did not with M1:
- **Spread.** The average spread across the three volatility conditions is 2.1 points, against 1.6 with M1 and 0.5 for bid-ratio conditions.
- **Ordering.** The buffer rises Low ≤ Medium ≤ High at every k except k = 1, in every category.
- **The M1 anomaly is gone.** With M1, Cat A at k = 4 needed less when turbulent (4.3%) than when calm (6.0%). With M6 the turbulent row is 9.0%.

The underwater rates show the limits. At k = 4, Cat B is underwater on 26% of turbulent-spell sales against 21% of calm ones. Cat E in calm markets is underwater on 37%: when volatility is very low, M6's proportional buffer becomes too small.

**The dealer's one-line formula** (`buffer_formula.csv`): buffer = *z_k* × EWMA volatility × premium.

| k | 1 | 1.5 | 2.3 | 4 | 9 | 19 |
|---|---|---|---|---|---|---|
| *z_k* | 0.15 | 0.41 | 0.76 | 1.20 | 1.79 | 2.34 |

At k = 4 a dealer adds 1.2 × volatility: 3.6% of the premium when volatility is 3%, 9.6% when it is 8%. Fixed 5% equals this rule only when volatility is 4.2%.

**Buffer to charge today.** The forecast was made at October 2026 round 2 for a purchase two rounds later. All three categories are calm. Their EWMA volatility is below its long-run median:

| Category | EWMA volatility today | Long-run median |
|---|---|---|
| A | 3.8% | 6.4% |
| B | 4.6% | 6.7% |
| E | 2.9% | 5.3% |

M6's buffers are:

| k | A | B | E |
|---|---|---|---|
| 4 | **S$6,101** | **S$7,284** | **S$4,802** |
| 9 | S$9,101 | S$10,865 | S$7,163 |

The tied models disagree by a few thousand dollars (`buffer_today_by_model.csv`). For Cat B at k = 4 the range is S$7,089 (M7) to S$10,016 (M3), with M1 at S$7,617. A dealer can treat that range as model uncertainty.

### 4.5 Backtest

Results for h = 2, Cat A/B/E pooled, 894 quotes from January 2014 to September 2026:

| Rule | Underwater | Avg shortfall | Worst shortfall | Avg buffer | Cost k = 4 | Cost k = 9 |
|---|---|---|---|---|---|---|
| R0 No buffer | 54.5% | S$4,351 | S$20,216 | S$0 | 2,873 | 2,594 |
| R1 Fixed 5% | 27.7% | S$3,251 | S$14,822 | S$3,647 | 1,949 | 1,367 |
| R2 Rolling percentile, τ = 0.8 | 19.4% | S$2,789 | S$14,447 | S$5,306 | 1,911 | 1,149 |
| R3 SARIMAX, τ = 0.8 | 11.0% | S$2,814 | S$10,707 | S$7,911 | 2,273 | 1,182 |
| R4 Quantile regression, τ = 0.8 | 18.2% | S$2,944 | S$13,647 | S$5,612 | 1,984 | 1,180 |
| R6 Combination, τ = 0.8 | 19.0% | S$2,767 | S$14,047 | S$5,459 | 1,925 | 1,144 |
| **R7 Volatility-scaled (buffer model), τ = 0.8** | 20.2% | S$3,212 | S$14,226 | S$4,992 | 1,971 | 1,237 |
| R8 M1 adaptive, τ = 0.8 | 19.6% | S$2,826 | S$14,232 | S$5,314 | 1,930 | 1,165 |

**Frontier chart** (`figures/07_frontier.png`). The model rules, R7 included, and the family of fixed-% rules lie on **almost the same frontier**. At the fixed-5% rule's protection level (28% underwater), the average buffer needed is:
- R1 (fixed 5%): S$3,647;
- R4: S$3,673;
- R7: S$3,751.

**The proposal's hoped-for "comparable protection at a lower cost" does not hold at the 5% rule's own operating point, even with a buffer that scales with volatility.**

**Where the newsvendor rule does help**, against fixed 5%, with 95% block-bootstrap CIs on the cost difference per car:

| k (τ*) | R7 Vol-scaled (buffer model) vs R1 | R2 Rolling percentile vs R1 | R4 Quantile regression vs R1 |
|---|---|---|---|
| 1 (0.5) | −S$716 [−998, −339] | −S$694 [−935, −374] | −S$682 [−910, −350] |
| 2.3 (0.7) | +S$32 [−34, +113] | +S$32 [−55, +127] | +S$81 [+17, +153] |
| 4 (0.8) | +S$22 [−57, +97] | −S$38 [−156, +52] | +S$35 [−88, +125] |
| 9 (0.9) | −S$229 [−402, −95] | −S$243 [−411, −120] | −S$219 [−418, −78] |

Fixed 5% is close to optimal for a dealer whose k is 2–4, which is the k it implies. For any other cost ratio the newsvendor rule is clearly cheaper, because it charges the right quantile.

The volatility-scaled rule was the best candidate to break the tie at k = 2–4, and it does not. Its intervals at k = 2.3 and k = 4 straddle zero just like M1's. The reason is that volatility only partly predicts the next jump. M6 spends more in turbulent spells and less in calm ones. But its calm-market buffers were sometimes too small (Cat E calm: 37% underwater at k = 4), and its turbulent-market buffers were partly wasted after spikes that had already passed.

Against the **best fixed % chosen with hindsight** (1% for k = 1, 4% for k = 2.3, 6.5% for k = 4, 9.5% for k = 9), the model rules come within 0.2–4% of the cost without needing hindsight (R7 0.6–4%, R2 0.2–3%).

Compared with no buffer, R4 saves S$889 per car at k = 4 (CI [−1,288, −562]) and S$1,445 at k = 9.

**Test period only (2020–26).** Fixed 5% was underwater on 33.5% of sales with an average buffer of S$4,650. To match that protection, the model rules need:
- R7: S$4,615, the only model rule slightly cheaper (−0.8%);
- R2: S$4,927;
- R8: S$5,016.

Against fixed 5%:

| k | R7 vs fixed 5% | R2 vs fixed 5% |
|---|---|---|
| 1 | −S$524 [−1,006, +85] | −S$455 [−807, −22] |
| 2.3 | **+S$114 [+39, +240]**: significantly *worse* | +S$97 [−24, +244] |
| 4 | −S$2 [−141, +156] | −S$127 [−341, +46] |
| 9 | −S$384 [−669, −137] | −S$431 [−700, −228] |

The conclusions from the full period hold out of sample: a clear saving for k = 9 and a tie around k = 2–4.

**By period.** In 2020–22, fixed 5% left the dealer underwater on 39–44% of sales. The model rules adapted to the higher volatility, at the cost of larger buffers:

| Rule | Underwater in 2020–22 |
|---|---|
| R1 Fixed 5% | 39–44% |
| R2 | 22–25% |
| R8 | 17–20% |
| R4 / R6 | 28–36% |
| R7 | 28–34% |

**Big jumps.** Before the Cat B November 2023 round-1 quote (rise S$20,099 over two rounds), fixed 5% charged S$5,500. The τ = 0.9 model buffers were:

| Rule | Buffer |
|---|---|
| R2 | S$11,126 |
| R3 | S$15,164 |
| R4 | S$16,804 |
| R6 | S$13,965 |
| R7 | S$16,422 |

For the January 2024 round-1 quote (rise S$17,328), R7 charged S$17,447 and R5 S$17,328, the only rules that fully covered it; R4 charged S$16,923. The volatility-scaled and feature-based models widened their buffers after the turbulent late-2023 rounds. R2's 48-round window did not (S$8,762).

**Sensitivity.** As the horizon grows from 1 to 4 rounds, fixed 5% is underwater on 16% → 35% of sales. The model rules stay near 18–21% at τ = 0.8, because their buffers grow with the horizon (R4: S$3,388 → S$7,641; R7: S$3,068 → S$7,451). In the **refund case** (unused buffer returned to the buyer), the buyer pays on average S$1,720 under R7 and S$1,833 under R4 at τ = 0.8, instead of S$4,992 and S$5,612. The dealer's risk is unchanged.

### 4.6 Robustness (notebook 08)

**M1's window.**
- Windows of 36, 48 and 72 rounds give the same validation loss (within 0.1%), and test losses of S$1,913–1,922.
- 24 rounds is clearly worse (+4% validation, +9% test): too few past rises to estimate a 95th percentile.
- 96 rounds is slightly worse.
- For every window, the newsvendor cost against fixed 5% at k = 4 is a tie: every CI includes zero.

**Refit frequency.** Refitting M3 every 1, 2 or 8 rounds instead of 4, or M4 every 2 or 8, changes the loss by at most 0.2% on validation and 0.6% on test. The "refit every 4" shortcut costs nothing.

**M6 design.**
- 9 of the 16 variants beat M1 on validation, but only 2 do on test.
- The validation winner (EWMA, β = 1, not centred) ranks 3rd of 16 on test.
- The validation and test rankings agree only moderately (Spearman 0.61).

Volatility scaling is therefore not a reliable improvement over a rolling percentile. Its validation edge is partly the luck of choosing among 16 variants.

**Adaptive calibration** (`figures/08_calibration_tradeoff.png`). On every model it closes the 2020–22 coverage gap, but it raises the loss:

| Model | 2020–22 coverage gap | 2020–22 loss | Test loss |
|---|---|---|---|
| M1 | 2.6 → 0.4 points | +2% | +2% |
| M3 | 9.7 → 1.5 points | +4% | +8% |
| M6 | 8.8 → 0.5 points | +5% | +8% |

On this data, **calibration and cost are different targets**. A forecast whose 80% line covers exactly 80% is not automatically the cheapest buffer. The newsvendor cost depends on how far off the misses are, not only how many there are.

## 5. Limitations

- **Hypothetical performance.** The backtest simulates one price-taking dealer, not proven savings. Dealers are the main bidders, so if the whole market adopted a new rule, premiums themselves could change.
- **c_o is not public.** We cannot name one "right" buffer; we report the full range of k and the k implied by current rules. The classic newsvendor also assumes c_o is linear per dollar, whereas small overprices may cost few sales.
- **Small sample and regime changes.** There are about 400 rounds per category, the May 2022 redefinition, and evidence that the drivers shifted after 2022 (p = 0.034). Linear and simple models are a deliberate choice, and their differences are within a few percent.
- **Model selection.** The buffer model is chosen on 2014–19 and tested on 2020–26. M6 and M7, however, were designed after the first results had been seen, including the 2020–22 coverage drift, so the test period is not fully untouched for them. M6's 1.7% validation lead disappears on the test period. The Diebold–Mariano tests confirm that M1, M3, M4, M5 and M6 are statistically tied, and the choice between them changes today's Cat B buffer at k = 4 by about S$300 (M6 S$7,284 vs M1 S$7,617). It does not drive the conclusions.
- **Unforeseeable policy shocks.** Only announced events enter the features. The directional event flags are rare (12 rows each), and the direction of each event is our judgement (`policy_events.csv`). Two announcement days (Mar 2017 VES, end of month used; the 2018 zero-growth cut, from a news report of the LTA release) are approximate.
- **Private contract terms.** Dealer margins, volumes and refund terms are private. The refund case is a sensitivity check only.

## 6. Recommendation

1. **Always charge a buffer.** No buffer loses money on more than half of sales (64% since 2023) and acts as if overpricing were the bigger risk.
2. **Choose the cost ratio first, then the buffer.** For k = 4 (underpricing four times as costly as overpricing), charge about 6–8% of the premium at h = 2: about S$7,300 at today's calm Cat B market, rising to about S$10,300 after a turbulent spell. A dealer with thin margins and loyal buyers (high k = 9) should go to about 8–12.5%: S$10,900 (8.2%) in today's calm Cat B market. A dealer in a price war (k ≈ 1) needs only 0–1%.
3. **Let the buffer move with volatility and horizon, not bid pressure.** Two equally good ways:
   - the one-line formula, buffer = *z_k* × recent volatility (z = 1.2 at k = 4);
   - a rolling 48-round percentile of past rises.

   No feature-based model forecasts significantly better. Both match the best hindsight fixed % and adapt in turbulent spells such as 2020–22, when fixed 5% failed on about 40% of sales. Do not bother with automatic recalibration: it fixes coverage but raises the cost.
4. **Keep a fixed 5% only if your k is around 2–4.** For that dealer the fixed rule is about as good as any model, volatility-scaled or not.

What stays uncertain: the true c_o, how buyers respond to visible buffers, and how a market-wide change of dealer behaviour would feed back into the premium.

## Appendix

- Code: notebooks `01–08`, `src/` (see `README.md`). Data: `data/raw/` (frozen at October 2026 round 2).
- Extra tables, all in `data/processed/`:
  - forecast scores by period and horizon: `forecast_eval*.csv`
  - buffers for 2023–26 and by bid ratio: `buffer_table_2023_26.csv`, `buffer_table_by_bidratio.csv`
  - Diebold–Mariano tests: `dm_tests.csv`; model selection: `best_model.json`
  - test-period bootstrap: `bootstrap_vs_R1_test.csv`
  - backtest by category × period: `backtest_by_category_period.csv`
  - horizon sensitivity: `backtest_sensitivity_h.csv`
  - bootstrap results: `bootstrap_vs_R1.csv`
  - hindsight benchmark: `vs_best_fixed_pct.csv`
  - the one-line formula and today's buffer under each model: `buffer_formula.csv`, `buffer_today_by_model.csv`
  - robustness: `sensitivity_window.csv`, `sensitivity_refit.csv`, `sensitivity_volscaled.csv`, `sensitivity_calibration.csv`
- Gradient boosting (M4/R5) is no better than M1 (pinball 1,579 vs 1,574; Diebold–Mariano p = 0.89), so, following the prof's tip that more techniques are not better, it stays in the appendix.

## References

[1] Land Transport Authority, "COE Bidding Results / Prices," data.gov.sg, 2026.
[2] Ministry of Transport, parliamentary reply on guaranteed COE packages, 5 May 2026.
[3] Ministry of Transport, written reply on mitigating a potential COE spike in late 2025, 23 Sep 2025.
[4] G.-Y. Ban and C. Rudin, "The Big Data Newsvendor: Practical Insights from Machine Learning," *Operations Research*, 67(1), 2019.
[6] I. Gibbs and E. Candès, "Adaptive Conformal Inference Under Distribution Shift," *NeurIPS*, 2021; A. Angelopoulos, E. Candès and R. Tibshirani, "Conformal PID Control for Time Series Prediction," *NeurIPS*, 2023 (online quantile tracking, used for M7).
[7] J.P. Morgan/Reuters, *RiskMetrics Technical Document*, 1996 (EWMA volatility, λ = 0.94).
[5] Policy-event sources: see `data/raw/policy_events.csv` (LTA, MOT and MAS releases with URLs).
