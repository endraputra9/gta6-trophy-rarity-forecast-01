# GTA VI Trophy Rarity Forecast

A program that forecasts, for every PlayStation 5 trophy of Grand Theft
Auto VI, the earn rate printed on its Exophase page at a case's target —
the percent of tracked players who have earned it — as five quantiles
(5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

`main.py` (Python 3.13, standard library only) reads every case file in
the input directory and writes one forecast file per case. It runs
offline, deterministically, in well under the 8-second limit.

## The model

Exophase refreshes a game's counts in jumps, hours apart: the page sits
frozen, then the tracked-player count leaps and almost every rate falls,
because the newly tracked players hold far fewer trophies. The program
models that step process directly:

1. **State compression.** Consecutive identical reads are collapsed to
   distinct states; a frozen page repeated forty times is one piece of
   evidence, not forty.
2. **Moving history** (≥3 states over ≥6 h): each trophy's logit rate is
   fitted against ln(1 + age in hours) over the last 36 hours of states,
   duration-weighted. Slopes are shrunk 80/20 toward their tier's median
   slope, the forecast blends the fitted line with an anchor on the last
   printed rate, and the extrapolation is damped by 0.9. The spread
   combines fit residuals with a random-walk term; if the final page
   looks stale, a partial dilution branch is folded in.
3. **Frozen history** (the launch-night shape): a two-branch mixture on
   the logit scale — the page stays frozen, or a refresh lands first
   (probability from a refresh hazard over the frozen time plus the
   horizon) and dilutes every rate toward what incoming players have
   earned. Quantiles are read off the mixture itself by bisection.

## Validation

There is no local server grade for this competition, so validation is
local and documented:

- **Practice reconstruction.** The kit states the practice answers'
  aggregates (3,868 players, 4,276 trophies earned after the refresh;
  every rate but the 100% one fell). Reconstructing per-trophy answers
  under two bracketing allocations of the 103 non-universal new earnings,
  the house baseline scores 1.37–1.40 (the rules state ≈1.51) and this
  program 0.48–0.53, with 90% coverage 0.98 against the baseline's 0.
- **Synthetic final backtest.** On simulated f00-shaped worlds
  (logistic player growth, saturating per-trophy adoption curves,
  refresh dilution every 3–8 h), this program's mean loss is ≈0.19–0.20,
  against ≈0.24 for the house curve baseline and ≈0.32 for persistence.

## License

MIT — see `LICENSE`.
