"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

What the page does, and what this program therefore models:

Exophase does not update a game's trophy counts continuously. The house's
recording shows the page frozen for hours, then jumping at a refresh: the
tracked-player count leaps and almost every earn rate falls, because the
newly tracked players hold far fewer trophies than the early crowd did.
Between refreshes the printed rates do not move at all. A forecast made
from the printed rates is therefore a forecast of a step process, and this
program treats it as one:

1. The history is first compressed to its distinct states (a new state
   starts when the players-tracked count or any rate changes). Duplicate
   reads of a frozen page carry no information and would swamp any fit.

2. If the history holds real movement (at least three states spanning at
   least six hours), each trophy's logit rate is fitted against
   ln(1 + age in hours since release) by duration-weighted least squares.
   The per-trophy slope is shrunk toward the median slope of its tier
   (platinum, gold, silver, bronze move at characteristically different
   speeds, and one trophy's noisy slope is weak evidence), the forecast is
   anchored mostly on the last printed rate, and the extrapolation over
   the remaining log-age gap is damped slightly, because adoption curves
   bend. The spread combines the fit's residuals with a random-walk term.

3. If the page has never moved in the history (the launch-night shape),
   there is no trend to fit, and the honest forecast is a two-branch
   mixture on the logit scale: with probability 1 - p the page stays
   frozen until the target and the rate is where it was; with probability
   p a refresh lands first, the tracked population multiplies, and every
   rate is diluted toward what the incoming players have earned (almost
   nothing but the universal first trophy, which stays at 100%). p comes
   from a refresh hazard: the longer the page has been frozen and the
   longer the horizon, the likelier a refresh intervenes. The five
   quantiles are read off the mixture itself, not a normal approximation
   to it, so the 5% tail sits down in the diluted branch where it belongs.

Two states, or a short span, blend the two: the observed step, shrunk
halfway to flat, plus the staleness mixture on top.

Everything is computed from the case file alone: no network, no clock, no
randomness, so two runs write the same bytes. Standard library only.

Run by the grader as `python3 -P -s /bot/main.py` (Python 3.13). Outside
the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input
and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Random-walk spread of a logit rate, per square-root day (the house
# baseline's own constant; kept so the no-refresh branch is never less
# honest than the baseline about ordinary drift).
RW_SIGMA = 0.5
# Mean hours between Exophase refreshes of a young game's page. The house
# saw launch night frozen from 23:48 to past 04:00, then a refresh by
# 04:30; a week-old game is refreshed less frantically, so the gap prior
# widens with the game's age.
GAP_HOURS_YOUNG = 5.5
GAP_HOURS_OLD = 14.0
# If a refresh lands, the tracked population multiplies by about this
# before the target, early in a game's life when the base is small.
REFRESH_GROWTH = 3.2
# The share of a trophy's current rate that incoming tracked players have
# typically earned: they have barely started, so for every trophy but the
# universal one this is small.
INCOMING_ADOPTION = 0.08
# How much of the fitted log-age extrapolation to trust over the horizon.
EXTRAP_DAMP = 0.9


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_instant(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_start(case):
    day = datetime.fromisoformat(case["game"]["release_date"])
    return day.replace(tzinfo=timezone.utc)


def age_hours(case, instant):
    return (instant - release_start(case)).total_seconds() / 3600.0


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantiles(components, levels):
    """Quantiles of a mixture of normals (in logit space) by bisection.

    components: [(weight, mu, sigma)] with weights summing to 1. Returns
    the logit-space quantile at each level, deterministically.
    """
    out = []
    for tau in levels:
        lo, hi = -14.0, 14.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            cdf = sum(w * norm_cdf((mid - mu) / sd) for w, mu, sd in components)
            if cdf < tau:
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return out


def distinct_states(case):
    """The history with consecutive duplicate reads collapsed.

    Each state keeps the first read that showed it and how long it lasted
    (until the next state, or the checkpoint for the last one).
    """
    states = []
    checkpoint = parse_instant(case["at"])
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if states and states[-1]["key"] == key:
            continue
        states.append({"key": key, "at": parse_instant(read["at"]),
                       "players": read["players_tracked"],
                       "rates": read["rates"]})
    for i, state in enumerate(states):
        end = states[i + 1]["at"] if i + 1 < len(states) else checkpoint
        state["duration_h"] = max(0.05, (end - state["at"]).total_seconds() / 3600.0)
    return states


def last_rates(states, n):
    """Each trophy's rate in the last state that printed one for it."""
    out = []
    for i in range(n):
        rate = None
        for state in reversed(states):
            if state["rates"][i] is not None:
                rate = state["rates"][i]
                break
        out.append(rate if rate is not None else 0.0)
    return out


def refresh_probability(case, states):
    """Chance at least one refresh lands between the last state change
    that the checkpoint can see and the target.

    The exposure is the time the current page has already been frozen
    (a refresh is overdue in proportion) plus the horizon itself.
    """
    checkpoint = parse_instant(case["at"])
    frozen_h = (checkpoint - states[-1]["at"]).total_seconds() / 3600.0
    horizon_h = case["horizon_minutes"] / 60.0
    age = age_hours(case, checkpoint)
    gap = GAP_HOURS_YOUNG if age < 72 else GAP_HOURS_OLD
    return 1.0 - math.exp(-(frozen_h + horizon_h) / gap)


def dilution_shift(rate):
    """Logit shift of one trophy's rate if a refresh multiplies the
    tracked population by REFRESH_GROWTH before the target."""
    if rate >= 99.0:
        return 0.0
    incoming = rate * INCOMING_ADOPTION
    diluted = (rate + (REFRESH_GROWTH - 1.0) * incoming) / REFRESH_GROWTH
    return logit(diluted) - logit(rate)


def stale_components(case, states, rate):
    """The two-branch mixture for a page that has shown no movement."""
    p = refresh_probability(case, states)
    days = case["horizon_minutes"] / 1440.0
    calm_sigma = RW_SIGMA * math.sqrt(days) + 0.05
    base = logit(rate)
    components = [(1.0 - p, base, calm_sigma)]
    if rate < 99.0:
        components.append((p, base + dilution_shift(rate), 0.55))
    else:
        components[0] = (1.0, base, calm_sigma)
    return components


def fit_slopes(case, states):
    """Per-trophy log-age slopes with tier shrinkage, or None when the
    history is too thin to fit.

    The fit runs over the recent window (the last 36 hours of states):
    an adoption curve's early near-vertical rise says little about its
    slope now, and a whole-history fit is dragged by it. State durations
    weight the fit (a state Exophase kept for six hours is six hours of
    evidence) but are capped so one long freeze cannot dominate.
    """
    if len(states) < 3:
        return None
    span_h = (states[-1]["at"] - states[0]["at"]).total_seconds() / 3600.0
    if span_h < 6.0:
        return None
    last_at = states[-1]["at"]
    recent = [s for s in states
              if (last_at - s["at"]).total_seconds() / 3600.0 <= 36.0]
    if len(recent) >= 4:
        states = recent
    xs = [math.log1p(max(0.0, age_hours(case, s["at"]))) for s in states]
    ws = [min(2.0, s["duration_h"]) for s in states]
    n = len(case["trophies"])
    slopes, residuals, centres = [], [], []
    for i in range(n):
        pts = [(x, logit(s["rates"][i]), w)
               for x, s, w in zip(xs, states, ws) if s["rates"][i] is not None]
        if len(pts) < 3:
            slopes.append(0.0)
            residuals.append(0.3)
            centres.append(None)
            continue
        sw = sum(w for _, _, w in pts)
        mx = sum(w * x for x, _, w in pts) / sw
        my = sum(w * y for _, y, w in pts) / sw
        sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
        if sxx <= 1e-9:
            slopes.append(0.0)
            residuals.append(0.3)
            centres.append(None)
            continue
        slope = sum(w * (x - mx) * (y - my) for x, y, w in pts) / sxx
        resid = math.sqrt(sum(w * (y - (my + slope * (x - mx))) ** 2
                              for x, y, w in pts) / sw)
        slopes.append(slope)
        residuals.append(min(0.9, max(0.12, resid)))
        centres.append((mx, my))
    tiers = {}
    for trophy, slope in zip(case["trophies"], slopes):
        tiers.setdefault(trophy["tier"], []).append(slope)
    tier_median = {}
    for tier, values in tiers.items():
        ordered = sorted(values)
        tier_median[tier] = ordered[len(ordered) // 2]
    shrunk, line_at_last = [], []
    for trophy, slope, centre in zip(case["trophies"], slopes, centres):
        target = tier_median[trophy["tier"]]
        slope = 0.8 * slope + 0.2 * target
        shrunk.append(slope)
        if centre is None:
            line_at_last.append(None)
        else:
            mx, my = centre
            line_at_last.append(my + slope * (xs[-1] - mx))
    return shrunk, residuals, xs[-1], line_at_last


def trend_components(case, states, rate, index, fit):
    slopes, residuals, x_last, line_at_last = fit
    target_age = age_hours(case, parse_instant(case["target"]))
    dx = math.log1p(max(0.0, target_age)) - x_last
    move = slopes[index] * dx * EXTRAP_DAMP
    move = max(-3.2, min(3.2, move))
    anchored = logit(rate) + move
    if line_at_last[index] is None:
        mu = anchored
    else:
        # Half anchored on the last printed rate, half the fitted line:
        # the anchor resists a bad fit, the line resists a last state
        # that a dilution dip happened to freeze.
        mu = 0.5 * anchored + 0.5 * (line_at_last[index] + move)
    days = case["horizon_minutes"] / 1440.0
    sigma = math.sqrt(residuals[index] ** 2 + (RW_SIGMA * math.sqrt(days)) ** 2 * 0.64 + 0.04)
    return [(1.0, mu, sigma)]


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels")
    states = distinct_states(case)
    if not states:
        raise ValueError("no history")
    rates = last_rates(states, len(case["trophies"]))
    fit = fit_slopes(case, states)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = rates[i]
        if fit is not None:
            components = trend_components(case, states, rate, i, fit)
            # Even a fitted history can end on a stale page: fold a small
            # dilution branch in when a refresh looks likely and near.
            p = refresh_probability(case, states)
            if p > 0.45 and rate < 99.0:
                mu, sd = components[0][1], components[0][2]
                components = [(1.0 - 0.35 * p, mu, sd),
                              (0.35 * p, mu + 0.6 * dilution_shift(rate), 0.6)]
        else:
            components = stale_components(case, states, rate)
        qs = [inv_logit(y) for y in mixture_quantiles(components, LEVELS)]
        qs = [round4(min(100.0, max(0.0, q))) for q in qs]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
