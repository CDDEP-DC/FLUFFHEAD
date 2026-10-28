"""From region trajectories to state and national FluSight targets.

1. Joint samples across regions. Each region has its own trajectories; a
   Gaussian copula on the rank of each trajectory's forecast-window total (shared
   factor with correlation REGION_RHO) picks one per region per joint sample,
   so a bad season in one region tends to go with a bad season elsewhere.
2. States. A state's sample is the sum of its pieces divided by the share of the
   state's population they cover: the unsimulated remainder is assumed to have
   the simulated part's per-capita rate. States in SCALED_STATES (Puerto Rico)
   take another group's per-capita curve, levelled to their own recent reports.
3. Observation noise (negative binomial) on each state-week, then US = sum of
   the jurisdictions.
4. Weeks already reported (horizon -1) come from the nowcast, as in flusight-fm.
5. Quantiles, rate-change probabilities and peak targets from the samples.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .calibrate import NB_SIZE

log = logging.getLogger(__name__)


def season_weeks(start: dt.date, end: dt.date) -> pd.DatetimeIndex:
    """Saturdays ending epiweeks from the first full week after `start` to `end`."""
    first = start + dt.timedelta(days=(5 - start.weekday()) % 7)
    return pd.date_range(first, end, freq="W-SAT")


def region_weekly(res: dict, weeks: pd.DatetimeIndex) -> np.ndarray:
    """[traj, weeks, pieces] weekly sums of a region's daily trajectories."""
    d = res["daily"]
    dates = res["dates"]
    out = np.zeros((d.shape[0], len(weeks), d.shape[2]))
    for i, w in enumerate(weeks):
        m = (dates > w - pd.Timedelta(days=7)) & (dates <= w)
        out[:, i] = d[:, m].sum(1)
    return out


def joint_samples(weekly: dict[str, np.ndarray], window: np.ndarray, rng, n: int = C.N_SAMPLES,
                  rho: float = C.REGION_RHO) -> dict[str, np.ndarray]:
    """{group: trajectory index per joint sample}."""
    shared = rng.standard_normal(n)
    picks = {}
    for g, w in weekly.items():
        order = np.argsort(w[:, window].sum((1, 2)), kind="stable")
        z = rho * shared + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        rank = np.minimum((stats.norm.cdf(z) * len(order)).astype(int), len(order) - 1)
        picks[g] = order[rank]
    return picks


def state_samples(weekly: dict, pieces: dict, shares: dict, picks: dict, observed: pd.DataFrame,
                  weeks: pd.DatetimeIndex, locations: pd.DataFrame, cutoff: dt.date, rng) -> dict[str, np.ndarray]:
    """{state FIPS: [samples, weeks]} expected admissions (no observation noise yet)."""
    n = len(next(iter(picks.values())))
    tot, cov = {}, {}
    for g, w in weekly.items():
        sel = w[picks[g]]                                  # [samples, weeks, pieces]
        for k, s in enumerate(pieces[g]):
            tot[s] = tot.get(s, 0) + sel[:, :, k]
            cov[s] = cov.get(s, 0) + shares[g][k]
    out = {s: tot[s] / cov[s] for s in tot}
    log.info("state coverage: %s", ", ".join(f"{s} {cov[s]:.0%}" for s in sorted(cov) if cov[s] < 0.5))

    # scaled states: another group's per-capita curve, levelled to the state's last 6 reported weeks
    pop = locations.set_index("location")["population"]
    obs = observed.pivot_table(index="date", columns="location", values="value")
    recent = weeks[(weeks <= pd.Timestamp(cutoff))][-6:]
    for s, g in C.SCALED_STATES.items():
        if g not in weekly:
            log.warning("no results for %s; %s (scaled from it) is left out", g, s)
            continue
        src = weekly[g][picks[g]].sum(2)                   # [samples, weeks]
        src_pop = sum(shares[g][k] * pop[p] for k, p in enumerate(pieces[g]))
        per_cap = src / src_pop
        y = obs.reindex(recent)[s].to_numpy() if s in obs else np.zeros(len(recent))
        idx = np.searchsorted(weeks, recent)
        m = (np.nansum(y) + 1) / (per_cap[:, idx].sum(1) * pop[s] + 1)
        out[s] = per_cap * pop[s] * m[:, None]
    return out


def add_noise(x: np.ndarray, rng) -> np.ndarray:
    m = np.maximum(x, 1e-9)
    return rng.negative_binomial(NB_SIZE, NB_SIZE / (NB_SIZE + m)).astype(float)


def nowcast_draws(nowcasts: pd.DataFrame, loc: str, date: pd.Timestamp, n: int, rng) -> np.ndarray | None:
    row = nowcasts[(nowcasts.location == loc) & (nowcasts.date == date)]
    if not len(row):
        return None
    r = row.iloc[0]
    return np.maximum(np.exp(r.mu + r.sd * rng.standard_normal(n)) - C.LOG_OFFSET, 0)


# ------------------------------------------------------------------ targets
def rate_change_pmf(diff: np.ndarray, stable: float, large: float) -> np.ndarray:
    """Category probabilities for (target week - horizon -1 week) samples, as in flusight-fm:
    within +/- stable is stable, beyond +/- large is a large change."""
    stable = max(stable, C.MIN_STABLE_COUNT)
    large = max(large, stable)
    cats = np.digitize(diff, [-large, -stable, stable, large])
    return np.bincount(cats, minlength=5) / len(diff)


def build(samples: dict[str, np.ndarray], weeks: pd.DatetimeIndex, revised: pd.DataFrame, nowcasts: pd.DataFrame,
          locations: pd.DataFrame, ref_date: dt.date, rng, jurisdictions: list[str]) -> tuple[pd.DataFrame, dict]:
    """jurisdictions: every hub location the US total is made of (all except US itself)."""
    """Submission frame plus the noisy samples used (for the report)."""
    last_obs = pd.Timestamp(ref_date) - pd.Timedelta(weeks=1)
    n = len(next(iter(samples.values())))
    noisy = {s: add_noise(x, rng) for s, x in samples.items()}
    # US is the sum of all jurisdictions; with any missing it would be biased low, so leave it out
    if all(l in noisy for l in jurisdictions):
        noisy["US"] = sum(noisy[l] for l in jurisdictions)
    else:
        log.warning("US left out: no forecast for %d jurisdictions", sum(l not in noisy for l in jurisdictions))
    obs = revised.pivot_table(index="date", columns="location", values="value")

    # reported weeks: observed (nowcast-revised) values; horizon -1 and lag 1 from nowcast draws
    for loc, x in noisy.items():
        for i, w in enumerate(weeks):
            if w > last_obs:
                break
            d = nowcast_draws(nowcasts, loc, w, n, rng)
            if d is not None:
                x[:, i] = d
            elif loc in obs and w in obs.index and not np.isnan(obs.at[w, loc]):
                x[:, i] = obs.at[w, loc]

    locs = locations.set_index("location")
    cap = (locs["population"] * C.MAX_POP_FRACTION).to_dict()
    rows = []
    for loc, x in noisy.items():
        if loc not in locs.index:
            continue
        hidx = {h: int(np.searchsorted(weeks, pd.Timestamp(ref_date) + pd.Timedelta(weeks=h))) for h in C.HORIZONS}
        for h, i in hidx.items():
            ted = (pd.Timestamp(ref_date) + pd.Timedelta(weeks=h)).date()
            q = np.minimum(np.rint(np.quantile(x[:, i], C.QUANTILES)), cap[loc])
            rows += [(ref_date, C.HOSP_TARGET, h, ted, loc, "quantile", f"{p:g}", int(v)) for p, v in zip(C.QUANTILES, q)]
        base = x[:, hidx[-1]]
        for h in C.RATE_CHANGE_HORIZONS:
            ted = (pd.Timestamp(ref_date) + pd.Timedelta(weeks=h)).date()
            s_col, l_col = C.RATE_CHANGE_COLS[h]
            pmf = rate_change_pmf(x[:, hidx[h]] - base, locs.at[loc, s_col], locs.at[loc, l_col])
            rows += [(ref_date, C.RATE_TARGET, h, ted, loc, "pmf", cat, float(p))
                     for cat, p in zip(C.RATE_CHANGE_CATEGORIES, pmf)]
        sw = (weeks >= pd.Timestamp(C.SEASON_START)) & (weeks <= pd.Timestamp(C.SEASON_END))
        season = x[:, sw]
        pk = np.bincount(season.argmax(1), minlength=sw.sum()) / n
        rows += [(ref_date, C.PEAK_WEEK_TARGET, None, None, loc, "pmf", str(w.date()), float(p))
                 for w, p in zip(weeks[sw], pk)]
        size = np.minimum(np.rint(np.quantile(season.max(1), C.QUANTILES)), cap[loc])
        rows += [(ref_date, C.PEAK_INC_TARGET, None, None, loc, "quantile", f"{p:g}", int(v))
                 for p, v in zip(C.QUANTILES, size)]
    cols = ["reference_date", "target", "horizon", "target_end_date", "location", "output_type", "output_type_id", "value"]
    df = pd.DataFrame(rows, columns=cols)
    df["horizon"] = df["horizon"].astype("Int64")
    return df.sort_values(["target", "location", "horizon"], kind="stable").reset_index(drop=True), noisy


def validate(df: pd.DataFrame, locations: pd.DataFrame) -> list[str]:
    """Basic checks; the hub's own validation runs on the pull request."""
    problems = []
    keys = ["target", "location", "horizon"]
    q = df[df.output_type == "quantile"]
    for key, g in q.groupby(keys, dropna=False):
        v = g.assign(level=g.output_type_id.astype(float)).sort_values("level").value.to_numpy(float)
        if len(v) != len(C.QUANTILES):
            problems.append(f"{key}: {len(v)} quantiles, expected {len(C.QUANTILES)}")
        if (np.diff(v) < 0).any() or (v < 0).any():
            problems.append(f"{key}: quantiles not non-decreasing/non-negative")
    hosp = q[q.target == C.HOSP_TARGET]
    for loc in locations.location:
        if set(hosp.loc[hosp.location == loc, "horizon"]) != set(C.HORIZONS):
            problems.append(f"{C.HOSP_TARGET}: missing horizons for {loc}")
    for key, g in df[df.output_type == "pmf"].groupby(keys, dropna=False):
        if abs(g.value.sum() - 1) > 1e-6:
            problems.append(f"{key}: pmf does not sum to 1")
    return problems


def save(df: pd.DataFrame, ref_date: dt.date) -> Path:
    d = C.FORECAST_DIR / "model-output" / C.model_id()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{ref_date}-{C.model_id()}.csv"
    df.to_csv(path, index=False)
    return path
