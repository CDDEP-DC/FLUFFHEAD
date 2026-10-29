"""Where in a state the flu is: NSSP emergency department visits by sub-state region.

CDC dataset rdmq-nq56 gives, each week, the percent of ED visits that are for
influenza in each Health Service Area (HSA; one or more counties that share care
seeking), reported on every county of the HSA, plus a state row. 13 states
(including CA, FL, PA, OH, WA) report no sub-state values; their counties take the
state value, so they fall back to population shares, unless a state health
department source gives county-to-state ratios (state_data.py: CA regional
admissions, PA county cases), in which case county % = state % x ratio.

A county's flu activity in a week is its population x (3-week mean flu % + FLOOR).
The floor keeps weeks with almost no flu (and tiny, noisy percentages) close to
population shares. A piece's share of its state's admissions is then its counties'
activity over the state's. This assumes ED visits per person and admissions per flu
ED visit are about the same across a state's HSAs.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

NSSP_URL = "https://data.cdc.gov/resource/rdmq-nq56.csv"
FLOOR = 0.1          # percentage points added to every county's flu % of ED visits
SMOOTH_WEEKS = 3
FORECAST_WEEKS = 3   # shares for forecast weeks: mean of the last this many weeks


def fetch(start: dt.date, end: dt.date) -> pd.DataFrame:
    """County and state rows of flu % of ED visits for weeks ending start..end (cached per day)."""
    import requests
    path = C.CACHE_DIR / f"nssp-{start}-{end}-{dt.date.today()}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(NSSP_URL, timeout=300, params={
            "$select": "week_end,geography,fips,trend_source,percent_visits_influenza",
            "$where": f"week_end >= '{start}T00:00:00' AND week_end <= '{end}T00:00:00' "
                      "AND trend_source in ('HSA', 'State')",
            "$limit": 2_000_000})
        r.raise_for_status()
        path.write_bytes(r.content)
    d = pd.read_csv(path, dtype={"fips": str})
    d["date"] = pd.to_datetime(d.week_end).dt.tz_localize(None).dt.normalize()
    return d


def activity(start: dt.date, end: dt.date) -> pd.DataFrame:
    """[weeks, county FIPS] flu activity (population x smoothed flu % + floor), every county."""
    from .immunity import _state_names
    counties = pd.read_csv(C.REGIONS_DIR / "counties.csv", dtype={"fips": str, "state": str}, keep_default_na=False)
    pop = counties.set_index("fips")["pop"]
    state_of = counties.set_index("fips")["state"]
    fetch_from = start - dt.timedelta(weeks=SMOOTH_WEEKS)
    try:
        d = fetch(fetch_from, end)
    except Exception as e:   # no ED data -> population shares
        log.warning("NSSP ED data unavailable (%s); using population shares", e)
        return pd.DataFrame()
    weeks = pd.DatetimeIndex(sorted(d.date.unique()))
    hsa = d[d.trend_source == "HSA"].assign(fips=lambda x: x.fips.str.zfill(5))
    pct = hsa.pivot_table(index="date", columns="fips", values="percent_visits_influenza").reindex(
        index=weeks, columns=pop.index)
    names = {v: k for k, v in _state_names().items()}
    st = d[d.trend_source == "State"].assign(state=lambda x: x.geography.map(names)).dropna(subset=["state"])
    state_pct = st.pivot_table(index="date", columns="state", values="percent_visits_influenza").reindex(weeks)
    # counties without a value: their state's value, else the population-weighted mean of reported counties
    reported = (pct * pop).T.groupby(state_of).sum(min_count=1).T / pct.notna().mul(pop).T.groupby(state_of).sum().T
    fill = state_pct.combine_first(reported)
    for s, cols in pct.columns.groupby(state_of[pct.columns]).items():
        if s in fill:
            pct[cols] = pct[cols].apply(lambda col: col.fillna(fill[s]))
    from . import state_data as SD
    for s, ratio in SD.ratios(weeks).items():
        if s not in fill:
            continue
        r = ratio.reindex(columns=[c for c in pct.columns if state_of[c] == s])
        pct[r.columns] = (r.mul(fill[s], axis=0)).where(r.notna(), pct[r.columns])
        log.info("county ratios from state data for %s (%d weeks)", s, int(r.notna().any(axis=1).sum()))
    pct = pct.rolling(SMOOTH_WEEKS, min_periods=1).mean()
    act = (pct + FLOOR) * pop
    n_sub = hsa.dropna(subset=["percent_visits_influenza"]).fips.str[:2].nunique()
    log.info("NSSP ED activity: %d weeks to %s, sub-state values in %d states", len(weeks),
             weeks.max().date() if len(weeks) else None, n_sub)
    return act[act.index >= pd.Timestamp(start)]


def piece_shares(act: pd.DataFrame, piece_counties: list[list[str]], states: list[str],
                 dates: pd.DatetimeIndex, pop_share: np.ndarray) -> np.ndarray:
    """[dates, pieces] each piece's share of its state's flu activity. Weeks without ED data
    use population shares; weeks after the last ED week use the mean of the last few."""
    out = np.tile(pop_share, (len(dates), 1)).astype(float)
    if act.empty:
        return out
    counties = pd.read_csv(C.REGIONS_DIR / "counties.csv", dtype={"fips": str, "state": str}, keep_default_na=False)
    a = act.reindex(columns=counties.fips)
    state_tot = a.T.groupby(counties.state.to_numpy()).sum(min_count=1).T
    sh = pd.DataFrame({k: a[c].sum(1, min_count=1) / state_tot[s]
                       for k, (c, s) in enumerate(zip(piece_counties, states))})
    sh = sh.fillna(pd.Series(pop_share, index=sh.columns))   # a state with no ED data at all (Iowa)
    if sh.empty:
        return out
    last = sh.iloc[-FORECAST_WEEKS:].mean().to_numpy()
    for i, d in enumerate(dates):
        if d in sh.index:
            out[i] = sh.loc[d].to_numpy()
        elif d > sh.index.max():
            out[i] = last
    return out
