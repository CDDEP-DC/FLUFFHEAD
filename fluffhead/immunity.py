"""Vaccination and prior immunity for each agent.

Vaccination: CDC dataset vh55-3he6 (NIS-Flu / BRFSS) gives cumulative seasonal
vaccination coverage by state, age group and month. Each agent draws a uniform
u; if u is below their state and age group's end-of-season coverage they are
vaccinated on the day the cumulative coverage curve reaches u. Before a season's
own data exist, the latest complete season stands in for it.

Prior immunity: all-or-nothing protection from earlier seasons' infections and
cross-reactive antibodies, by age group; a calibrated parameter (see disease.py).
"""
from __future__ import annotations

import datetime as dt
import functools
import logging

import numpy as np
import pandas as pd

from . import config as C
from .engine import Immunity
from .network import N_AGE, Network

log = logging.getLogger(__name__)

VAX_URL = "https://data.cdc.gov/api/views/vh55-3he6/rows.csv?accessType=DOWNLOAD"
VAX_PATH = C.RAW_DIR / "cdc" / "vh55-3he6.csv"
# model age group -> (dataset age dimension, weight); 5-17 combines two dimensions by years of age
AGE_DIMS = [[("6 Months - 4 Years", 1.0)],
            [("5-12 Years", 8 / 13), ("13-17 Years", 5 / 13)],
            [("18-49 Years", 1.0)],
            [("50-64 Years", 1.0)],
            [(">=65 Years", 1.0)]]
INFANT_SHARE = 0.1   # under 6 months are not eligible: 1/2 year of the 0-4 group's 5 years
# season month -> calendar (year offset from season start year, month); coverage is cumulative to month end
SEASON_MONTHS = [(0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]


def fetch(force: bool = False) -> None:
    import requests
    if VAX_PATH.exists() and not force:
        return
    VAX_PATH.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(VAX_URL, timeout=300)
    r.raise_for_status()
    VAX_PATH.write_bytes(r.content)


def _state_names() -> dict[str, str]:
    from .greasypop import ABBR
    names = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
             "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
             "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
             "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
             "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
             "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
             "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
             "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
             "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
             "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
             "PR": "Puerto Rico"}
    return {fips: names[a] for fips, a in ABBR.items()}


@functools.lru_cache(maxsize=4)
def coverage(season_start_year: int) -> pd.DataFrame:
    """Cumulative coverage (fraction) by state FIPS, model age group and month end.

    Uses the requested season if the dataset has it, else the latest season available.
    Missing or unreliable state estimates fall back to the national value.
    """
    fetch()
    v = pd.read_csv(VAX_PATH, dtype=str)
    v = v[(v.Vaccine == "Seasonal Influenza") & (v["Dimension Type"] == "Age")]
    seasons = sorted(v["Season/Survey Year"].unique())
    want = f"{season_start_year}-{str(season_start_year + 1)[2:]}"
    used = want if want in seasons else seasons[-1]
    if used != want:
        log.info("vaccination: no %s data yet; using %s coverage", want, used)
    v = v[v["Season/Survey Year"] == used].copy()
    v["est"] = pd.to_numeric(v["Estimate (%)"].str.extract(r"([\d.]+)")[0], errors="coerce") / 100
    v["Month"] = v.Month.astype(int)

    names = _state_names()
    nat = v[v.Geography == "United States"]
    rows = []
    for fips, name in names.items():
        s = v[v.Geography == name]
        for ag, dims in enumerate(AGE_DIMS):
            for yo, mo in SEASON_MONTHS:
                tot = 0.0
                for dim, w in dims:
                    x = s[(s.Dimension == dim) & (s.Month == mo)].est
                    if x.isna().all() or not len(x):
                        x = nat[(nat.Dimension == dim) & (nat.Month == mo)].est
                    tot += w * (float(x.iloc[0]) if len(x) and not np.isnan(x.iloc[0]) else 0.0)
                if ag == 0:
                    tot *= 1 - INFANT_SHARE
                end = dt.date(season_start_year + yo, mo, 1) + pd.offsets.MonthEnd(0)
                rows.append((fips, ag, pd.Timestamp(end).date(), tot))
    out = pd.DataFrame(rows, columns=["state", "age_group", "date", "coverage"])
    # cumulative: never decreasing within a season (survey noise can dip)
    out["coverage"] = out.groupby(["state", "age_group"]).coverage.cummax()
    out.attrs["season_used"] = used
    return out


def vaccination_days(net: Network, cov: pd.DataFrame, start: dt.date, rng: np.random.Generator,
                     default_state: str) -> np.ndarray:
    """Day (since `start`) each agent is vaccinated, or -1. Vaccinations before `start` map to day 0."""
    state = np.array([c[:2] for c in net.counties])[net.county]
    state = np.where(state == "00", default_state, state)
    ag = np.where(net.age_group < 0, 2, net.age_group)
    u = rng.random(net.n)
    out = np.full(net.n, -1, np.int16)
    season0 = dt.date(cov.date.min().year, 7, 1)
    for (st, a), g in cov.groupby(["state", "age_group"]):
        who = np.flatnonzero((state == st) & (ag == a))
        if not len(who):
            continue
        g = g.sort_values("date")
        days = np.r_[0, [(d - season0).days for d in g.date]]
        cum = np.r_[0.0, g.coverage.to_numpy()]
        hit = u[who] < cum[-1]
        # invert the piecewise-linear cumulative curve (tiny slope keeps plateaus invertible)
        when = np.interp(u[who][hit], cum + np.arange(len(cum)) * 1e-9, days)
        day = when + (season0 - start).days
        out[who[hit]] = np.clip(np.round(day), 0, np.iinfo(np.int16).max).astype(np.int16)
    return out


def prior_immunity(net: Network, p_by_age: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ag = np.where(net.age_group < 0, 2, net.age_group)
    return rng.random(net.n) < np.asarray(p_by_age)[ag]


def build(net: Network, p_by_age: np.ndarray, start: dt.date, season_start_year: int,
          default_state: str, seed: int) -> Immunity:
    rng = np.random.default_rng(seed)
    cov = coverage(season_start_year)
    return Immunity(prior_immunity(net, p_by_age, rng), vaccination_days(net, cov, start, rng, default_state))
