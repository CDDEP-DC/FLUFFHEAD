"""State health department data on where in the state the flu is.

For states without sub-state NSSP ED values, these sources give each county's flu
activity relative to its state, by week: ratio = county per-capita rate / state
per-capita rate (1 = state average). ed.py turns ratios into county activity.

    CA  CDPH respiratory virus dashboard (data.chhs.ca.gov): NHSN flu admissions
        for six Regional Public Health Office (RPHO) regions, weekly. County ->
        region map in data/state/ca_rpho_counties.csv (checked against the
        dashboard's region populations; Rural North comes out ~15% larger than
        CDPH's, so a few of its small counties may belong elsewhere). Regions
        with no admissions in the window (suppressed) get the state average.
    PA  data.pa.gov mrpb-ugjv: season-to-date lab-confirmed flu cases by county
        (cases 1-4 redacted, counted as 2.5). Only the current cumulative table is
        published, so each pull is archived in data/state/pa/; once there are
        earlier pulls, the ratio uses the cases of the last few weeks (difference
        of two pulls) instead of the whole season to date.

Ohio and Florida publish only through dashboards/PDFs that refuse scripted
requests; a file dropped in data/state/oh/ or data/state/fl/ named
manual-<week ending YYYY-MM-DD>.csv with columns fips, ratio is used for that
week and later (see load_manual).
"""
from __future__ import annotations

import datetime as dt
import io
import logging

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

STATE_DIR = C.DATA_DIR / "state"
CA_URL = ("https://data.chhs.ca.gov/dataset/6724b36f-b89e-4fa7-9cbb-290526894511/resource/"
          "00a147ba-0410-4699-9e34-fd18bbb7017d/download/rv-dashboard.csv")
PA_URL = "https://data.pa.gov/resource/mrpb-ugjv.csv"
REDACTED = 2.5          # PA redacts counts of 1-4
WINDOW_WEEKS = 3        # recent weeks the ratio is computed over


def _get(url: str, tries: int = 3, **params) -> bytes:
    import time
    import requests
    for k in range(tries):
        try:
            r = requests.get(url, params=params or None, timeout=300, headers={"User-Agent": "FLUFFHEAD"})
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def _counties(state: str) -> pd.DataFrame:
    c = pd.read_csv(C.REGIONS_DIR / "counties.csv", dtype={"fips": str, "state": str}, keep_default_na=False)
    return c[c.state == state]


# ---------------------------------------------------------------- California
def california(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """[weeks, CA county FIPS] ratio of the county's RPHO region admissions per capita to the state's."""
    raw = pd.read_csv(io.BytesIO(_get(CA_URL)))
    d = raw[raw.AGE_GRP == "All Ages"].assign(date=lambda x: pd.to_datetime(x.WEEKENDING))
    adm = d.pivot_table(index="date", columns="RPHO_REGION", values="FLU_ADM")
    pop = d.pivot_table(index="date", columns="RPHO_REGION", values="POP").ffill().bfill()
    adm = adm.rolling(WINDOW_WEEKS, min_periods=1).sum()
    adm = adm.where(adm > 0)        # small counts are suppressed (Rural North shows 0): state average instead
    rate = adm / pop
    ratio = rate.drop(columns="California").div(rate["California"], axis=0)   # region rate / state rate
    regions = pd.read_csv(STATE_DIR / "ca_rpho_counties.csv", dtype={"fips": str})
    out = pd.DataFrame({r.fips: ratio[r.rpho_region] if r.rpho_region in ratio else np.nan
                        for r in regions.itertuples()})
    out = out.where(np.isfinite(out))
    log.info("CA: RPHO admissions to %s", adm.index.max().date())
    return _align(out, weeks)


# --------------------------------------------------------------- Pennsylvania
def _pa_pull() -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(_get(PA_URL, **{"$limit": 5000})))
    end = pd.to_datetime(d.end_date).max().date()
    path = STATE_DIR / "pa" / f"{end}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        d.to_csv(path, index=False)
    return d


def _pa_cases(d: pd.DataFrame) -> pd.Series:
    """Cases by county name; redacted counts as REDACTED, and a redacted total as A + B."""
    a = d.influenza_a_case_count.fillna(REDACTED)
    b = d.influenza_b_case_count.fillna(REDACTED)
    tot = d.influenza_cases.where(d.influenza_cases.notna(), a + b)
    return pd.Series(tot.to_numpy(float), index=d.county.str.upper().str.strip())


def pennsylvania(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """[weeks, PA county FIPS] ratio of the county's recent flu cases per capita to the state's."""
    try:
        _pa_pull()
    except Exception as e:
        log.warning("PA: download failed (%s); using archived pulls", e)
    pulls = {pd.Timestamp(p.stem): pd.read_csv(p) for p in sorted((STATE_DIR / "pa").glob("????-??-??.csv"))}
    if not pulls:
        return pd.DataFrame(index=weeks)
    c = _counties("42")
    fips = dict(zip(c.name.str.replace(" County, Pennsylvania", "").str.upper(), c.fips))
    pop = c.set_index("fips")["pop"]
    rows = {}
    for end, d in pulls.items():
        cases = _pa_cases(d)
        earlier = [e for e in pulls if e <= end - pd.Timedelta(weeks=WINDOW_WEEKS)]
        if earlier:   # cases in the last WINDOW_WEEKS weeks
            cases = (cases - _pa_cases(pulls[max(earlier)]).reindex(cases.index).fillna(0)).clip(lower=0)
        cases.index = cases.index.map(fips)
        cases = cases[cases.index.notna()].reindex(pop.index).fillna(0)
        rows[end] = (cases / pop) / (cases.sum() / pop.sum())
    out = pd.DataFrame(rows).T.sort_index()
    log.info("PA: %d pulls, latest through %s", len(pulls), out.index.max().date())
    start = pd.to_datetime(next(iter(pulls.values())).start_date).min()
    return _align(out, weeks, first=start)


# ------------------------------------------------------------------- manual
def load_manual(folder: str, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    files = sorted((STATE_DIR / folder).glob("manual-????-??-??.csv"))
    if not files:
        return pd.DataFrame(index=weeks)
    rows = {pd.Timestamp(f.stem[7:]): pd.read_csv(f, dtype={"fips": str}).set_index("fips").ratio for f in files}
    return _align(pd.DataFrame(rows).T, weeks)


def _align(ratio: pd.DataFrame, weeks: pd.DatetimeIndex, first: pd.Timestamp | None = None) -> pd.DataFrame:
    """Ratios on `weeks`: each week takes the latest ratio at or before it (later weeks reuse the
    last one); weeks before the source's first week (or `first`) are left empty."""
    ratio = ratio.sort_index()
    out = ratio.reindex(ratio.index.union(weeks)).ffill().reindex(weeks)
    out[weeks < (first if first is not None else ratio.index.min())] = np.nan
    return out


SOURCES = {"06": california, "42": pennsylvania}
MANUAL = {"39": "oh", "12": "fl"}


def ratios(weeks: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """{state FIPS: [weeks, county FIPS] ratio} for every state with a working source."""
    out = {}
    for st, f in SOURCES.items():
        try:
            r = f(weeks)
        except Exception as e:     # one source failing shouldn't stop the forecast
            log.warning("state data for %s unavailable: %s", st, e)
            continue
        if r.notna().any().any():
            out[st] = r
    for st, folder in MANUAL.items():
        r = load_manual(folder, weeks)
        if r.notna().any().any():
            out[st] = r
    return out
