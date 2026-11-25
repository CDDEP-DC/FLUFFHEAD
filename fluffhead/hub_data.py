# Vendored from flusight-fm (flusight_fm/data.py, commit 4c5e8a5, CC0-1.0). Keep in sync by hand;
# FLUFFHEAD does not depend on the flusight-fm package (it needs torch).
"""Downloading and caching data.

- current hub target data and locations
- past versions ("vintages") of the hub target data, rebuilt from the hub's git
  history, used to learn how reports get revised and to run honest backtests
- long historical series (NHSN since 2020, ILINet since 1997) for fine-tuning
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
import time

import pandas as pd
import requests

from . import config as C

log = logging.getLogger(__name__)


def reference_date_for(day: dt.date) -> dt.date:
    """The Saturday ending the epiweek containing `day` (the Saturday after Wednesday's deadline)."""
    return day + dt.timedelta(days=(5 - day.weekday()) % 7)


def _get(url: str, **kwargs) -> requests.Response:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120, **kwargs)
            r.raise_for_status()
            return r
        except requests.HTTPError:
            if r.status_code == 404 or attempt == 2:
                raise
        except requests.RequestException:
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _write_atomic(path, text: str) -> None:
    """Write via a temporary file and rename, so processes reading the cache at the same time
    (parallel backtest cases each refresh locations.csv) never see it half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read_target_csv(text_or_path) -> pd.DataFrame:
    src = io.StringIO(text_or_path) if isinstance(text_or_path, str) and "\n" in text_or_path else text_or_path
    df = pd.read_csv(src, dtype={"location": str}, parse_dates=["date"])
    return df[["date", "location", "value"]]


def load_locations() -> pd.DataFrame:
    path = C.CACHE_DIR / "locations.csv"
    try:
        text = _get(C.hub_raw(C.LOCATIONS_PATH)).text
        _write_atomic(path, text)
    except requests.RequestException:
        if not path.exists():
            raise
        log.warning("Could not download locations.csv; using cached copy")
    return pd.read_csv(path, dtype={"location": str})


def load_current(key: str) -> pd.DataFrame:
    """Latest version of a target data file ("hosp" or "ed")."""
    path = C.CACHE_DIR / "current" / f"{key}.csv"
    try:
        text = _get(C.hub_raw(C.TARGET_PATHS[key])).text
        _write_atomic(path, text)
    except requests.RequestException:
        if not path.exists():
            raise
        log.warning("Could not download %s target data; using cached copy", key)
    return _read_target_csv(path)


# -------------------------------------------------------------------- vintages
def _github_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    return headers


def _commits(path: str, since: str) -> list[tuple[str, str]]:
    out, page = [], 1
    while True:
        batch = _get(f"https://api.github.com/repos/{C.HUB_REPO}/commits", headers=_github_headers(),
                     params={"path": path, "per_page": 100, "page": page, "since": f"{since}T00:00:00Z"}).json()
        out += [(c["sha"], c["commit"]["committer"]["date"]) for c in batch]
        if len(batch) < 100:
            return out
        page += 1


def _eastern_date(committed_utc: str) -> dt.date:
    return (pd.Timestamp(committed_utc) - pd.Timedelta(hours=5)).date()  # ignores DST; close enough


def availability_date(committed_utc: str) -> dt.date:
    """Reference date of the first submission that could have used data committed at this time,
    on the usual schedule (data by Wednesday). See version_dates for late weeks."""
    return reference_date_for(_eastern_date(committed_utc) + dt.timedelta(days=3))


def version_dates(committed_utc: list[str]) -> list[dt.date]:
    """Reference date for each commit time. As availability_date, except that a Thursday or Friday
    commit in a week with no commit from Sunday to Wednesday is that week's late release (e.g.
    Veterans Day 2026, when the hub posted Thursday and moved the deadline), so it is dated to that
    week's Saturday rather than the next one."""
    days = [_eastern_date(str(t)) for t in committed_utc]
    on_time = {reference_date_for(d) for d in days if d.weekday() in (6, 0, 1, 2)}
    return [reference_date_for(d) if d.weekday() in (3, 4) and reference_date_for(d) not in on_time
            else availability_date(str(t)) for d, t in zip(days, committed_utc)]


def load_versions(key: str, since: str | None = None) -> list[tuple[str, pd.Timestamp, pd.DataFrame]]:
    """Every committed version of a target data file since `since` (default VERSIONS_START):
    [(sha, commit time UTC, data)], oldest first."""
    since = since or C.VERSIONS_START
    path = C.TARGET_PATHS[key]
    d = C.CACHE_DIR / "versions" / key
    d.mkdir(parents=True, exist_ok=True)
    index = d / "commits.json"
    try:
        commits = _commits(path, since)
        _write_atomic(index, json.dumps(commits))
    except requests.RequestException as e:
        log.warning("Could not list hub commits for %s (%s); using cached list", path, e)
        commits = json.loads(index.read_text()) if index.exists() else []

    out = []
    for sha, when in sorted(commits, key=lambda c: c[1]):
        if when[:10] < since:
            continue
        f = d / f"{sha[:12]}.csv.gz"
        if not f.exists():
            try:
                _read_target_csv(_get(C.hub_raw(path, sha)).text).to_csv(f, index=False)
            except (requests.RequestException, ValueError, KeyError) as e:
                log.warning("Skipping %s version %s: %s", key, sha[:10], e)
                continue
        out.append((sha, pd.Timestamp(when), _read_target_csv(f)))
    return out


def load_vintages(key: str, since: str | None = None) -> dict[dt.date, pd.DataFrame]:
    """One version of a target data file per reference date: the last one committed before that week's deadline."""
    versions = load_versions(key, since or C.REVISION_START)
    out = dict(zip(version_dates([str(when) for _, when, _ in versions]), (df for _, _, df in versions)))
    log.info("Loaded %d versions of %s target data", len(out), key)
    return dict(sorted(out.items()))


def vintage_for(vintages: dict[dt.date, pd.DataFrame], ref_date: dt.date) -> pd.DataFrame | None:
    """The data as it stood for a given reference date, if we have that version."""
    eligible = [r for r in vintages if r <= ref_date]
    return vintages[max(eligible)] if eligible else None


# --------------------------------------------------------------------- history
def epiweek_end(epiweek: int) -> dt.date:
    """Saturday ending an MMWR week given as YYYYWW."""
    year, week = divmod(epiweek, 100)
    jan4 = dt.date(year, 1, 4)
    week1_sunday = jan4 - dt.timedelta(days=(jan4.weekday() + 1) % 7)
    return week1_sunday + dt.timedelta(weeks=week - 1, days=6)


def fetch_nhsn_history(locations: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Weekly confirmed flu admissions by jurisdiction since Aug 2020 (data.cdc.gov ua7e-t2fy)."""
    path = C.CACHE_DIR / "history" / "nhsn.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"location": str}, parse_dates=["date"])
    r = _get("https://data.cdc.gov/resource/ua7e-t2fy.csv",
             params={"$select": "weekendingdate,jurisdiction,totalconfflunewadm", "$limit": 1_000_000})
    raw = pd.read_csv(io.StringIO(r.text))
    codes = dict(zip(locations["abbreviation"], locations["location"])) | {"USA": "US"}
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["weekendingdate"]).dt.normalize(),
        "location": raw["jurisdiction"].map(codes),
        "value": pd.to_numeric(raw["totalconfflunewadm"], errors="coerce"),
    }).dropna(subset=["location"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fetch_ilinet_history(locations: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Weekly % ILI (weighted where available) since 1997 from the Delphi fluview API."""
    path = C.CACHE_DIR / "history" / "ilinet.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"])
    regions = ["nat"] + [f"hhs{i}" for i in range(1, 11)] + [
        a.lower() for a in locations["abbreviation"] if a != "US"]
    last_week = int(dt.date.today().strftime("%Y")) * 100 + 53
    params = {"epiweeks": f"199740-{last_week}"}
    if os.environ.get("DELPHI_API_KEY"):
        params["api_key"] = os.environ["DELPHI_API_KEY"]
    rows = []
    for region in regions:
        res = _get("https://api.delphi.cmu.edu/epidata/fluview/", params=params | {"regions": region}).json()
        if res.get("result") != 1:
            log.warning("No ILINet data for %s: %s", region, res.get("message"))
            continue
        for e in res["epidata"]:
            value = e.get("wili") if e.get("wili") is not None else e.get("ili")
            rows.append((epiweek_end(e["epiweek"]), region, value))
    df = pd.DataFrame(rows, columns=["date", "location", "value"])
    df["date"] = pd.to_datetime(df["date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
