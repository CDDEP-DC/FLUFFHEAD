"""Absolute humidity by region: observed, forecast, and climatology.

Transmission is modulated by absolute humidity as in Shaman et al. (2010, PLoS
Biol 8:e1000316, Eq. 4): R0(t) = exp(-180 q(t) + log(R0max - R0min)) + R0min,
q specific humidity in kg/kg, i.e. relative transmissibility

    h(t) = rho + (1 - rho) exp(-180 q(t)),     rho = R0min / R0max   (h = 1 at q = 0).

Barreca & Shimshack (2012, AJE 176:S114) find the mortality effect concentrated
below ~6 g/kg; the exponential is nearly flat above ~10 g/kg, which agrees.

Observations: NWS/FAA ASOS stations through the Iowa Environmental Mesonet
archive (daily max and min dew point; NCEI's GSOD, LCD and Global Hourly
products stopped updating in August 2025, and NCEP-NCAR reanalysis in March
2026). Each county's population-weighted centroid (Census 2020) takes its
nearest long-record stations (inverse-distance weights); a region's daily q is
the population-weighted mean over its counties. Specific humidity from dew
point and the station's standard-atmosphere pressure (elevation).

Future days: the NWS gridpoint forecast (about 7 days) at the region's
population centroid; after that the region's 2011-2025 daily climatology plus
a historical year's anomaly (drawn per trajectory, so weather is part of the
forecast uncertainty), with the last known anomaly fading over ANOMALY_DAYS.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

WEATHER_DIR = C.RAW_DIR / "weather"
IEM = "https://mesonet.agron.iastate.edu"
CENPOP_URL = "https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt"
NWS = "https://api.weather.gov"
USER_AGENT = "FLUFFHEAD influenza forecasting (research)"
CLIM_YEARS = (2011, 2025)
MAX_KM = 80.0            # stations used for a county: up to N_NEAR within this (else the nearest)
N_NEAR = 2
ANOMALY_DAYS = 5.0       # e-folding of the last known humidity anomaly
SHAMAN_A = -180.0        # per kg/kg


def factor(q_gkg: np.ndarray, rho: float) -> np.ndarray:
    """Relative transmissibility in (0, 1] from specific humidity in g/kg."""
    return rho + (1 - rho) * np.exp(SHAMAN_A * np.asarray(q_gkg) / 1000)


def specific_humidity(dewpoint_c, elevation_m) -> np.ndarray:
    """g/kg, from dew point (deg C) and standard-atmosphere pressure at the elevation."""
    p = 1013.25 * (1 - 2.25577e-5 * np.asarray(elevation_m, float)) ** 5.25588
    e = 6.112 * np.exp(17.67 * np.asarray(dewpoint_c) / (np.asarray(dewpoint_c) + 243.5))
    return 1000 * 0.622 * e / (p - 0.378 * e)


def _get(url: str, params=None, tries: int = 3, headers=None) -> bytes:
    import time
    import requests
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=300, headers={"User-Agent": USER_AGENT, **(headers or {})})
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


# ------------------------------------------------------------------ stations
def _centroids() -> pd.DataFrame:
    path = WEATHER_DIR / "cenpop2020_county.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(CENPOP_URL))
    c = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    return pd.DataFrame({"fips": c.STATEFP + c.COUNTYFP, "lat": c.LATITUDE.astype(float),
                         "lon": c.LONGITUDE.astype(float)})


def _stations(abbrs) -> pd.DataFrame:
    rows = []
    for a in sorted(set(abbrs)):
        path = WEATHER_DIR / "stations" / f"{a}_ASOS.geojson"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_get(f"{IEM}/geojson/network/{a}_ASOS.geojson"))
        for f in json.loads(path.read_text())["features"]:
            p = f["properties"]
            rows.append(dict(sid=p["sid"], network=p["network"], elevation=p["elevation"], begin=p["archive_begin"],
                             end=p["archive_end"], online=p["online"], lon=f["geometry"]["coordinates"][0],
                             lat=f["geometry"]["coordinates"][1]))
    s = pd.DataFrame(rows)
    ok = s.online & s.end.isna() & (pd.to_datetime(s.begin) <= pd.Timestamp(CLIM_YEARS[0], 7, 1))
    return s[ok].reset_index(drop=True)


def _km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def station_weights(group: str) -> pd.DataFrame:
    """[station, network, elevation, weight]: population x inverse-distance weights for a region."""
    from .greasypop import ABBR
    counties = pd.read_csv(C.REGIONS_DIR / "counties.csv", dtype={"fips": str, "state": str}, keep_default_na=False)
    mine = counties[counties.group == group].merge(_centroids(), on="fips", how="left")
    near_states = {ABBR[s] for s in mine.state}
    st = _stations(near_states)
    rows = []
    for c in mine.itertuples():
        d = _km(c.lat, c.lon, st.lat.to_numpy(), st.lon.to_numpy())
        order = np.argsort(d)
        pick = [i for i in order[:N_NEAR] if d[i] <= MAX_KM] or [order[0]]
        w = 1 / np.maximum(d[pick], 5.0)
        for i, wi in zip(pick, w / w.sum()):
            rows.append((st.sid[i], st.network[i], st.elevation[i], c.pop * wi))
    out = pd.DataFrame(rows, columns=["sid", "network", "elevation", "weight"])
    out = out.groupby(["sid", "network", "elevation"], as_index=False).weight.sum()
    out["weight"] /= out.weight.sum()
    return out


# ------------------------------------------------------------------- series
def _read(src) -> pd.DataFrame:
    # station ids such as INF, NAN or 0E0 must stay text
    return pd.read_csv(src, dtype={"station": str}, keep_default_na=False, na_values=["None", "M", ""])


def _daily(network: str, sids: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    params = [("network", network)] + [("stations", s) for s in sids] + [
        ("year1", start.year), ("month1", start.month), ("day1", start.day),
        ("year2", end.year), ("month2", end.month), ("day2", end.day),
        ("var", "max_dewpoint_f"), ("var", "min_dewpoint_f"), ("format", "csv")]
    d = _read(io.BytesIO(_get(f"{IEM}/cgi-bin/request/daily.py", params)))
    return d[["station", "day", "max_dewpoint_f", "min_dewpoint_f"]]


def station_q(stations: pd.DataFrame, today: dt.date | None = None) -> pd.DataFrame:
    """[days, station] daily mean specific humidity (g/kg) from 2011 to yesterday (cached)."""
    today = today or dt.date.today()
    hist_end = dt.date(today.year - 1, 12, 31)
    frames = []
    for net, grp in stations.groupby("network"):
        need = list(grp.sid)
        hist = WEATHER_DIR / "daily"
        hist.mkdir(parents=True, exist_ok=True)
        missing = [s for s in need if not (hist / f"{net}-{s}-{CLIM_YEARS[0]}-{hist_end.year}.csv").exists()]
        for k in range(0, len(missing), 10):
            d = _daily(net, missing[k:k + 10], dt.date(CLIM_YEARS[0], 1, 1), hist_end)
            for s in missing[k:k + 10]:
                d[d.station == s].to_csv(hist / f"{net}-{s}-{CLIM_YEARS[0]}-{hist_end.year}.csv", index=False)
        recent = C.CACHE_DIR / "weather" / f"{net}-{today}-{'-'.join(sorted(need))[:120]}.csv"
        if not recent.exists():
            recent.parent.mkdir(parents=True, exist_ok=True)
            _daily(net, need, dt.date(today.year, 1, 1), today - dt.timedelta(days=1)).to_csv(recent, index=False)
        parts = [_read(hist / f"{net}-{s}-{CLIM_YEARS[0]}-{hist_end.year}.csv") for s in need]
        frames += parts + [_read(recent)]
    d = pd.concat(frames, ignore_index=True)
    elev = stations.set_index("sid").elevation
    td = lambda f: (f - 32) * 5 / 9
    q = (specific_humidity(td(d.max_dewpoint_f), elev[d.station].to_numpy())
         + specific_humidity(td(d.min_dewpoint_f), elev[d.station].to_numpy())) / 2
    d = d.assign(q=q, day=pd.to_datetime(d.day))
    q = d.pivot_table(index="day", columns="station", values="q")
    q = q.where((q > 0.1) & (q < 30))
    return q.reindex(pd.date_range(q.index.min(), today - dt.timedelta(days=1)))


def _climatology(q: pd.Series) -> pd.Series:
    """Mean by day of year (1-366), smoothed with a circular 31-day window."""
    yrs = q[(q.index.year >= CLIM_YEARS[0]) & (q.index.year <= CLIM_YEARS[1])]
    m = yrs.groupby(yrs.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    ext = np.concatenate([m.to_numpy()[-15:], m.to_numpy(), m.to_numpy()[:15]])
    sm = np.convolve(ext, np.ones(31) / 31, mode="valid")
    return pd.Series(sm, index=m.index)


def _nws_forecast(lat: float, lon: float) -> pd.Series:
    """Daily mean specific humidity (g/kg) from the NWS gridpoint forecast (UTC days)."""
    pt = json.loads(_get(f"{NWS}/points/{lat:.4f},{lon:.4f}", headers={"Accept": "application/geo+json"}))
    grid = json.loads(_get(pt["properties"]["forecastGridData"], headers={"Accept": "application/geo+json"}))
    elev = grid["properties"]["elevation"]["value"]
    hours = []
    for v in grid["properties"]["dewpoint"]["values"]:
        t0, dur = v["validTime"].split("/")
        n = pd.Timedelta(dur.replace("PT", "").lower() if dur.startswith("PT") else "P" + dur).total_seconds() // 3600
        n = max(int(n), 1)
        for h in range(n):
            hours.append((pd.Timestamp(t0) + pd.Timedelta(hours=h), v["value"]))
    s = pd.DataFrame(hours, columns=["t", "td"]).dropna()
    s["q"] = specific_humidity(s.td.to_numpy(), elev)
    day = s.t.dt.tz_convert(None).dt.normalize()
    out = s.groupby(day).q.agg(["mean", "size"])
    return out["mean"][out["size"] >= 18]


@dataclass
class Humidity:
    """A region's daily specific humidity (g/kg) on the simulation days."""
    dates: pd.DatetimeIndex
    known: np.ndarray           # observed (and forecast) q; NaN where unknown
    last_obs: pd.Timestamp      # last observed day
    clim: np.ndarray            # climatology on `dates`
    anomalies: np.ndarray       # [years, dates] historical anomalies on the same calendar days
    years: list

    def path(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Observed/forecast where known, then climatology plus the fading last anomaly; with `rng`, plus
        a random historical year's anomaly (weighted in as the last anomaly fades)."""
        q = self.known.copy()
        k = np.flatnonzero(~np.isnan(q))
        last = k.max() if len(k) else -1
        a_last = q[last] - self.clim[last] if last >= 0 else 0.0
        after = np.arange(len(q)) > last
        lag = np.arange(len(q)) - last
        fade = np.exp(-np.maximum(lag, 0) / ANOMALY_DAYS)
        hist = self.anomalies[rng.integers(len(self.anomalies))] if rng is not None and len(self.anomalies) else 0.0
        fill = self.clim + fade * a_last + (1 - fade) * hist
        q[after] = fill[after]
        q[np.isnan(q)] = self.clim[np.isnan(q)]       # gaps before the last known day
        return np.clip(q, 0.1, None)


def region_humidity(group: str, start: dt.date, days: int, today: dt.date | None = None,
                    forecast: bool = True, as_of: dt.date | None = None) -> Humidity:
    """`as_of`: backtests; the weather stops there, as it would have then (no NWS forecast)."""
    today = today or dt.date.today()
    st = station_weights(group)
    q = station_q(st, today)
    w = st.set_index("sid").weight.reindex(q.columns).fillna(0)
    # weighted mean over the stations reporting that day (if they carry at least half the weight)
    present = q.notna().mul(w, axis=1)
    reg = (q.fillna(0).mul(w, axis=1).sum(1) / present.sum(1)).where(present.sum(1) > 0.5)
    clim_doy = _climatology(reg)
    dates = pd.date_range(start, periods=days)
    clim = clim_doy.reindex(dates.dayofyear).to_numpy()
    known = reg.reindex(dates).to_numpy().copy()
    if as_of is not None:
        reg_known = reg[reg.index <= pd.Timestamp(as_of)]
        known = reg_known.reindex(dates).to_numpy().copy()
        last_obs = reg_known.dropna().index.max()
        forecast = False
    else:
        last_obs = reg.dropna().index.max()
    if forecast:
        counties = pd.read_csv(C.REGIONS_DIR / "counties.csv", dtype={"fips": str}, keep_default_na=False)
        cen = counties[counties.group == group].merge(_centroids(), on="fips")
        lat, lon = np.average(cen.lat, weights=cen["pop"]), np.average(cen.lon, weights=cen["pop"])
        try:
            fc = _nws_forecast(lat, lon)
            fc = fc[fc.index > last_obs]
            idx = dates.get_indexer(fc.index)
            known[idx[idx >= 0]] = fc.to_numpy()[idx >= 0]
            log.info("%s: NWS humidity forecast %s to %s", group, fc.index.min().date(), fc.index.max().date())
        except Exception as e:
            log.warning("%s: NWS forecast unavailable (%s); climatology after %s", group, e, last_obs.date())
    years, anoms = [], []
    for y in range(CLIM_YEARS[0], CLIM_YEARS[1] + 1):
        shifted = dates - pd.DateOffset(years=start.year - y)
        a = reg.reindex(shifted).to_numpy() - clim
        if np.isfinite(a).mean() > 0.9:
            years.append(y)
            anoms.append(np.nan_to_num(a))
    log.info("%s: humidity from %d stations to %s; %d historical years", group, len(st), last_obs.date(), len(years))
    return Humidity(dates, known, last_obs, clim, np.array(anoms), years)


def prefetch(groups: list[str]) -> None:
    """Download station data for all regions (sequentially; workers then read the cache)."""
    for g in groups:
        try:
            station_q(station_weights(g))
        except Exception as e:
            log.warning("%s: humidity prefetch failed (%s)", g, e)
