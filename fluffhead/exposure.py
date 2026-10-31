"""Who meets people from outside the region, and how community contact varies with density.

Importation. A person's daily hazard of infection from outside the region is
national incidence x (iota x travel_i + commute_i):

  travel_i   exposure to travel and visitors, normalised so the population-weighted
             US mean of a resident's value is 1 (iota, fitted with one tight prior
             for all regions, is then the average resident's share of risk met
             outside the region):
               region:  (1 - AIRPORT_WEIGHT) + AIRPORT_WEIGHT x sqrt(airport
                        enplanements per resident / US), FAA CY2025 commercial
                        service airports in the region's counties (both residents
                        flying out and visitors flying in; the square root damps
                        hubs, whose connecting passengers never leave the airport)
               person:  by age (AGE_TRAVEL), and for workers 1 + VISITOR_GAIN x the
                        share of their work tract's jobs that face visitors (LODES
                        WAC: retail, transportation, arts/entertainment,
                        accommodation and food), normalised to mean 1 in the region
  commute_i  fixed, not fitted: residents working outside the region meet
             OUT_WORK_SHARE of their risk at that workplace; people living outside
             and working inside (the network's non-residents) meet IN_COMMUTER_SHARE
             of theirs at home.

Community density. A tract's community encounter rate is multiplied by
(density / region mean)^gamma (DiseaseParams.community_density_gamma), density =
the network's residents and workers there per square mile of land (2019 Census
gazetteer, 2010 tracts as in GREASYPOP's ACS). Normalised to mean 1 over the
region's people, so it moves contact between a region's tracts without changing
the region's total (the per-region R prior sets the level).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

TRAVEL_DIR = C.RAW_DIR / "travel"
ENPLANEMENTS = TRAVEL_DIR / "cy2025-commercial-service-enplanements.xlsx"
ENPLANEMENTS_URL = ("https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/"
                    "arp-cy2025-commercial-service-enplanements.xlsx")
GAZETTEER = TRAVEL_DIR / "2019_Gaz_tracts_national.txt"
GAZETTEER_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2019_Gazetteer/2019_Gaz_tracts_national.zip"
AIRPORT_WEIGHT = 0.5
AGE_TRAVEL = np.array([0.6, 0.9, 1.2, 1.1, 0.7])   # 0-4, 5-17, 18-49, 50-64, 65+ (long-distance trips peak at working ages)
VISITOR_INDUSTRIES = ["CNS07", "CNS08", "CNS17", "CNS18"]
VISITOR_GAIN = 2.0
OUT_WORK_SHARE = 0.12       # ~ workplace share of infections in the model
IN_COMMUTER_SHARE = 0.88
DENSITY_CAP = (1 / 3, 3.0)


def _download(url: str, path) -> None:
    import io
    import zipfile
    from .weather import _get
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _get(url, headers={"User-Agent": "Mozilla/5.0"})
    if url.endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(data)).extractall(path.parent)
    else:
        path.write_bytes(data)


# ------------------------------------------------------------------ airports
def airport_index() -> pd.Series:
    """{group: region travel factor}, population-weighted mean 1 over the simulated regions."""
    path = TRAVEL_DIR / "airport_index.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0).iloc[:, 0]
    from .greasypop import ABBR
    if not ENPLANEMENTS.exists():
        _download(ENPLANEMENTS_URL, ENPLANEMENTS)
    e = pd.read_excel(ENPLANEMENTS).dropna(subset=["Locid"])
    e = e.assign(enpl=e["CY 25 Enplanements"].fillna(0))
    import json
    fips_of = {a: f for f, a in ABBR.items()}
    from .weather import _centroids, _get, _km
    cen = _centroids()
    county = {}              # airport -> county FIPS, from the ASOS station at the airport
    for abbr in ABBR.values():
        gj = C.RAW_DIR / "weather" / "stations" / f"{abbr}_ASOS.geojson"
        try:
            if not gj.exists():
                gj.parent.mkdir(parents=True, exist_ok=True)
                gj.write_bytes(_get(f"https://mesonet.agron.iastate.edu/geojson/network/{abbr}_ASOS.geojson"))
        except Exception as ex:
            log.warning("ASOS stations for %s unavailable (%s)", abbr, ex)
            continue
        for f in json.loads(gj.read_text())["features"]:
            u = f["properties"].get("ugc_county") or ""
            if len(u) == 6 and u[2] == "C" and u[:2] in fips_of:
                fips = fips_of[u[:2]] + u[3:]
            else:            # no county code (e.g. Alaska boroughs): nearest county population centroid
                lon, lat = f["geometry"]["coordinates"]
                fips = cen.fips.iloc[int(np.argmin(_km(lat, lon, cen.lat.to_numpy(), cen.lon.to_numpy())))]
            sid = f["properties"]["sid"]
            county[sid] = fips
            if len(sid) == 4:    # ICAO ids (PHNL, PANC, TJSJ) -> FAA ids
                county.setdefault(sid[1:], fips)
    e["fips"] = e.Locid.map(county)
    miss = e[e.fips.isna()].sort_values("enpl", ascending=False)
    log.info("airports: %.1f%% of enplanements matched to a county; largest unmatched: %s",
             100 * e.enpl[e.fips.notna()].sum() / e.enpl.sum(),
             ", ".join(f"{r.Locid} ({r.enpl / 1e6:.1f}M)" for r in miss.head(5).itertuples()))
    counties = pd.read_csv(C.REGIONS_DIR / "counties.csv", dtype={"fips": str, "state": str}, keep_default_na=False)
    us_rate = e.enpl.sum() / counties["pop"].sum()
    c = counties.merge(e.groupby("fips").enpl.sum().rename("enpl"), left_on="fips", right_index=True, how="left")
    g = c[c.group != ""].groupby("group").agg(pop=("pop", "sum"), enpl=("enpl", "sum"))
    a = (1 - AIRPORT_WEIGHT) + AIRPORT_WEIGHT * np.sqrt(g.enpl.fillna(0) / g["pop"] / us_rate)
    a = (a / np.average(a, weights=g["pop"])).rename("travel_factor")
    a.to_csv(path)
    return a


# ------------------------------------------------------------------ tracts
def visitor_share(tracts: np.ndarray) -> np.ndarray:
    """Share of each tract's jobs in visitor-facing industries (NaN where no LODES jobs)."""
    from .greasypop import ABBR
    out = pd.Series(np.nan, index=tracts)
    for s in sorted({t[:2] for t in tracts}):
        path = TRAVEL_DIR / f"visitor_jobs_{s}.csv"
        if not path.exists():
            files = sorted((C.RAW_DIR / "greasypop" / "work").glob(f"{ABBR[s].lower()}_wac_S000_JT01_*.csv.gz"))
            if not files:
                continue
            w = pd.read_csv(files[-1], usecols=["w_geocode", "C000"] + VISITOR_INDUSTRIES, dtype={"w_geocode": str})
            width = 15 if w.w_geocode.str.len().max() > 12 else 12     # blocks, or block groups
            w["tract"] = w.w_geocode.str.zfill(width).str[:11]
            t = w.groupby("tract")[["C000"] + VISITOR_INDUSTRIES].sum()
            pd.DataFrame({"share": t[VISITOR_INDUSTRIES].sum(1) / t.C000.clip(lower=1)}).to_csv(path)
        sh = pd.read_csv(path, dtype={"tract": str}, index_col=0).share
        out.update(sh.reindex(out.index))
    return out.to_numpy()


def land_sqmi(tracts: np.ndarray) -> np.ndarray:
    if not GAZETTEER.exists():
        _download(GAZETTEER_URL, GAZETTEER)
    g = pd.read_csv(GAZETTEER, sep="\t", dtype={"GEOID": str})
    g.columns = g.columns.str.strip()
    return g.set_index("GEOID").ALAND_SQMI.reindex(tracts).to_numpy()


def community_multiplier(net, gamma: float) -> np.ndarray:
    """[tracts] community encounter rate multiplier, mean 1 over the region's tract memberships."""
    size = np.diff(net.tract_ptr).astype(float)
    if gamma == 0 or net.tracts is None:
        return np.ones(len(size))
    dens = size / np.maximum(land_sqmi(net.tracts), 0.05)
    ok = np.isfinite(dens) & (dens > 0)
    ref = np.exp(np.average(np.log(dens[ok]), weights=size[ok])) if ok.any() else 1.0
    m = np.where(ok, (dens / ref) ** gamma, 1.0)
    m = np.clip(m / np.average(m, weights=size), *DENSITY_CAP)
    return m / np.average(m, weights=size)


def import_weights(group: str, net) -> tuple[np.ndarray, np.ndarray]:
    """Per-person (travel, commute) importation weights; see the module docstring."""
    try:
        a_region = float(airport_index().get(group, 1.0))
    except Exception as e:
        log.warning("%s: airport data unavailable (%s); travel factor 1", group, e)
        a_region = 1.0
    ag = np.where(net.age_group < 0, 2, net.age_group)
    m = AGE_TRAVEL[ag].copy()
    worker = (net.work_tract >= 0) & net.resident
    if net.tracts is not None:
        v = visitor_share(net.tracts)
        vw = v[net.work_tract[worker]]
        m[worker] *= 1 + VISITOR_GAIN * np.nan_to_num(vw, nan=np.nanmean(v) if np.isfinite(v).any() else 0.0)
    travel = np.where(net.resident, m, 0.0)
    travel *= a_region / travel[net.resident].mean()
    commute = np.where(net.out_worker & net.resident, OUT_WORK_SHARE, 0.0)
    commute[~net.resident] = IN_COMMUTER_SHARE
    log.info("%s: import exposure: travel factor %.2f, worker visitor-job multiplier mean %.2f, "
             "%d out-commuters, %d in-commuters", group, a_region, m[worker].mean() / AGE_TRAVEL[ag[worker]].mean()
             if worker.any() else 1.0, int((net.out_worker & net.resident).sum()), int((~net.resident).sum()))
    return travel, commute
