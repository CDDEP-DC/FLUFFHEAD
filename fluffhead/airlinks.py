"""Each region's linkage to a variant's origin (strains.py): the share of the people its
travellers meet who come from, or have been in, the origin.

  air    BTS DB1B Market (10% sample of domestic itineraries, origin to destination with
         connections collapsed), DB1B_QUARTER: of all passengers on markets with an
         airport in the region at either end, the share whose other end is in the origin.
  drive  exp(-d / DRIVE_KM), d the distance between the regions' population centroids, for
         trips by road (San Diego, Las Vegas from Los Angeles).

link = 1 - (1 - air)(1 - drive), at least LINK_FLOOR (indirect routes through other
regions, which a region-by-region simulation doesn't follow); 1 for the origin itself.

International arrivals (asia_arrivals): a variant from abroad enters with travellers.
BTS T-100 international segments (DOT "International Report: Passengers", INTL_YEAR of
data), passengers between the region's airports and Japan (weight 1) or the rest of East
Asia (China, Hong Kong, South Korea, Taiwan: weight ASIA_OTHER_WEIGHT), halved for
arrivals, per day.
"""
from __future__ import annotations

import logging
import zipfile

import numpy as np
import pandas as pd

from . import config as C
from . import exposure as X

log = logging.getLogger(__name__)

DB1B_QUARTER = "2025_1"            # January-March: flu season travel
DB1B_URL = "https://www.transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_{q}.zip"
DB1B_DIR = C.RAW_DIR / "travel" / "db1b"
DRIVE_KM = 200.0
LINK_FLOOR = 0.02
INTL_URL = "https://data.transportation.gov/resource/xgub-n9bw.json"
INTL_FROM, INTL_TO = "2025-07-01", "2026-07-01"      # twelve months of data
INTL_PATH = C.RAW_DIR / "travel" / "intl" / "east_asia_2025-07_2026-06.json"
JAPAN_WAC = 736                                      # BTS world area codes
ASIA_OTHER_WAC = (713, 729, 778, 781)                # China, Hong Kong, South Korea, Taiwan
ASIA_OTHER_WEIGHT = 0.5


def _db1b() -> pd.DataFrame:
    """Passengers by airport pair (sampled itineraries, both directions summed)."""
    out = DB1B_DIR / f"pairs_{DB1B_QUARTER}.csv"
    if out.exists():
        return pd.read_csv(out)
    z = DB1B_DIR / f"DB1BMarket_{DB1B_QUARTER}.zip"
    if not z.exists():
        from .weather import _get
        z.parent.mkdir(parents=True, exist_ok=True)
        z.write_bytes(_get(DB1B_URL.format(q=DB1B_QUARTER), headers={"User-Agent": "Mozilla/5.0"}))
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        parts = []
        for ch in pd.read_csv(zf.open(name), usecols=["Origin", "Dest", "Passengers"], chunksize=2_000_000):
            parts.append(ch.groupby(["Origin", "Dest"]).Passengers.sum())
    pairs = pd.concat(parts).groupby(level=[0, 1]).sum().rename("pax").reset_index()
    pairs.to_csv(out, index=False)
    return pairs


def _group_of_airport() -> dict[str, str]:
    counties = pd.read_csv(C.REGIONS_DIR / "counties.csv", dtype={"fips": str, "state": str}, keep_default_na=False)
    g = counties.set_index("fips").group.to_dict()
    return {a: g.get(f, "") for a, f in X.airport_counties().items()}


def _centroids() -> pd.DataFrame:
    from .weather import _centroids as county_centroids
    counties = pd.read_csv(C.REGIONS_DIR / "counties.csv", dtype={"fips": str, "state": str}, keep_default_na=False)
    c = counties[counties.group != ""].merge(county_centroids(), on="fips")
    w = c["pop"].to_numpy(float)
    return c.assign(wlat=c.lat * w, wlon=c.lon * w, w=w).groupby("group")[["wlat", "wlon", "w"]].sum() \
        .pipe(lambda d: pd.DataFrame({"lat": d.wlat / d.w, "lon": d.wlon / d.w}))


def links(origin: str) -> pd.DataFrame:
    """Per group: air share, drive term and link to `origin` (cached)."""
    path = DB1B_DIR / f"links_{origin}_{DB1B_QUARTER}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    from .weather import _km
    pairs = _db1b()
    g = _group_of_airport()
    pairs = pairs.assign(go=pairs.Origin.map(g).fillna(""), gd=pairs.Dest.map(g).fillna(""))
    cen = _centroids()
    o_lat, o_lon = cen.loc[origin, ["lat", "lon"]]
    rows = []
    for grp in cen.index:
        touch = (pairs.go == grp) | (pairs.gd == grp)
        tot = pairs.pax[touch].sum()
        to_o = pairs.pax[touch & ((pairs.go == origin) | (pairs.gd == origin)) & ~((pairs.go == grp) & (pairs.gd == grp))].sum()
        air = to_o / tot if tot > 0 else 0.0
        d = float(_km(cen.loc[grp, "lat"], cen.loc[grp, "lon"], np.array([o_lat]), np.array([o_lon]))[0])
        drive = float(np.exp(-d / DRIVE_KM))
        link = 1.0 if grp == origin else max(LINK_FLOOR, 1 - (1 - air) * (1 - drive))
        rows.append((grp, tot, air, d, drive, link))
    t = pd.DataFrame(rows, columns=["group", "air_pax", "air", "km", "drive", "link"]).set_index("group")
    t.to_csv(path)
    return t


def link(group: str, origin: str) -> float:
    if group == origin:
        return 1.0
    t = links(origin)
    return float(t.link.get(group, LINK_FLOOR))


def _intl() -> pd.DataFrame:
    if not INTL_PATH.exists():
        import urllib.parse
        from .weather import _get
        wacs = ",".join(f"'{w}'" for w in (JAPAN_WAC,) + ASIA_OTHER_WAC)
        q = {"$select": "usg_apt,fg_wac,sum(total) as pax", "$group": "usg_apt,fg_wac", "$limit": "50000",
             "$where": f"type='Passengers' AND data_dte>='{INTL_FROM}' AND data_dte<'{INTL_TO}' AND fg_wac in({wacs})"}
        INTL_PATH.parent.mkdir(parents=True, exist_ok=True)
        INTL_PATH.write_bytes(_get(INTL_URL + "?" + urllib.parse.urlencode(q),
                                   headers={"User-Agent": "FLUFFHEAD influenza forecasting (research)"}))
    d = pd.read_json(INTL_PATH)
    return d.assign(pax=d.pax.astype(float), fg_wac=d.fg_wac.astype(int))


def asia_arrivals(group: str) -> float:
    """Weighted arrivals per day from Japan and East Asia at the region's airports (module docstring)."""
    d = _intl()
    g = _group_of_airport()
    d = d[d.usg_apt.map(g).fillna("") == group]
    w = np.where(d.fg_wac == JAPAN_WAC, 1.0, ASIA_OTHER_WEIGHT)
    days = (pd.Timestamp(INTL_TO) - pd.Timestamp(INTL_FROM)).days
    return float((w * d.pax).sum() / 2 / days)


def air_share(group: str, others) -> float:
    """Share of the region's domestic air passengers whose other end is in one of `others`."""
    pairs = _db1b()
    g = _group_of_airport()
    go, gd = pairs.Origin.map(g).fillna(""), pairs.Dest.map(g).fillna("")
    touch = (go == group) | (gd == group)
    other = set(others) - {group}
    to = touch & (go.isin(other) | gd.isin(other))
    tot = pairs.pax[touch].sum()
    return float(pairs.pax[to].sum() / tot) if tot > 0 else 0.0
