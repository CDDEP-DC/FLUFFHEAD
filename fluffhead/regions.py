"""Choosing the regions to simulate.

Each county belongs to a unit: its CSA, or its CBSA when the CBSA is not part
of a CSA. Counties outside every CBSA belong to no unit and are never simulated.

1. Take the N_TOP_REGIONS largest units.
   States in SCALED_STATES (Puerto Rico) are never simulated; they are scaled
   from another group.
2. For each state (and DC) whose population is less than MIN_STATE_COVERAGE
   covered, add its largest remaining unit, so every state has dynamics of its own
   to scale from.
3. Merge small units that are strongly linked by commuting into one simulation
   group (one GREASYPOP network, one model run).

Outputs, under data/regions/:
    counties.csv    every county: state, population, unit, group ("" if not simulated)
    groups.csv      one row per simulation group
    coupling.csv    commuters between groups (and to/from the unsimulated rest of the US)
    <group>/geo.json   GREASYPOP region specification
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C

log = logging.getLogger(__name__)

GEO_RAW = C.RAW_DIR / "geo"
OUTSIDE = "outside"


def census_key() -> str:
    """The Census API key, from the environment or ~/.Renviron (never stored in the repo)."""
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        renv = Path.home() / ".Renviron"
        if renv.exists():
            for line in renv.read_text().splitlines():
                if line.startswith("CENSUS_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise RuntimeError("set CENSUS_API_KEY (environment or ~/.Renviron)")
    return key


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def fetch() -> None:
    _download(C.DELINEATION_URL, GEO_RAW / "list1_2020.xls")
    _download(C.COUNTY_POP_URL, GEO_RAW / "co-est2019-alldata.csv")
    _download(C.COMMUTING_URL, GEO_RAW / "commuting_flows_2015.xlsx")
    pr = GEO_RAW / "pr_municipio_pop_acs2019.csv"
    if not pr.exists():
        # the county estimates file has no Puerto Rico; use the 2019 ACS total population
        r = requests.get("https://api.census.gov/data/2019/acs/acs5",
                         params={"get": "NAME,B01003_001E", "for": "county:*", "in": "state:72",
                                 "key": census_key()}, timeout=120)
        r.raise_for_status()
        rows = r.json()
        pd.DataFrame(rows[1:], columns=rows[0]).to_csv(pr, index=False)


def load_counties() -> pd.DataFrame:
    """Every county with its state, 2019 population and unit (CSA or stand-alone CBSA)."""
    pop = pd.read_csv(GEO_RAW / "co-est2019-alldata.csv", dtype=str, encoding="latin-1")
    pop = pop[pop.SUMLEV == "050"]
    pop = pd.DataFrame({"fips": pop.STATE + pop.COUNTY, "state": pop.STATE,
                        "name": pop.CTYNAME + ", " + pop.STNAME,
                        "pop": pop.POPESTIMATE2019.astype(int)})
    pr = pd.read_csv(GEO_RAW / "pr_municipio_pop_acs2019.csv", dtype=str)
    pr = pd.DataFrame({"fips": pr.state + pr.county, "state": pr.state, "name": pr.NAME,
                       "pop": pr.B01003_001E.astype(int)})
    pop = pd.concat([pop, pr], ignore_index=True)

    d = pd.read_excel(GEO_RAW / "list1_2020.xls", header=2, dtype=str)
    d = d.dropna(subset=["FIPS County Code"])
    d = pd.DataFrame({
        "fips": d["FIPS State Code"] + d["FIPS County Code"],
        "unit": np.where(d["CSA Code"].notna(), "csa" + d["CSA Code"].fillna(""), "cbsa" + d["CBSA Code"]),
        "unit_name": d["CSA Title"].fillna(d["CBSA Title"]),
    })
    out = pop.merge(d, on="fips", how="left")
    out["unit"] = out["unit"].fillna("")
    out["unit_name"] = out["unit_name"].fillna("")
    return out


def load_commuting() -> pd.DataFrame:
    """Residence county -> workplace county worker flows (ACS 2011-2015)."""
    x = pd.read_excel(GEO_RAW / "commuting_flows_2015.xlsx", header=None, skiprows=7, dtype=str,
                      usecols=[0, 1, 4, 5, 8])
    x.columns = ["rs", "rc", "ws", "wc", "workers"]
    x = x.dropna(subset=["rc", "wc", "workers"])
    # foreign workplaces have a non-numeric or out-of-range state code; keep US + PR only
    x = x[x.ws.str.fullmatch(r"0\d\d")]
    return pd.DataFrame({"res": x.rs + x.rc, "work": x.ws.str[1:] + x.wc,
                         "workers": pd.to_numeric(x.workers, errors="coerce").fillna(0).astype(int)})


def select_units(counties: pd.DataFrame) -> pd.DataFrame:
    """Units to simulate, with the reason each was chosen."""
    eligible = counties[(counties.unit != "") & ~counties.state.isin(C.SCALED_STATES)]
    units = (eligible.groupby(["unit", "unit_name"], as_index=False)["pop"].sum()
             .sort_values("pop", ascending=False, ignore_index=True))
    chosen = {u: "top" for u in units.unit[:C.N_TOP_REGIONS]}

    state_pop = counties.groupby("state")["pop"].sum()
    covered = counties[counties.unit.isin(chosen)].groupby("state")["pop"].sum().reindex(state_pop.index).fillna(0)
    for st in (covered / state_pop)[lambda s: s < C.MIN_STATE_COVERAGE].index.difference(list(C.SCALED_STATES)):
        # largest unchosen unit by its population inside this state
        cand = (counties[(counties.state == st) & (counties.unit != "") & ~counties.unit.isin(chosen)]
                .groupby("unit")["pop"].sum())
        if len(cand):
            chosen[cand.idxmax()] = f"state {st}"
    units = units[units.unit.isin(chosen)].copy()
    units["reason"] = units.unit.map(chosen)
    return units.reset_index(drop=True)


def merge_units(units: pd.DataFrame, counties: pd.DataFrame, flows: pd.DataFrame) -> dict[str, str]:
    """Unit -> group id. Small, strongly commuting-linked units share a group."""
    cu = counties.set_index("fips")["unit"]
    f = flows.assign(ru=flows.res.map(cu), wu=flows.work.map(cu)).dropna(subset=["ru", "wu"])
    resident_workers = f.groupby("ru")["workers"].sum()
    between = f[f.ru != f.wu].groupby(["ru", "wu"])["workers"].sum()

    pop = units.set_index("unit")["pop"]
    small = [u for u in units.unit if pop[u] < C.MERGE_MAX_POP]
    links = []
    for i, a in enumerate(small):
        for b in small[i + 1:]:
            w = between.get((a, b), 0) + between.get((b, a), 0)
            if w:
                link = w / min(resident_workers.get(a, 1), resident_workers.get(b, 1))
                if link >= C.MERGE_MIN_LINK:
                    links.append((link, a, b))

    # union-find, strongest links first, capped by group population
    parent = {u: u for u in units.unit}
    gpop = pop.to_dict()

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for link, a, b in sorted(links, reverse=True):
        ra, rb = find(a), find(b)
        if ra != rb and gpop[ra] + gpop[rb] <= C.MERGE_GROUP_MAX_POP:
            if gpop[rb] > gpop[ra]:
                ra, rb = rb, ra
            parent[rb] = ra   # the larger unit names the group
            gpop[ra] += gpop[rb]
    return {u: find(u) for u in units.unit}


def build(out_dir: Path = C.REGIONS_DIR) -> pd.DataFrame:
    fetch()
    counties = load_counties()
    flows = load_commuting()
    units = select_units(counties)
    group_of = merge_units(units, counties, flows)

    counties["group"] = counties.unit.map(group_of).fillna("")
    names = units.set_index("unit")["unit_name"]
    reasons = units.set_index("unit")["reason"]

    # commuters between groups; everything unsimulated is "outside"
    g = counties.set_index("fips")["group"].replace("", OUTSIDE)
    f = flows.assign(rg=flows.res.map(g).fillna(OUTSIDE), wg=flows.work.map(g).fillna(OUTSIDE))
    coupling = f.groupby(["rg", "wg"], as_index=False)["workers"].sum()
    coupling.columns = ["home_group", "work_group", "workers"]

    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for gid, members in counties[counties.group != ""].groupby("group"):
        member_units = sorted(set(members.unit), key=lambda u: -members[members.unit == u]["pop"].sum())
        fips = sorted(members.fips)
        main_states = sorted(set(members.state))
        # states supplying or receiving a non-trivial share of this group's workers
        gf = f[(f.rg == gid) | (f.wg == gid)]
        other = gf.res.where(gf.rg != gid, gf.work).str[:2]
        share = gf.groupby(other)["workers"].sum() / max(gf["workers"].sum(), 1)
        commute_states = sorted(set(main_states) | set(share[share >= C.COMMUTE_STATE_MIN_SHARE].index))
        geo = {"geos": fips, "commute_states": commute_states, "use_pums": main_states}
        (out_dir / gid).mkdir(exist_ok=True)
        (out_dir / gid / "geo.json").write_text(json.dumps(geo, indent=2) + "\n")
        rows.append({"group": gid, "name": " + ".join(names[u] for u in member_units),
                     "pop": int(members["pop"].sum()), "n_units": len(member_units),
                     "n_counties": len(fips), "states": " ".join(main_states),
                     "commute_states": " ".join(commute_states),
                     "units": " ".join(member_units),
                     "reason": " ".join(sorted(set(reasons[u] for u in member_units)))})

    groups = pd.DataFrame(rows).sort_values("pop", ascending=False, ignore_index=True)
    groups.to_csv(out_dir / "groups.csv", index=False)
    counties.sort_values("fips").to_csv(out_dir / "counties.csv", index=False)
    coupling.to_csv(out_dir / "coupling.csv", index=False)
    return groups


def coverage(counties: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of each state's population inside a simulated group."""
    if counties is None:
        counties = pd.read_csv(C.REGIONS_DIR / "counties.csv", dtype={"fips": str, "state": str},
                               keep_default_na=False)
    tot = counties.groupby("state")["pop"].sum()
    sim = counties[counties.group != ""].groupby("state")["pop"].sum().reindex(tot.index).fillna(0)
    return pd.DataFrame({"pop": tot, "simulated": sim.astype(int), "coverage": sim / tot})
