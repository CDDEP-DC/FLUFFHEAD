"""FLUFFHEADMetro: forecasts for the Flu MetroCast hub (reichlab/flu-metrocast).

The hub scores sub-state locations: NSSP Health Service Areas (HSAs) and health-department
regions (NYC, North Carolina, Utah, California), plus their states. Run with
FLUFFHEAD_VARIANT=local (config.py), which points the model at data/regions-local/ and writes to
forecasts-local/ and runs-local/.

Regions (`fluffhead local-regions`, this module's build()):
  - each hub location's counties: NSSP's HSA-county crosswalk (data.cdc.gov rdmq-nq56) for HSAs,
    the county names in the hub's locations.csv for the other region types;
  - hub locations strongly linked by commuting share one simulation group (one network), as in
    regions.merge_units: link = commuters between the two / resident workers of the smaller, at
    least MERGE_MIN_LINK, groups up to MERGE_GROUP_MAX_POP;
  - every hub location is a piece of its group, so the model's output is kept per location.
    In FLUFFHEAD a piece is the part of a group in one state; here it is one hub location (always
    within one state). Each piece is fitted to its state's admissions x its share of the state's
    flu activity (ED-weighted, ed.py), as FLUFFHEAD's pieces are.

Outputs, under data/regions-local/ (same files as regions.py writes, plus):
    counties.csv    every county, with `piece` (hub location) for simulated ones
    locations.csv   hub locations: state, population, group, number of counties
"""
from __future__ import annotations

import io
import json
import logging

import numpy as np
import pandas as pd
import requests

from . import config as C
from . import regions as R

log = logging.getLogger(__name__)

HUB_REPO = "reichlab/flu-metrocast"
HUB_LOCATIONS = f"https://raw.githubusercontent.com/{HUB_REPO}/main/auxiliary-data/locations.csv"
NSSP_URL = "https://data.cdc.gov/resource/rdmq-nq56.csv"
RAW = C.RAW_DIR / "metrocast"
USER_AGENT = "FLUFFHEAD influenza forecasting (research)"

MERGE_MIN_LINK = 0.05          # commuters between two hub locations / resident workers of the smaller
MERGE_GROUP_MAX_POP = 15_000_000


def _get(url: str, **params) -> bytes:
    r = requests.get(url, params=params or None, timeout=300, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.content


def hub_locations(refresh: bool = False) -> pd.DataFrame:
    path = RAW / "locations.csv"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(HUB_LOCATIONS))
    return pd.read_csv(path, dtype=str)


def hsa_crosswalk(refresh: bool = False) -> pd.DataFrame:
    """NSSP's HSA of every county: state name, hsa_nci_id, county FIPS."""
    path = RAW / "hsa_counties.csv"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        last = pd.read_csv(io.BytesIO(_get(NSSP_URL, **{"$select": "max(week_end) as w"}))).w.iloc[0]
        path.write_bytes(_get(NSSP_URL, **{
            "$select": "geography,hsa_nci_id,fips,county,hsa",
            "$where": f"trend_source='HSA' AND week_end='{last[:10]}T00:00:00'",
            "$limit": "10000"}))
    x = pd.read_csv(path, dtype=str)
    return x.assign(fips=x.fips.str.zfill(5))


def location_counties(locs: pd.DataFrame, counties: pd.DataFrame) -> pd.DataFrame:
    """[location, fips] for every sub-state hub location."""
    hsa = hsa_crosswalk()
    by_name = counties.assign(cname=counties.name.str.split(",").str[0],
                              stname=counties.name.str.split(", ").str[-1])
    rows = []
    for r in locs.itertuples():
        if r.original_location_code == "All":
            continue
        if r.location_type == "hsa_nci_id":
            f = hsa[(hsa.hsa_nci_id == r.original_location_code) & (hsa.geography == r.state)].fips.tolist()
        else:                            # NYC, NC, Utah, California: named counties
            s = by_name[by_name.stname == r.state]
            f = []
            for n in (x.strip() for x in r.hsa_counties.split(",")):
                m = s[s.cname.isin([n + " County", n + " Parish", n])]
                if len(m) != 1:
                    raise ValueError(f"{r.location}: county {n!r} matched {len(m)} counties")
                f += m.fips.tolist()
        if not f:
            raise ValueError(f"{r.location}: no counties")
        rows += [(r.location, x) for x in f]
    out = pd.DataFrame(rows, columns=["location", "fips"])
    gone = out[~out.fips.isin(counties.fips)]
    if len(gone):   # e.g. Bedford city, VA (51515), merged into Bedford County in 2013
        log.info("counties not in the 2019 county list (dropped): %s", ", ".join(gone.fips))
    dup = out[out.fips.duplicated(keep=False)]
    if len(dup):
        raise ValueError(f"counties in more than one hub location: {dup.to_dict('records')}")
    return out[out.fips.isin(counties.fips)]


def merge_locations(pop: pd.Series, lc: pd.DataFrame, flows: pd.DataFrame) -> dict[str, str]:
    """Hub location -> group (named after its largest location)."""
    loc = lc.set_index("fips").location
    f = flows.assign(a=flows.res.map(loc), b=flows.work.map(loc))
    resident = f.dropna(subset=["a"]).groupby("a").workers.sum()
    between = f.dropna(subset=["a", "b"]).query("a != b").groupby(["a", "b"]).workers.sum()
    names = sorted(pop.index)
    links = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            w = between.get((a, b), 0) + between.get((b, a), 0)
            if w:
                links.append((w / min(resident.get(a, 1), resident.get(b, 1)), a, b))
    parent, gpop = {u: u for u in names}, pop.to_dict()

    def find(u):
        while parent[u] != u:
            u = parent[u]
        return u

    for link, a, b in sorted(links, reverse=True):
        if link < MERGE_MIN_LINK:
            break
        ra, rb = find(a), find(b)
        if ra != rb and gpop[ra] + gpop[rb] <= MERGE_GROUP_MAX_POP:
            if gpop[rb] > gpop[ra]:
                ra, rb = rb, ra
            parent[rb] = ra
            gpop[ra] += gpop[rb]
    return {u: "loc-" + find(u) for u in names}


def build(out_dir=None) -> pd.DataFrame:
    """Write data/regions-local/ (see the module docstring)."""
    out_dir = out_dir or C.DATA_DIR / "regions-local"
    R.fetch()
    counties = R.load_counties()
    flows = R.load_commuting()
    locs = hub_locations()
    lc = location_counties(locs, counties)
    counties = counties.merge(lc.rename(columns={"location": "piece"}), on="fips", how="left")
    counties["piece"] = counties.piece.fillna("")
    pop = counties[counties.piece != ""].groupby("piece")["pop"].sum()
    group_of = merge_locations(pop, lc, flows)
    counties["group"] = counties.piece.map(group_of).fillna("")

    g = counties.set_index("fips")["group"].replace("", R.OUTSIDE)
    f = flows.assign(rg=flows.res.map(g).fillna(R.OUTSIDE), wg=flows.work.map(g).fillna(R.OUTSIDE))
    coupling = f.groupby(["rg", "wg"], as_index=False)["workers"].sum()
    coupling.columns = ["home_group", "work_group", "workers"]

    names = locs.set_index("location").location_name
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for gid, members in counties[counties.group != ""].groupby("group"):
        pieces = sorted(set(members.piece), key=lambda p: -members[members.piece == p]["pop"].sum())
        fips = sorted(members.fips)
        main_states = sorted(set(members.state))
        gf = f[(f.rg == gid) | (f.wg == gid)]
        other = gf.res.where(gf.rg != gid, gf.work).str[:2]
        share = gf.groupby(other)["workers"].sum() / max(gf["workers"].sum(), 1)
        commute_states = sorted(set(main_states) | set(share[share >= C.COMMUTE_STATE_MIN_SHARE].index))
        geo = {"geos": fips, "commute_states": commute_states, "use_pums": main_states}
        (out_dir / gid).mkdir(exist_ok=True)
        (out_dir / gid / "geo.json").write_text(json.dumps(geo, indent=2) + "\n")
        rows.append({"group": gid, "name": " + ".join(names[p] for p in pieces),
                     "pop": int(members["pop"].sum()), "n_units": len(pieces), "n_counties": len(fips),
                     "states": " ".join(main_states), "commute_states": " ".join(commute_states),
                     "units": " ".join(pieces), "reason": "metrocast"})
    groups = pd.DataFrame(rows).sort_values("pop", ascending=False, ignore_index=True)
    groups.to_csv(out_dir / "groups.csv", index=False)
    counties.sort_values("fips").to_csv(out_dir / "counties.csv", index=False)
    coupling.to_csv(out_dir / "coupling.csv", index=False)
    sim = counties[counties.piece != ""]
    lt = locs[locs.original_location_code != "All"][["location", "state_abb", "location_name", "location_type"]]
    lt = lt.merge(sim.groupby("piece").agg(state=("state", "first"), pop2019=("pop", "sum"),
                                           n_counties=("fips", "size"), group=("group", "first")),
                  left_on="location", right_index=True)
    lt.to_csv(out_dir / "locations.csv", index=False)
    log.info("%d hub locations in %d groups, %.1f M people, %d counties", len(lt), len(groups),
             sim["pop"].sum() / 1e6, len(sim))
    return groups


# ================================================================== submission
# The model forecasts admissions per hub location (its pieces). The hub scores the percent of ED
# visits due to flu (all locations except California) and, for California's regions and the state,
# flu admissions as CHHS reports them. Both are made from the model's growth, anchored at each
# location's last reported value:
#     value_h = last * W(t_h + LEAD) / W(t_last + LEAD) * exp(sd_h z)
# W the trajectory's weekly admissions (interpolated between weeks; ED leads admissions), z standard
# normal per sample and week. Tuned on main FLUFFHEAD's backtests (13 regions, 2023-24 to 2025-26,
# 484 region x state cases, ED% of the region's counties from NSSP): lead 0.5 week and these sds
# gave WIS 0.74 of a flat forecast with tuned spread (h0 0.79, h3 0.74), 80% coverage 72%.
# The 14 states other than California come from main FLUFFHEAD's state samples for the same week
# (runs/<ref>/state-samples.npz), or else from the sum of this model's pieces in the state.
ED_TARGET = "Flu ED visits pct"
HOSP_TARGET = "Flu hosp admits"
LEAD_WEEKS = 0.5
# 2025-26 hub-location backtest (676 cases, 89 locations, cutoffs 2025-11-08..2026-01-31): with the
# sds below x1.0, WIS 0.75 of epiENGAGE-baseline but 80% coverage 58%; x1.5 (these) same WIS, 74%.
ED_SD = (0.375, 0.525, 0.675, 0.825)        # log scale, hub horizons 0..3
HOSP_SD = (0.15, 0.25, 0.35, 0.45, 0.55)    # horizons -1..3 (CHHS data are a week behind)
N_OUT = 100                                 # samples per location, as the hub requires
HUB_RAW = f"https://raw.githubusercontent.com/{HUB_REPO}/main"


def hub_tasks() -> dict[str, dict]:
    """{target: {"locations": [...], "reference_dates": [...]}} from the hub's current round."""
    path = C.CACHE_DIR / "metrocast" / "tasks.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(f"{HUB_RAW}/hub-config/tasks.json"))
    except requests.RequestException:
        if not path.exists():
            raise
        log.warning("could not download the hub's tasks.json; using the cached copy")
    rnd = json.loads(path.read_text())["rounds"][-1]
    out = {}
    for m in rnd["model_tasks"]:
        t = m["task_ids"]
        vals = lambda k: (t[k].get("required") or []) + (t[k].get("optional") or [])
        for target in vals("target"):
            out[target] = {"locations": vals("location"), "reference_dates": vals("reference_date"),
                           "target_end_dates": vals("target_end_date")}
    return out


def target_data() -> pd.DataFrame:
    """The hub's latest data (target-data/latest-data.csv), downloaded once a day."""
    import datetime as dt
    path = C.CACHE_DIR / "metrocast" / f"latest-data-{dt.date.today()}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(f"{HUB_RAW}/target-data/latest-data.csv"))
    d = pd.read_csv(path)
    d["target_end_date"] = pd.to_datetime(d.target_end_date)
    return d


def _log_at(lw: np.ndarray, i: float) -> np.ndarray:
    """log weekly values [traj, weeks] at fractional week index i; past the end, the last growth."""
    n = lw.shape[1]
    i0 = int(np.floor(i))
    f = i - i0
    if i0 + 1 <= n - 1:
        return lw[:, i0] * (1 - f) + lw[:, i0 + 1] * f
    return lw[:, -1] + (lw[:, -1] - lw[:, -2]) * (i - (n - 1))


def growth(w: np.ndarray, weeks: pd.DatetimeIndex, anchor: pd.Timestamp, targets: list[pd.Timestamp],
           lead: float = LEAD_WEEKS) -> np.ndarray:
    """[traj, targets] W(target + lead) / W(anchor + lead) for weekly trajectories w [traj, weeks]."""
    lw = np.log(np.maximum(w, 0.05))
    a = weeks.get_loc(anchor) + lead
    base = _log_at(lw, a)
    return np.stack([np.exp(_log_at(lw, weeks.get_loc(t) + lead) - base) for t in targets], 1)


def _pick(n_have: int, rng) -> np.ndarray:
    return rng.choice(n_have, N_OUT, replace=n_have < N_OUT)


def piece_samples(region_dir, weeks: pd.DatetimeIndex) -> tuple[dict, dict]:
    """{hub location: [traj, weeks] weekly admissions}, {hub location: group}."""
    from . import submission as S
    out, group = {}, {}
    for f in sorted(region_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        w = S.region_weekly({"daily": z["daily"].astype(float), "dates": pd.DatetimeIndex(z["dates"])}, weeks)
        for k, p in enumerate(z["pieces"]):
            out[str(p)], group[str(p)] = w[:, :, k], f.stem
    return out, group


def _sum_ranked(parts: list[np.ndarray], window: np.ndarray) -> np.ndarray:
    """Sum [traj, weeks] arrays from different groups, pairing trajectories by rank of their total over
    the window (groups are fitted separately; pairing by rank stands in for their shared dynamics)."""
    out = 0
    for p in parts:
        idx = np.linspace(0, p.shape[0] - 1, N_OUT).round().astype(int)
        out = out + p[np.argsort(p[:, window].sum(1), kind="stable")][idx]
    return out


def build_submission(ref_date, region_dir, main_states=None, seed: int = 11) -> tuple[pd.DataFrame, list[str]]:
    """The hub file for `ref_date` from this week's regions; returns (frame, notes)."""
    from . import submission as S
    from .immunity import _state_names
    rng = np.random.default_rng(seed)
    ref = pd.Timestamp(ref_date)
    tasks = hub_tasks()
    obs = target_data()
    weeks = S.season_weeks(pd.Timestamp(ref.year if ref.month >= 8 else ref.year - 1, 8, 1).date(), C.SEASON_END)
    pieces, group = piece_samples(region_dir, weeks)
    locs = pd.read_csv(C.REGIONS_DIR / "locations.csv", dtype=str)
    hub_locs = hub_locations()
    state_fips = {v: k for k, v in _state_names().items()}      # name -> FIPS
    notes, rows = [], []
    if main_states is None:
        notes.append("main FLUFFHEAD's state samples not found: states come from this model's pieces only")
    window = (weeks > ref - pd.Timedelta(weeks=2)) & (weeks <= ref + pd.Timedelta(weeks=3))
    ids = {loc: i for i, loc in enumerate(hub_locs.location)}

    def trajectories(loc: str) -> tuple[np.ndarray | None, str]:
        if loc in pieces:
            return pieces[loc][_pick(pieces[loc].shape[0], rng)], f"piece of {group[loc]}"
        row = hub_locs[hub_locs.location == loc]
        if not len(row) or row.original_location_code.iloc[0] != "All":
            return None, "not simulated"
        st_name = row.state.iloc[0]
        parts = [pieces[p] for p in locs[locs.state_abb == row.state_abb.iloc[0]].location if p in pieces]
        if st_name == "California":            # CHHS's state total is the sum of its regions
            return (_sum_ranked(parts, window), "sum of its regions") if parts else (None, "no regions")
        fips = state_fips.get(st_name)
        if main_states is not None and fips in main_states:
            x = main_states[fips]
            return x[_pick(x.shape[0], rng)], "main FLUFFHEAD state samples"
        if parts:
            return _sum_ranked(parts, window), "sum of this model's pieces (partial state)"
        return None, "no samples"

    for target, sds, first_h in ((ED_TARGET, ED_SD, 0), (HOSP_TARGET, HOSP_SD, -1)):
        if target not in tasks:
            continue
        if str(ref.date()) not in tasks[target]["reference_dates"]:
            notes.append(f"{target}: {ref.date()} is not one of the hub's reference dates")
        o = obs[obs.target == target]
        targets = [ref + pd.Timedelta(weeks=h) for h in range(first_h, 4)]
        for loc in tasks[target]["locations"]:
            x, src = trajectories(loc)
            if x is None:
                notes.append(f"{target} {loc}: left out ({src})")
                continue
            ol = o[(o.location == loc) & (o.target_end_date < targets[0])].dropna(subset=["observation"])
            if ol.empty:
                notes.append(f"{target} {loc}: left out (no reported value)")
                continue
            last = ol.sort_values("target_end_date").iloc[-1]
            if last.target_end_date < targets[0] - pd.Timedelta(weeks=3):
                notes.append(f"{target} {loc}: last report {last.target_end_date.date()} is old")
            g = growth(x, weeks, last.target_end_date, targets)
            v = float(last.observation) * g * np.exp(np.asarray(sds)[: len(targets)] * rng.standard_normal(g.shape))
            if target == ED_TARGET:
                v = np.clip(v, 0, 100)
            for k in range(N_OUT):
                sid = f"L{ids.get(loc, 999):03d}s{k:03d}"
                rows += [(ref.date(), loc, target, t.date(), "sample", sid, round(float(v[k, j]), 4))
                         for j, t in enumerate(targets)]
            log.debug("%s %s: %s, anchor %s = %s", target, loc, src, last.target_end_date.date(), last.observation)
    df = pd.DataFrame(rows, columns=["reference_date", "location", "target", "target_end_date", "output_type",
                                     "output_type_id", "value"])
    return df, notes


def check(df: pd.DataFrame) -> list[str]:
    """The hub's rules this file must meet (its own validation runs on the pull request)."""
    problems = []
    tasks = hub_tasks()
    for (target, loc), g in df.groupby(["target", "location"]):
        if target not in tasks or loc not in tasks[target]["locations"]:
            problems.append(f"{target} {loc}: not a hub task")
            continue
        n = g.groupby("target_end_date").output_type_id.nunique()
        if (n != N_OUT).any():
            problems.append(f"{target} {loc}: {sorted(set(n))} samples per week, need {N_OUT}")
        bad = set(g.target_end_date.astype(str)) - set(tasks[target]["target_end_dates"])
        if bad:
            problems.append(f"{target} {loc}: target_end_date not allowed: {sorted(bad)}")
    if (df.value < 0).any() or df.value.isna().any():
        problems.append("negative or missing values")
    if (df.output_type_id.str.len() > 20).any():
        problems.append("output_type_id longer than 20 characters")
    if df.groupby("output_type_id").location.nunique().max() > 1:
        problems.append("a sample id spans more than one location")
    return problems


def save(df: pd.DataFrame, ref_date) -> "Path":
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = C.FORECAST_DIR / "model-output" / C.model_id()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{ref_date}-{C.model_id()}.parquet"
    schema = pa.schema([("reference_date", pa.date32()), ("location", pa.string()), ("target", pa.string()),
                        ("target_end_date", pa.date32()), ("output_type", pa.string()),
                        ("output_type_id", pa.string()), ("value", pa.float64())])
    pq.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False), path)
    return path


def main_state_samples(ref_date) -> dict | None:
    """Main FLUFFHEAD's state samples for this reference date, if it has run (pipeline.combine)."""
    path = C.REPO / "runs" / str(ref_date) / "state-samples.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return {k: z[k].astype(float) for k in z.files if k != "weeks"}


def write(ref_date, region_dir) -> tuple["Path", list[str]]:
    """Build, check and save the hub file; returns (path, problems). Notes (left-out locations) are logged."""
    df, notes = build_submission(ref_date, region_dir, main_state_samples(ref_date))
    problems = check(df)
    path = save(df, ref_date)
    for n in notes:
        log.warning("%s", n)
    for p in problems:
        log.error("check: %s", p)
    log.info("FLUFFHEADMetro submission: %s (%d locations, %d rows)", path, df.location.nunique(), len(df))
    return path, problems


# ================================================================== backtests
def _wis(samples: np.ndarray, y: float, qs=(0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)) -> float:
    """WIS on the natural scale from samples, with the 23 quantile levels the hub scores."""
    qs = np.asarray(qs)
    lo, hi = np.quantile(samples, qs), np.quantile(samples, 1 - qs[:-1])
    s = 0.5 * abs(y - lo[-1])
    for a, l, u in zip(2 * qs[:-1], lo[:-1], hi):
        s += a / 2 * (u - l) + (l - y) * (y < l) + (y - u) * (y > u)
    return s / (len(qs) - 0.5)


def score_backtest(variant: str, lead: float = LEAD_WEEKS, ed_sd=ED_SD, flat_sd=(0.4, 0.6, 0.8, 1.0),
                   seed: int = 1) -> pd.DataFrame:
    """Backtest cases (runs-local/backtest/<variant>/) scored as hub ED forecasts: anchored at the
    cutoff week's reported ED% for each hub location, horizons 0-3 = 1-4 weeks after the cutoff, vs
    the hub's latest data. Also a flat forecast (the anchor, log-normal spread flat_sd) as a yardstick."""
    from . import backtest as B
    rng = np.random.default_rng(seed)
    obs = target_data()
    ed = obs[obs.target == ED_TARGET].pivot_table(index="target_end_date", columns="location", values="observation")
    rows = []
    for f in sorted((B.OUT / variant).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        season, cutoff, group = f.stem[:4], pd.Timestamp(f.stem[5:15]), f.stem[16:]
        weeks = pd.DatetimeIndex(z["weeks"])
        w = z["w"].astype(float)
        for k, loc in enumerate(z["pieces"]):
            loc = str(loc)
            if loc not in ed.columns or cutoff not in ed.index or np.isnan(ed.at[cutoff, loc]):
                continue
            targets = [cutoff + pd.Timedelta(weeks=h) for h in range(1, 5)]
            g = growth(w[:, :, k], weeks, cutoff, targets, lead)
            e0 = float(ed.at[cutoff, loc])
            for h, t in enumerate(targets):
                if t not in ed.index or np.isnan(ed.at[t, loc]):
                    continue
                y = float(ed.at[t, loc])
                s = np.repeat(e0 * g[:, h], 5) * np.exp(ed_sd[h] * rng.standard_normal(g.shape[0] * 5))
                fl = e0 * np.exp(flat_sd[h] * rng.standard_normal(500))
                q10, q90 = np.quantile(s, [0.1, 0.9])
                rows.append(dict(season=int(season), cutoff=cutoff.date(), group=group, location=loc, h=h,
                                 y=y, median=float(np.median(s)), wis=_wis(s, y), wis_flat=_wis(fl, y),
                                 cover80=bool(q10 <= y <= q90)))
    return pd.DataFrame(rows)
