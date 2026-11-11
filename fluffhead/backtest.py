"""Backtests: forecasts from past cutoffs, scored against what happened.

Each case is (season, cutoff, region, variant). The model sees admissions and weather up to the
cutoff only; after it, weather is climatology with the fading last anomaly (no NWS forecast).
Admissions are final values, so the scores leave out reporting revisions and the nowcast, and
measure the model. Known leaks, small for this purpose: the humidity climatology covers all years,
and the season's own vaccination coverage is used (a real forecast has last season's).

A variant is a dict (see VARIANTS in local/backtest/ or pass one directly):
    profile       prior immunity by age, 5 values (default DiseaseParams.prior_immune)
    profile_by    {subtype: 5 values}: the season's dominant subtype picks the profile
    scale         fix immune_scale at this value
    scale_prior   (lo, hi) for immune_scale (default (0.5, 1.5))
    scale_fn      name of a function in SCALE_FNS: (group, season) -> multiplier of the profile
    r_prior       (lo, hi) for r_now (default Prior's), or {"national": half_width}: centered on the
                  national R at the cutoff (national_r), which small regions can't see for themselves
    set           {calibrate module constant: value}, e.g. FORCING
    calendar      {calendar module constant: value}, e.g. WINTER_BREAK
    structure     {"team": size, "by_home": bool, "class": size}: split workplaces into teams and
                  school grades into classes (apply_structure); mean contacts per person unchanged

Outputs in runs/backtest/<variant>/<season>-<cutoff>-<group>.npz: weekly admissions [traj, weeks,
pieces], truth, posterior means. score() reads them.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import calendar as K
from . import calibrate as CB
from . import config as C
from . import ed as ED
from . import hub_data as D
from . import network as N
from . import strains as ST

log = logging.getLogger(__name__)

OUT = C.REPO / "runs" / "backtest"
WEATHER_TODAY = dt.date(2026, 11, 4)      # the cached weather files
PANEL = {"csa438": "Portland ME", "csa162": "Burlington VT", "csa106": "Albuquerque NM", "csa332": "Las Vegas NV",
         "csa416": "Oklahoma City OK", "csa142": "Birmingham AL", "csa340": "Little Rock AR", "csa218": "Des Moines IA",
         "csa278": "Hartford CT", "csa376": "Milwaukee WI", "csa482": "Salt Lake City UT", "csa400": "Nashville TN"}
# dominant influenza A subtype of the season (WHO FluNet, US; 2024-25 was co-dominant)
SEASON_SUBTYPE = {2022: "H3N2", 2023: "H1N1", 2024: "H1N1", 2025: "H3N2"}
CUTOFFS = {2022: ("2022-10-15", "2022-11-05"),            # 2022-23 peaked in the week to Dec 3
           2023: ("2023-10-28", "2023-11-18", "2023-12-09"),
           2024: ("2024-11-09", "2024-11-30", "2024-12-14"),   # NHSN reporting was voluntary to Nov 1, 2024
           2025: ("2025-10-25", "2025-11-15", "2025-12-06")}
import pandas as _pd
# weekly cutoffs around each season's takeoff, peak and decline (national peaks Dec 30, 2023; Feb 8, 2025; Jan 3, 2026)
WEEKLY = {2023: tuple(str(d.date()) for d in _pd.date_range("2023-11-11", "2024-01-27", freq="7D")),
          2024: tuple(str(d.date()) for d in _pd.date_range("2024-11-16", "2025-02-22", freq="7D")),
          2025: tuple(str(d.date()) for d in _pd.date_range("2025-11-08", "2026-01-31", freq="7D"))}
HORIZON_WEEKS = 4                         # the hub's horizons 0-3
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
WEEKLY_GROWTH_PER_LOG_R = 1.94            # model generation time ~3.6 days
SCALE_FNS: dict = {}
R_MAP = CB.r_map                          # run_case replaces CB.r_map for structured networks


def truth() -> pd.DataFrame:
    return D._read_target_csv(C.CACHE_DIR / "current" / "hosp.csv")


def _activity(season: int, cutoff: dt.date) -> pd.DataFrame:
    p = C.CACHE_DIR / "immunity" / f"activity-{season}-{cutoff}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    a = ED.activity(CB.season_start(cutoff), cutoff)
    p.parent.mkdir(parents=True, exist_ok=True)
    a.to_pickle(p)
    return a


def national_r(us: pd.Series, cut: dt.date) -> float:
    """R implied by the last four weeks of national growth above the summer background, known at `cut`."""
    bg = us[f"{cut.year}-07-01":f"{cut.year}-08-31"].median()
    v = np.log((us[:pd.Timestamp(cut)].iloc[-4:] - bg).clip(lower=1).to_numpy())
    g = np.polyfit(np.arange(4), v, 1)[0]
    return float(np.clip(np.exp(g / WEEKLY_GROWTH_PER_LOG_R), 0.95, 1.4))


def apply_structure(net: N.Network, st: dict) -> None:
    """Relabel blocks so that most of a group's contacts fall inside a team or class.

    A group's edges are drawn block by block with the layer's associativity, so a team as a block
    gets associativity x K contacts inside it and the rest across the group: same mean degree,
    more clustering. Teams nest within income category and classes within grade. With `by_home`,
    workers are ordered by home tract before the split (dummies take a random coworker's tract)."""
    for L, size in ((N.WP, st.get("team")), (N.SCH, st.get("class"))):
        if not size:
            continue
        rng = np.random.default_rng(1000 + L)
        K = N.layer_rule(net, L)[1]
        ptr, mem, blk = net.grp_ptr[L], net.grp_mem[L], net.grp_blk[L].copy()
        for g in np.flatnonzero(np.diff(ptr) >= max(K + 2, 2 * size)):
            a, b = ptr[g], ptr[g + 1]
            n = b - a
            home = rng.random(n)
            if L == N.WP and st.get("by_home"):
                h = net.home_tract[mem[a:b]].astype(float)
                known = h[h >= 0]
                if len(known):
                    h[h < 0] = rng.choice(known, int((h < 0).sum()))
                    home = h
            order = np.lexsort((rng.random(n), home, blk[a:b]))
            n_teams = max(round(n / size), 1)
            lab = np.empty(n, np.int16)
            lab[order] = np.arange(n) * n_teams // n
            blk[a:b] = lab
        net.grp_blk[L] = blk


def case_path(variant: str, season: int, cutoff: str, group: str) -> Path:
    return OUT / variant / f"{season}-{cutoff}-{group}.npz"


def run_case(variant: str, spec: dict, season: int, cutoff: str, group: str, n_draws: int, n_cont: int,
             draw_workers: int) -> Path:
    logging.basicConfig(level=logging.WARNING)
    for k, v in spec.get("set", {}).items():
        setattr(CB, k, v)
    for k, v in spec.get("calendar", {}).items():
        setattr(K, k, v)
    t0 = time.time()
    cut = dt.date.fromisoformat(cutoff)
    end = cut + dt.timedelta(weeks=HORIZON_WEEKS)
    tr = truth()
    locs = D.load_locations()
    seen = tr[tr.date <= pd.Timestamp(cut)]
    seen = seen[seen.location.isin(locs.location)].drop_duplicates(["date", "location"], keep="last").copy()
    seen["value"] = seen["value"].astype(float)
    national = (seen[seen.location == "US"].set_index("date").value / locs.set_index("location").population["US"])
    scenario = ST.Scenario(subtype=SEASON_SUBTYPE[season])
    net = N.load(group)
    region = CB.setup(group, net, cut, end, locs, n_parallel=draw_workers, activity=_activity(season, cut),
                      scenario=scenario, weather_as_of=cut, weather_today=WEATHER_TODAY)
    region.draw_workers = draw_workers
    for k, v in spec.get("disease", {}).items():     # DiseaseParams overrides, e.g. asym_fraction
        setattr(region.base, k, np.array(v, float) if isinstance(v, (list, tuple)) else v)
    prof = spec.get("profile_by", {}).get(SEASON_SUBTYPE[season], spec.get("profile"))
    if prof is not None:
        region.base.prior_immune = np.array(prof, float)
    if spec.get("scale_fn"):
        region.base.prior_immune = np.clip(region.base.prior_immune * SCALE_FNS[spec["scale_fn"]](group, season), 0, 0.9)
    pr = spec.get("scale_prior", (0.5, 1.5))
    if spec.get("scale") is not None:
        s = float(spec["scale"])
        pr = (s * 0.995, s * 1.005)
    seed = zlib.crc32(f"{group}-{cutoff}".encode()) % (2**31 - 2000) + 1 + spec.get("seed_add", 0)     # the same across variants unless asked
    prior = CB.Prior(immune_scale=pr)
    if spec.get("travel_prior"):
        prior.travel_share = tuple(spec["travel_prior"])
    rp = spec.get("r_prior")
    if isinstance(rp, dict) and "ed" in rp:     # R from the last three weeks' national ED growth, known at the cutoff
        ed = pd.read_csv(spec["ed_file"], parse_dates=["date"]).set_index("date").ed
        v = np.log(ed[:pd.Timestamp(cut)].iloc[-3:].to_numpy() + 0.05)
        c = float(np.clip(np.exp(np.polyfit(np.arange(3), v, 1)[0] / WEEKLY_GROWTH_PER_LOG_R), 0.95, 1.5))
        prior.r_now = (c - rp["ed"], c + rp["ed"])
    elif isinstance(rp, dict):
        c = national_r(seen[seen.location == "US"].set_index("date").value.sort_index(), cut)
        prior.r_now = (c - rp["national"], c + rp["national"])
    elif rp is not None:
        prior.r_now = tuple(rp)
    if spec.get("structure"):
        # r_now keeps its as-built meaning: the R map's second-generation R drops with clustering though
        # growth barely does, so a map of the structured network would ask for too high a beta
        rm = R_MAP(region, seed)
        apply_structure(net, spec["structure"])
        region.pool = N.EdgePool.create(net, region.pool.mode, 1)
        CB.r_map = lambda *_: rm              # a pool per variant, so this stays within it
    if spec.get("national_oracle"):             # imports after the cutoff follow the real national curve
        us = tr[tr.location == "US"].drop_duplicates("date", keep="last").set_index("date").value.astype(float).sort_index()
        days = pd.date_range(region.start, periods=region.days)
        daily = us.reindex(us.index.union(days)).interpolate().reindex(days)
        ref = daily[(days > pd.Timestamp(cut) - pd.Timedelta(days=7)) & (days <= pd.Timestamp(cut))].mean()
        region.national_path = (daily / ref).to_numpy()
    if spec.get("mult_all") is not None:
        CB.MULT_CENTER = np.full(len(region.pieces), float(spec["mult_all"]))
    if spec.get("mult_center"):
        ct = pd.read_csv(spec["mult_center"], dtype={"location": str})
        ct = ct[ct.season == season].set_index("location").center
        CB.MULT_CENTER = np.array([float(ct.get(p, 0.0)) for p in region.pieces])
    fit = CB.fit(region, seen, national, cut, n_draws=n_draws, n_keep=spec.get("n_keep", 20), seed=seed, prior=prior)
    fc = CB.forecast(region, fit, national, n_cont=n_cont, seed=seed + 1)
    probe = attack_probe(region, fit, national, cut)
    dates = pd.DatetimeIndex(fc["dates"])
    ends = pd.date_range(pd.Timestamp(cut) + pd.Timedelta(days=7), periods=HORIZON_WEEKS, freq="7D")
    hist = pd.date_range(pd.Timestamp(cut) - pd.Timedelta(weeks=7), periods=8, freq="7D")
    allw = hist.append(ends)
    daily = fc["daily"]
    w = np.stack([daily[:, (dates > e - pd.Timedelta(days=7)) & (dates <= e)].sum(1) for e in allw], 1)   # [traj, weeks, pieces]
    share = region.forecast_share()
    tv = tr.drop_duplicates(["date", "location"], keep="last").pivot(index="date", columns="location", values="value").astype(float)
    y = np.column_stack([tv[p].reindex(allw).to_numpy() * s for p, s in zip(region.pieces, share)])
    th = fit.kept
    p = case_path(variant, season, cutoff, group)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, w=w.astype(np.float32), y=y, weeks=allw.values, n_hist=len(hist), pieces=np.array(region.pieces), share=share,
                        post=json.dumps({k: float(th[k].mean()) for k in ("r_now", "immune_scale", "travel_share", "humidity_floor", "ve_infection")}),
                        probe=json.dumps(probe),
                        mult=np.array([np.exp(np.log(th[f"mult_{q}"]).mean()) for q in region.pieces]),
                        ess=fit.ess, phi=fit.phi, secs=time.time() - t0)
    return p


def attack_probe(region, fit, national: pd.Series, cut: dt.date, n: int = 5) -> dict:
    """Over some kept draws, to the cutoff: resident infection attack rate and expected admissions per infection."""
    pressure = CB.import_pressure(region, national, cut)
    day_cut = (cut - region.start).days + 1
    res, a0 = [], []
    for t in fit.kept.drop_duplicates("seed").head(n).itertuples():
        sim = CB.make_sim(region, t, pressure, int(t.seed))
        CB.advance(region, sim, t, pressure, day_cut, day_cut)
        j = sim.rec[0][:sim.rn]
        m = region.net.resident[j]
        k = int(m.sum())
        adm = float(CB.expected_admissions(region, sim)[:day_cut].sum()) / CB._people_per_agent(region)
        res.append((k / max(int(region.net.resident.sum()), 1), adm / max(k, 1)))
        a0.append(np.bincount(np.where(region.net.age_group[j[m]] < 0, 2, region.net.age_group[j[m]]), minlength=5) / max(k, 1))
    r = np.array(res)
    return {"attack_cut": float(r[:, 0].mean()), "adm_per_inf": float(r[:, 1].mean()), "age_mix": np.mean(a0, 0).round(4).tolist()}


def run(variants: dict, seasons=None, groups=None, workers: int = 3, draw_workers: int = 3, n_draws: int = 300,
        n_cont: int = 3, cutoffs: dict | None = None) -> None:
    groups = groups or list(PANEL)
    CUTS = cutoffs or CUTOFFS
    seasons = seasons or list(CUTS)
    jobs = [(v, s, c, g) for v in variants for s in seasons for c in CUTS[s] for g in groups
            if not case_path(v, s, c, g).exists()]
    for s in seasons:                     # the ED data once, before workers share the cache
        for c in CUTS[s]:
            _activity(s, dt.date.fromisoformat(c))
    log.info("%d cases to run", len(jobs))
    t0, done = time.time(), 0
    for name in variants:                 # a pool per variant: `set` and `calendar` change module constants
        todo = [j for j in jobs if j[0] == name]
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(run_case, v, variants[v], s, c, g, n_draws, n_cont, draw_workers): (v, s, c, g) for v, s, c, g in todo}
            for f in as_completed(futs):
                done += 1
                try:
                    f.result()
                    if done % 10 == 0 or done == len(jobs):
                        log.info("%d/%d cases, %.0f min", done, len(jobs), (time.time() - t0) / 60)
                except Exception as e:     # one failure shouldn't stop the rest
                    log.error("case %s failed: %s", futs[f], e)


def wis(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weighted interval score of log1p values; q [5 quantiles, ...] from QUANTILES."""
    m = np.abs(y - q[2]) / 2
    tot = m.copy()
    for lo, hi, a in ((q[0], q[4], 0.1), (q[1], q[3], 0.5)):
        tot = tot + a / 2 * ((hi - lo) + 2 / a * np.maximum(lo - y, 0) + 2 / a * np.maximum(y - hi, 0))
    return tot / 2.5


def score(variant: str) -> pd.DataFrame:
    rows = []
    for f in sorted((OUT / variant).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        season, cutoff, group = f.stem[:4], f.stem[5:15], f.stem[16:]
        nh = int(z["n_hist"])
        w, y = z["w"].astype(float), z["y"]
        post = json.loads(str(z["post"]))
        for k, p in enumerate(z["pieces"]):
            if z["share"][k] < 0.1:           # a sliver of a state
                continue
            traj = np.log1p(w[:, :, k])
            q = np.quantile(traj, QUANTILES, axis=0)
            ly = np.log1p(y[:, k])
            for i in range(nh, w.shape[1]):
                if np.isnan(ly[i]):
                    continue
                rows.append(dict(variant=variant, season=int(season), cutoff=cutoff, group=group, piece=str(p), h=i - nh + 1,
                                 week=str(z["weeks"][i])[:10], wis=wis(q[:, i], ly[i]), err=q[2, i] - ly[i],
                                 cover90=q[0, i] <= ly[i] <= q[4, i], cover50=q[1, i] <= ly[i] <= q[3, i],
                                 ess=float(z["ess"]), phi=float(z["phi"]), secs=float(z["secs"]), **post))
    return pd.DataFrame(rows)
