"""One week's forecast: data and nowcast, region fits and forecasts, submission.

    data      hub target data through the last reported week, nowcast-revised
              (flusight-fm's revision model, learned from the hub's data versions),
              and NSSP flu ED visits by sub-state region (ed.py)
    regions   each simulation group is fitted to its share of its states' data and
              run to the end of the season, in parallel worker processes
              (runs/<ref_date>/regions/<group>.npz). With a variant introduced in one
              region (strains.py), that region runs first and the others follow its
              trajectories.
    pass 2    in pass 1 a region's imports after the cutoff follow its own incidence (the
              region standing in for the nation), which feeds back: a region with a large
              import share and local R just under 1 burns slowly for months (Seattle, Nov
              2026: peak in March). Pass 2 sums the pass-1 forecasts into a national
              incidence curve (national_path) and re-runs every region's forecast with
              its imports following it. The fits are kept (imports before the cutoff come
              from data); pass-1 outputs move to regions/pass1/.
    combine   regions -> states -> US, then the submission file and its checks
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
import zlib
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibrate as CB
from . import config as C
from . import ed as ED
from . import hub_data as D
from . import network as N
from . import nowcast as nc
from . import strains as ST
from . import submission as S
from . import weather as W

log = logging.getLogger(__name__)
RUNS_DIR = C.REPO / "runs"
INFECTION_LAG_DAYS = 7          # national_path: admissions lag infections about this long


@dataclass
class WeekData:
    ref_date: dt.date
    last_obs: pd.Timestamp
    locations: pd.DataFrame
    revised: pd.DataFrame        # long: date, location, value (nowcast medians in the last weeks)
    nowcasts: pd.DataFrame       # flusight-fm nowcast table (mu, sd on log(x + 1))
    warnings: list
    activity: pd.DataFrame       # [weeks, county FIPS] NSSP flu ED activity (empty: population shares)


def reference_date_for(day: dt.date) -> dt.date:
    return D.reference_date_for(day)


def _github_token() -> None:
    """Use the gh CLI's token for the hub's commit history (higher API rate limit)."""
    if os.environ.get("GITHUB_TOKEN"):
        return
    try:
        tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10).stdout.strip()
        if tok:
            os.environ["GITHUB_TOKEN"] = tok
    except (OSError, subprocess.SubprocessError):
        pass


def load_week(ref_date: dt.date, nowcast: bool = True) -> WeekData:
    locations = D.load_locations()
    cur = D.load_current("hosp").dropna(subset=["value"])
    cutoff = pd.Timestamp(ref_date) - pd.Timedelta(weeks=1)
    cur = cur[(cur.date <= cutoff) & cur.location.isin(locations.location)]
    last_obs = cur.date.max()
    warnings = []
    if last_obs < cutoff:
        warnings.append(f"data ends {last_obs.date()}; this week's update may not be posted yet")
    series = {loc: g.drop_duplicates("date", keep="last").set_index("date").value.astype(float).sort_index()
              for loc, g in cur.groupby("location")}
    models, prior = {}, {}
    if nowcast:
        _github_token()
        vint = D.load_vintages("hosp", C.VERSIONS_START)
        models = nc.fit_revisions(vint, ref_date, C.LOG_OFFSET, C.NOWCAST_MODEL)
        dates = [last_obs - pd.Timedelta(weeks=k) for k in range(C.NOWCAST_WEEKS)]
        prior = nc.prior_forecasts(C.FORECAST_DIR, C.HOSP_TARGET, dates, C.LOG_OFFSET)
    revised, table = nc.nowcast(series, models, prior, C.LOG_OFFSET, adjust=nowcast)
    rev = pd.concat([s.rename("value").rename_axis("date").reset_index().assign(location=loc)
                     for loc, s in revised.items()], ignore_index=True)
    from .calibrate import season_start
    activity = ED.activity(season_start(last_obs.date()), last_obs.date())
    if activity.empty:
        warnings.append("NSSP ED data unavailable: regions get population shares of their states")
    elif activity.index.max() < last_obs:
        warnings.append(f"NSSP ED data end {activity.index.max().date()}; later weeks reuse the last shares")
    return WeekData(ref_date, last_obs, locations, rev, table, warnings, activity)


# ------------------------------------------------------------------ regions
def built_groups() -> list[str]:
    g = pd.read_csv(C.REGIONS_DIR / "groups.csv")
    return [x for x in g.sort_values("pop", ascending=False).group
            if (C.REGIONS_DIR / x / "network" / "network.npz").exists()]


def _by_size(groups: list[str]) -> list[str]:
    """Largest first."""
    rank = {g: i for i, g in enumerate(built_groups())}
    return sorted(groups, key=lambda g: rank.get(g, len(rank)))


def _in_slots(ex, submit, todo: list[str], slots: int, order: str, label: str = "") -> None:
    """Run `todo` (largest first) in `slots` parallel slots, a slot taking its next region when its
    last ends. order "largest": every slot takes the largest left. "barbell": the first slot takes the
    largest left, the others the smallest, so one large region runs at a time beside small ones
    (large regions together fill a small machine's memory). `submit(ex, group)` returns a future."""
    queue = deque(todo)
    running = {}                 # future -> (group, slot)
    def fill(slot):
        g = queue.popleft() if slot == 0 or order == "largest" else queue.pop()
        log.info("%sstarting %s", label, g)
        running[submit(ex, g)] = (g, slot)
    for slot in range(slots):
        if queue:
            fill(slot)
    while running:
        done, _ = wait(running, return_when=FIRST_COMPLETED)
        for f in done:
            g, slot = running.pop(f)
            try:
                f.result()
                log.info("%sfinished %s", label, g)
            except Exception as e:  # one region failing shouldn't stop the others
                log.error("%sregion %s failed: %s", label, g, e)
            if queue:
                fill(slot)


def origin_track(path: Path) -> ST.OriginTrack:
    z = np.load(path)
    return ST.OriginTrack(z["variant_share"], z["variant_escape"], z["variant_advantage"], z["variant_ve_loss"],
                          z["variant_start"], z["variant_intro"],
                          z["variant_escape_young"] if "variant_escape_young" in z.files else None)


def _seed(group: str) -> int:
    return zlib.crc32(group.encode()) % (2**31 - 2) + 1      # each region its own stream


def _region(group: str, wk: WeekData, out_dir: Path, n_parallel: int, scenario: ST.Scenario,
            draw_workers: int = 1):
    """The region set up for a fit or forecast (with its variant's origin and links), and national pressure."""
    lg = logging.getLogger(__name__)
    net = N.load(group)
    cutoff = wk.last_obs.date()
    national = (wk.revised[wk.revised.location == "US"].set_index("date").value
                / wk.locations.set_index("location").population["US"])
    region = CB.setup(group, net, cutoff, C.SEASON_END, wk.locations, n_parallel=n_parallel * draw_workers,
                      activity=wk.activity, scenario=scenario)
    region.draw_workers = draw_workers
    lg.info("%s: scenario: %s", group, scenario.label())
    if scenario.follows_origin(group):
        from . import airlinks as A
        o = out_dir / "pass1" / f"{scenario.origin}.npz"      # pass 2: the trajectories the fit followed
        region.origin_track = origin_track(o if o.exists() else out_dir / f"{scenario.origin}.npz")
        region.link = A.link(group, scenario.origin)
        lg.info("%s: linkage to the variant's origin %s: %.3f", group, scenario.origin, region.link)
    if scenario.strains == 2:
        from . import airlinks as A
        region.asia = A.asia_arrivals(group)
        if group == scenario.origin and scenario.share_groups:
            region.near = A.air_share(group, scenario.share_groups)
        lg.info("%s: arrivals from Japan / East Asia %.0f per day (weighted); origin's air share to the "
                "variant zone %.3f", group, region.asia, region.near)
    return region, national


def run_region(group: str, wk: WeekData, out_dir: Path, n_draws: int, n_keep: int, n_cont: int,
               n_parallel: int = 1, seed: int | None = None, scenario: ST.Scenario | None = None,
               draw_workers: int = 1) -> Path:
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s {group} %(message)s")
    lg = logging.getLogger(__name__)
    scenario = scenario or ST.Scenario()
    seed = seed if seed is not None else _seed(group)
    region, national = _region(group, wk, out_dir, n_parallel, scenario, draw_workers)
    cutoff = region.cutoff
    from . import hindcast as H
    imm = H.prior_for(group)
    lg.info("%s: immune_scale prior %.2f-%.2f", group, *imm)
    prior = CB.region_prior(region, imm)
    rounds = CB.N_ROUNDS
    if prior.extra:              # the variant's parameters too: a larger search
        n_draws, rounds = n_draws * CB.VARIANT_DRAWS_FACTOR, CB.VARIANT_ROUNDS
    fit = CB.fit(region, wk.revised, national, cutoff, n_draws=n_draws, n_keep=n_keep, seed=seed,
                 prior=prior, n_rounds=rounds)
    fc = CB.forecast(region, fit, national, n_cont=n_cont, seed=seed + 1)
    fit.theta.to_csv(out_dir / f"{group}-draws.csv", index=False)
    fit.kept.to_csv(out_dir / f"{group}-kept.csv", index=False)
    return _save(region, fit, fc, out_dir, scenario, {"pass": 1})


def _save(region, fit, fc, out_dir: Path, scenario: ST.Scenario, info: dict) -> Path:
    path = out_dir / f"{region.group}.npz"
    extra = {k: fc[k].astype(np.float32) for k in ("variant_share", "variant_escape", "variant_advantage", "variant_ve_loss",
                                                      "variant_start", "variant_intro", "variant_escape_young")
             if k in fc}
    np.savez_compressed(path, daily=fc["daily"].astype(np.float32), dates=fc["dates"].values,
                        pieces=np.array(region.pieces), shares=region.piece_share,
                        activity_shares=region.forecast_share(),
                        weeks=fit.weeks.values, observed=fit.observed, **extra)
    (out_dir / f"{region.group}.json").write_text(json.dumps({
        "ess": fit.ess, "phi": fit.phi, "mode": region.pool.mode, "n": int(region.net.n), "pieces": region.pieces,
        "scenario": scenario.to_json(), "link": region.link, **info}))
    return path


def national_path(region_dir: Path, cutoff: dt.date, replace: dict[str, Path] | None = None) -> np.ndarray:
    """[days] national incidence after the cutoff relative to the week to the cutoff: the sum over
    regions of the trajectories' mean daily admissions, moved INFECTION_LAG_DAYS earlier. `replace`
    swaps in other files for some regions (tests)."""
    files = {f.stem: f for f in sorted(region_dir.glob("*.npz"))}
    files.update(replace or {})
    tot, dates = None, None
    for f in files.values():
        z = np.load(f)
        m = z["daily"].astype(float).sum(2).mean(0)
        tot = m if tot is None else tot + m
        dates = pd.DatetimeIndex(z["dates"])
    s = pd.Series(tot, index=dates).shift(-INFECTION_LAG_DAYS).ffill()
    c = pd.Timestamp(cutoff)
    ref = s[(s.index > c - pd.Timedelta(days=7)) & (s.index <= c)].mean()
    return (s / ref).to_numpy()


def reforecast_region(group: str, wk: WeekData, out_dir: Path, path: np.ndarray, n_keep: int, n_cont: int,
                      n_parallel: int = 1, scenario: ST.Scenario | None = None, draw_workers: int = 1) -> Path:
    """Pass 2: the region's forecast again, from its pass-1 fit, with imports after the cutoff following
    the national curve `path`. Pass-1 outputs move to out_dir/pass1/."""
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s {group} %(message)s")
    lg = logging.getLogger(__name__)
    scenario = scenario or ST.Scenario()
    seed = _seed(group)
    p1 = out_dir / "pass1"
    p1.mkdir(exist_ok=True)
    for suffix in (".npz", ".json"):
        if (out_dir / f"{group}{suffix}").exists() and not (p1 / f"{group}{suffix}").exists():
            (out_dir / f"{group}{suffix}").rename(p1 / f"{group}{suffix}")
    region, national = _region(group, wk, out_dir, n_parallel, scenario, draw_workers)
    region.r_map = CB.r_map(region, seed)            # as in the fit (same seed)
    region.national_path = path
    theta = pd.read_csv(out_dir / f"{group}-draws.csv")
    kf = out_dir / f"{group}-kept.csv"
    if kf.exists():
        kept = pd.read_csv(kf)
    else:                                            # runs from before kept draws were saved: resample them
        idx = np.random.default_rng(seed).choice(len(theta), n_keep, replace=True, p=theta.weight / theta.weight.sum())
        kept = theta.iloc[idx].reset_index(drop=True)
        lg.warning("%s: no kept draws saved; resampled %d from the fit's weights", group, n_keep)
    tgt = CB.targets(region, wk.revised, region.cutoff)
    y = tgt.to_numpy()
    j1 = json.loads((p1 / f"{group}.json").read_text()) if (p1 / f"{group}.json").exists() else {}
    fit = CB.Fit(group, region.cutoff, theta, kept, tgt.index, y, j1.get("ess", float("nan")),
                 j1.get("phi", float("nan")), CB.baseline(y))
    lg.info("%s: pass 2: imports after the cutoff follow the national curve (peak x%.2f of the cutoff week on %s)",
            group, np.nanmax(path), (region.start + dt.timedelta(days=int(np.nanargmax(path)))))
    fc = CB.forecast(region, fit, national, n_cont=n_cont, seed=seed + 1)
    return _save(region, fit, fc, out_dir, scenario, {"pass": 2})


def second_pass(wk: WeekData, out_dir: Path, groups: list[str], workers: int, n_keep: int, n_cont: int,
                scenario: ST.Scenario | None = None, draw_workers: int = 1, order: str = "largest") -> None:
    done = lambda g: (out_dir / f"{g}.json").exists() and json.loads((out_dir / f"{g}.json").read_text()).get("pass") == 2
    saved = out_dir / "national_path.npy"
    if saved.exists():           # resuming: the curve pass 2 started with
        path = np.load(saved)
    else:
        path = national_path(out_dir, wk.last_obs.date(),
                             replace={f.stem: f for f in (out_dir / "pass1").glob("*.npz")})   # pass-1 files
        np.save(saved, path)
    todo = [g for g in groups if (out_dir / f"{g}-draws.csv").exists() and not done(g)]
    log.info("pass 2: %d regions to re-forecast with %d workers", len(todo), workers)
    if workers <= 1:
        for g in todo:
            reforecast_region(g, wk, out_dir, path, n_keep, n_cont, scenario=scenario, draw_workers=draw_workers)
        return
    todo = _by_size(todo)
    if order == "barbell" and todo and todo[0] == _by_size(groups)[0]:      # the largest alone, with every process
        try:
            reforecast_region(todo[0], wk, out_dir, path, n_keep, n_cont, 1, scenario, workers * draw_workers)
            log.info("pass 2 finished %s", todo[0])
        except Exception as e:
            log.error("pass 2 region %s failed: %s", todo[0], e)
        todo = todo[1:]
    with ProcessPoolExecutor(workers) as ex:
        _in_slots(ex, lambda e, g: e.submit(reforecast_region, g, wk, out_dir, path, n_keep, n_cont, workers, scenario,
                                            draw_workers), todo, workers, order, "pass 2 ")


def run_regions(wk: WeekData, groups: list[str], workers: int, n_draws: int, n_keep: int, n_cont: int,
                scenario: ST.Scenario | None = None, draw_workers: int = 1, order: str = "largest") -> Path:
    """Regions in `workers` processes, each running its draws in `draw_workers` processes; the
    variant's origin first, alone, with all of them (workers x draw_workers). With order "barbell"
    the largest region runs alone the same way, then _in_slots."""
    scenario = scenario or ST.Scenario()
    out = RUNS_DIR / str(wk.ref_date) / "regions"
    out.mkdir(parents=True, exist_ok=True)
    todo = [g for g in groups if not (out / f"{g}.npz").exists() and not (out / "pass1" / f"{g}.npz").exists()]
    log.info("%d regions to run (%d already done) with %d workers; %s", len(todo), len(groups) - len(todo),
             workers, scenario.label())
    if scenario.strains == 2 and scenario.origin != ST.EVERYWHERE:
        if scenario.origin not in built_groups():
            raise ValueError(f"variant origin {scenario.origin} is not a built group")
        if not (out / f"{scenario.origin}.npz").exists():     # the origin first: the others follow it
            log.info("running the variant's origin %s first", scenario.origin)
            run_region(scenario.origin, wk, out, n_draws, n_keep, n_cont, 1, scenario=scenario,
                       draw_workers=max(workers, 1) * draw_workers)
        todo = [g for g in todo if g != scenario.origin]
    if workers <= 1:
        for g in todo:
            run_region(g, wk, out, n_draws, n_keep, n_cont, scenario=scenario, draw_workers=draw_workers)
        return out
    todo = _by_size(todo)
    if order == "barbell" and todo and todo[0] == _by_size(groups)[0]:
        log.info("running the largest region %s alone", todo[0])
        try:
            run_region(todo[0], wk, out, n_draws, n_keep, n_cont, 1, scenario=scenario, draw_workers=workers * draw_workers)
            log.info("finished %s", todo[0])
        except Exception as e:
            log.error("region %s failed: %s", todo[0], e)
        todo = todo[1:]
    with ProcessPoolExecutor(workers) as ex:
        _in_slots(ex, lambda e, g: e.submit(run_region, g, wk, out, n_draws, n_keep, n_cont, workers, None, scenario,
                                            draw_workers), todo, workers, order)
    return out


# ------------------------------------------------------------------ combine
def combine(wk: WeekData, region_dir: Path, seed: int = 7):
    rng = np.random.default_rng(seed)
    weeks = S.season_weeks(dt.date(wk.ref_date.year if wk.ref_date.month >= 8 else wk.ref_date.year - 1, 8, 1),
                           C.SEASON_END)
    weekly, pieces, shares, act = {}, {}, {}, {}
    for f in sorted(region_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        res = {"daily": z["daily"].astype(float), "dates": pd.DatetimeIndex(z["dates"])}
        weekly[f.stem] = S.region_weekly(res, weeks)
        pieces[f.stem] = list(z["pieces"])
        shares[f.stem] = z["shares"]
        act[f.stem] = z["activity_shares"] if "activity_shares" in z else z["shares"]
    window = (weeks > wk.last_obs) & (weeks <= pd.Timestamp(C.SEASON_END))
    picks = S.joint_samples(weekly, window, rng)
    states = S.state_samples(weekly, pieces, shares, act, picks, wk.revised, weeks, wk.locations,
                             wk.last_obs.date(), rng)
    locs = wk.locations[wk.locations.location.isin(list(states) + ["US"])]
    juris = [l for l in wk.locations.location if l != "US"]
    df, noisy = S.build(states, weeks, wk.revised, wk.nowcasts, locs, wk.ref_date, rng, juris)
    problems = S.validate(df, locs[locs.location.isin(noisy)])
    absent = sorted(set(wk.locations.location) - set(locs.location))
    return df, noisy, weeks, problems, absent


def run(ref_date: dt.date, groups: list[str] | None = None, workers: int = 4, n_draws: int = 300,
        n_keep: int = 20, n_cont: int = 5, nowcast: bool = True, scenario: ST.Scenario | None = None,
        two_pass: bool = True, draw_workers: int = 1, order: str = "largest"):
    wk = load_week(ref_date, nowcast=nowcast)
    groups = groups or built_groups()
    W.prefetch(groups)     # station downloads once, before workers share the cache
    region_dir = run_regions(wk, groups, workers, n_draws, n_keep, n_cont, scenario, draw_workers, order)
    if two_pass:
        second_pass(wk, region_dir, groups, workers, n_keep, n_cont, scenario, draw_workers, order)
    df, noisy, weeks, problems, absent = combine(wk, region_dir)
    path = S.save(df, ref_date)
    from . import report
    report_path = report.write(wk, df, noisy, weeks, problems, region_dir)
    log.info("report: %s", report_path)
    for p in problems:
        log.error("validation: %s", p)
    if absent:
        log.warning("no forecast for %s (no simulated groups)", ", ".join(absent))
    return wk, df, noisy, weeks, problems, path
