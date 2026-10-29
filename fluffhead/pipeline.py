"""One week's forecast: data and nowcast, region fits and forecasts, submission.

    data      hub target data through the last reported week, nowcast-revised
              (flusight-fm's revision model, learned from the hub's data versions),
              and NSSP flu ED visits by sub-state region (ed.py)
    regions   each simulation group is fitted to its share of its states' data and
              run to the end of the season, in parallel worker processes
              (runs/<ref_date>/regions/<group>.npz)
    combine   regions -> states -> US, then the submission file and its checks
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from . import submission as S

log = logging.getLogger(__name__)
RUNS_DIR = C.REPO / "runs"


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


def run_region(group: str, wk: WeekData, out_dir: Path, n_draws: int, n_keep: int, n_cont: int,
               n_parallel: int = 1, seed: int | None = None) -> Path:
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s {group} %(message)s")
    seed = seed if seed is not None else zlib.crc32(group.encode()) % (2**31 - 2) + 1   # each region its own stream
    net = N.load(group)
    cutoff = wk.last_obs.date()
    national = (wk.revised[wk.revised.location == "US"].set_index("date").value
                / wk.locations.set_index("location").population["US"])
    region = CB.setup(group, net, cutoff, C.SEASON_END, wk.locations, n_parallel=n_parallel, activity=wk.activity)
    fit = CB.fit(region, wk.revised, national, cutoff, n_draws=n_draws, n_keep=n_keep, seed=seed)
    fc = CB.forecast(region, fit, national, n_cont=n_cont, seed=seed + 1)
    path = out_dir / f"{group}.npz"
    np.savez_compressed(path, daily=fc["daily"].astype(np.float32), dates=fc["dates"].values,
                        pieces=np.array(region.pieces), shares=region.piece_share,
                        activity_shares=region.forecast_share(),
                        weeks=fit.weeks.values, observed=fit.observed)
    fit.theta.to_csv(out_dir / f"{group}-draws.csv", index=False)
    (out_dir / f"{group}.json").write_text(json.dumps({"ess": fit.ess, "phi": fit.phi, "mode": region.pool.mode,
                                                       "n": int(net.n), "pieces": region.pieces}))
    return path


def run_regions(wk: WeekData, groups: list[str], workers: int, n_draws: int, n_keep: int, n_cont: int) -> Path:
    out = RUNS_DIR / str(wk.ref_date) / "regions"
    out.mkdir(parents=True, exist_ok=True)
    todo = [g for g in groups if not (out / f"{g}.npz").exists()]
    log.info("%d regions to run (%d already done) with %d workers", len(todo), len(groups) - len(todo), workers)
    if workers <= 1:
        for g in todo:
            run_region(g, wk, out, n_draws, n_keep, n_cont)
        return out
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(run_region, g, wk, out, n_draws, n_keep, n_cont, workers): g for g in todo}
        for f in as_completed(futs):
            g = futs[f]
            try:
                f.result()
                log.info("finished %s", g)
            except Exception as e:  # one region failing shouldn't stop the others
                log.error("region %s failed: %s", g, e)
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
        n_keep: int = 20, n_cont: int = 5, nowcast: bool = True):
    wk = load_week(ref_date, nowcast=nowcast)
    groups = groups or built_groups()
    region_dir = run_regions(wk, groups, workers, n_draws, n_keep, n_cont)
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
