"""Hindcasts: fit whole past seasons to learn each region's prior immunity.

A forecast from late October can't tell how much of the population is immune: the
current R is fitted directly, and almost no one has been infected yet, so immunity
changes neither the growth nor the shape of the curve up to the cutoff. It shows
only after the cutoff, as how soon growth stops. A past season's turnover does show
it, so each region is fitted to each past season in HINDCAST_CUTOFFS: data from the
season start to a few weeks after the national peak, all weeks weighted equally (no
recency half-life), with wider priors on immune_scale, r_now and the multiplier.
No forecast is made.

prior_for(group) then turns the region's posteriors into the current season's
immune_scale prior, using only informative seasons (posterior sd below
INFORMATIVE_SD and phi at least MIN_PHI; otherwise the default prior): centred on
the mean of their posterior means, half-width max(MIN_HALF_WIDTH, the spread
between seasons + the mean posterior sd).

Caveat (first runs, Las Vegas 2025-26): the model has one transmissibility level per
season (plus humidity and calendars), so it can't reproduce a strain that spreads
late and then explodes; fits prefer small epidemics with large multipliers, and
immune_scale stays close to its prior. Hence the informativeness rule.
Outputs: runs/hindcast-<season>/<group>-draws.csv and <group>.json.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibrate as CB
from . import config as C
from . import network as N

log = logging.getLogger(__name__)

# season start year -> cutoff: about three weeks after the national peak
HINDCAST_CUTOFFS = {2022: dt.date(2022, 12, 24),     # 2022-23 US peak week ending 2022-12-03 (early season)
                    2024: dt.date(2025, 3, 1),     # 2024-25 US peak week ending 2025-02-08
                    2025: dt.date(2026, 1, 24)}    # 2025-26 US peak week ending 2026-01-03
IMMUNE_PRIOR = (0.2, 1.8)        # hindcast prior on immune_scale (wider than the forecast's)
R_PRIOR = (0.8, 2.0)             # wider: a drifted strain can grow much faster than this season so far
MULT_LOG_SD = 1.0                # wider multiplier prior: lets admissions per infection differ by season
MIN_HALF_WIDTH = 0.2
INFORMATIVE_SD = 0.3             # prior sd of uniform(0.2, 1.8) is 0.46
MIN_PHI = 0.1
FLAT_HALF_LIFE = 1e6             # weeks: every week of the season counts equally
RUNS = C.REPO / "runs"


def out_dir(season: int) -> Path:
    return RUNS / f"hindcast-{season}-{str(season + 1)[2:]}"


def run_region(group: str, season: int, n_draws: int, n_parallel: int = 1) -> Path:
    from .pipeline import load_week
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s hindcast {season} {group} %(message)s")
    CB.HALF_LIFE_WEEKS = FLAT_HALF_LIFE          # these settings: this worker process only
    CB.MULT_LOG_SD = MULT_LOG_SD
    cutoff = HINDCAST_CUTOFFS[season]
    wk = load_week(cutoff + dt.timedelta(days=7), nowcast=False)   # final data, no backfill
    seed = zlib.crc32(f"{group}-{season}".encode()) % (2**31 - 2) + 1
    net = N.load(group)
    national = (wk.revised[wk.revised.location == "US"].set_index("date").value
                / wk.locations.set_index("location").population["US"])
    region = CB.setup(group, net, cutoff, cutoff, wk.locations, n_parallel=n_parallel, activity=wk.activity)
    prior = CB.Prior(immune_scale=IMMUNE_PRIOR, r_now=R_PRIOR)
    fit = CB.fit(region, wk.revised, national, cutoff, n_draws=n_draws, n_keep=1, seed=seed, prior=prior)
    d = out_dir(season)
    d.mkdir(parents=True, exist_ok=True)
    fit.theta.to_csv(d / f"{group}-draws.csv", index=False)
    w = fit.theta.weight.to_numpy()
    imm = fit.theta.immune_scale.to_numpy()
    m = float((w * imm).sum())
    (d / f"{group}.json").write_text(json.dumps({
        "ess": fit.ess, "phi": fit.phi, "cutoff": str(cutoff), "n": int(net.n), "pieces": region.pieces,
        "immune_mean": m, "immune_sd": float(np.sqrt((w * (imm - m) ** 2).sum())),
        "r_now": float((w * fit.theta.r_now).sum()), "humidity_floor": float((w * fit.theta.humidity_floor).sum()),
        "mult": {s: float(np.exp((w * np.log(fit.theta[f"mult_{s}"])).sum())) for s in region.pieces}}))
    return d / f"{group}.json"


def run_all(groups: list[str], seasons: list[int], workers: int, n_draws: int) -> None:
    jobs = [(g, s) for s in seasons for g in groups if not (out_dir(s) / f"{g}.json").exists()]
    log.info("hindcasts: %d region-seasons to run with %d workers", len(jobs), workers)
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(run_region, g, s, n_draws, workers): (g, s) for g, s in jobs}
        for f in as_completed(futs):
            g, s = futs[f]
            try:
                f.result()
                log.info("hindcast finished %s %d", g, s)
            except Exception as e:      # one failure shouldn't stop the rest
                log.error("hindcast %s %d failed: %s", g, s, e)


def summary(seasons: list[int] | None = None) -> pd.DataFrame:
    rows = []
    for s in seasons or list(HINDCAST_CUTOFFS):
        for f in sorted(out_dir(s).glob("*.json")):
            j = json.loads(f.read_text())
            rows.append({"group": f.stem, "season": s, **{k: j[k] for k in
                         ["ess", "phi", "immune_mean", "immune_sd", "r_now", "humidity_floor"]}})
    return pd.DataFrame(rows)


def prior_for(group: str, table: pd.DataFrame | None = None, default=(0.5, 1.5)) -> tuple[float, float]:
    """The current season's immune_scale prior for a region (see the module docstring)."""
    t = summary() if table is None else table
    t = t[(t.group == group) & (t.immune_sd < INFORMATIVE_SD) & (t.phi >= MIN_PHI)] if not t.empty else t
    if t.empty:
        return default
    c = float(t.immune_mean.mean())
    hw = max(MIN_HALF_WIDTH, float(t.immune_mean.max() - t.immune_mean.min()) + float(t.immune_sd.mean()))
    lo, hi = max(c - hw, 0.1), c + hw
    return (lo, hi)
