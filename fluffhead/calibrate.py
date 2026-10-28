"""Calibrating a region to admissions data, and forecasting from the fitted runs.

Targets. A region's counties can span several states; each (region, state) part is
a "piece". A piece's weekly target is its state's admissions times the piece's
share of the state's population (state data are nowcast for the last weeks).

Model admissions. Rather than the engine's random admissions, each symptomatic
resident infection contributes its admission probability (age, vaccination),
spread over the onset-to-admission delay. Each piece then gets a multiplier
(season severity x NHSN reporting in that state), estimated from the data in
closed form (gamma prior, Poisson-gamma conjugacy), so it needs no simulation.

Fitting. Draw N parameter sets from the prior, simulate each to the data cutoff,
weight by a negative binomial likelihood, and keep K by weighted resampling.

Forecast. Each kept run is replayed to the cutoff from its seed (cheap, and
identical) and continued M times with new seeds. Trajectories are expected
admissions; observation noise is added when quantiles are formed.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import special, stats

from . import calendar as K
from . import config as C
from . import immunity as I
from .disease import DiseaseParams
from .engine import Simulation
from .network import N_AGE, EdgePool, Network, choose_mode

log = logging.getLogger(__name__)

SEASON_START_MONTH = 8          # simulations start Aug 1
NB_SIZE = 20.0                  # negative binomial size for weekly counts (overdispersion)
MULT_PRIOR = (2.0, 2.0)         # gamma(shape, rate) prior on the per-piece admission multiplier
INFECTIONS_PER_ADMISSION = 150  # converts national admissions per capita to infection pressure
TRAVEL_SHARE = 0.02             # share of that pressure a resident meets from outside the region


@dataclass
class Prior:
    """Uniform (or log-uniform) priors on the calibrated parameters."""
    beta_scale: tuple = (1.8, 3.2)
    immune_scale: tuple = (0.5, 1.5)        # multiplies DiseaseParams.prior_immune
    import_scale: tuple = (0.25, 4.0)       # log-uniform; multiplies the national import pressure
    season_amplitude: tuple = (0.1, 0.35)

    def draw(self, rng, n) -> pd.DataFrame:
        u = lambda r: rng.uniform(r[0], r[1], n)
        return pd.DataFrame({"beta_scale": u(self.beta_scale), "immune_scale": u(self.immune_scale),
                             "import_scale": np.exp(u(np.log(self.import_scale))),
                             "season_amplitude": u(self.season_amplitude)})


# ------------------------------------------------------------------ setup
@dataclass
class Region:
    group: str
    net: Network
    start: dt.date
    days: int
    main_state: str
    pieces: list                 # state FIPS per piece
    piece_of: np.ndarray         # int8 per person: piece of a resident's county, -1 otherwise
    piece_scale: np.ndarray      # people represented per model resident, per piece
    piece_share: np.ndarray      # piece population / state population
    pool: EdgePool
    base: DiseaseParams = field(default_factory=DiseaseParams)


def season_start(day: dt.date) -> dt.date:
    y = day.year if day.month >= SEASON_START_MONTH else day.year - 1
    return dt.date(y, SEASON_START_MONTH, 1)


def setup(group: str, net: Network, ref_date: dt.date, horizon_end: dt.date, locations: pd.DataFrame,
          mode: str | None = None, n_parallel: int = 1, net_seed: int = 1) -> Region:
    counties = pd.read_csv(C.REGIONS_DIR / "counties.csv", dtype={"fips": str, "state": str}, keep_default_na=False)
    mine = counties[counties.group == group]
    state_pop2019 = counties.groupby("state")["pop"].sum()
    pieces = sorted(mine.state.unique())
    pidx = {s: i for i, s in enumerate(pieces)}
    cstate = np.array([c[:2] for c in net.counties])
    piece_of = np.array([pidx.get(s, -1) for s in cstate], np.int8)[net.county]
    piece_of[~net.resident] = -1
    # people per model resident: current hub population of the piece / model residents in it
    hubpop = locations.set_index("location")["population"]
    scale, share = [], []
    for s in pieces:
        piece_pop2019 = mine[mine.state == s]["pop"].sum()
        sh = piece_pop2019 / state_pop2019[s]
        share.append(sh)
        scale.append(sh * hubpop.get(s, state_pop2019[s]) / max((piece_of == pidx[s]).sum(), 1))
    start = season_start(ref_date)
    days = (horizon_end - start).days + 1
    main = mine.groupby("state")["pop"].sum().idxmax()
    mode = mode or choose_mode(net, n_parallel)
    return Region(group, net, start, days, main, pieces, piece_of, np.array(scale), np.array(share),
                  EdgePool.create(net, mode, net_seed))


def targets(region: Region, observed: pd.DataFrame, cutoff: dt.date) -> pd.DataFrame:
    """Weekly target per piece: state admissions x the piece's population share."""
    obs = observed[(observed.location.isin(region.pieces)) & (observed.date >= pd.Timestamp(region.start))
                   & (observed.date <= pd.Timestamp(cutoff))]
    w = obs.pivot_table(index="date", columns="location", values="value").reindex(columns=region.pieces)
    return w * region.piece_share


def import_pressure(region: Region, national: pd.Series, cutoff: dt.date) -> np.ndarray:
    """Per-person daily import hazard (before import_scale), from national admissions per capita."""
    days = pd.date_range(region.start, periods=region.days)
    weekly = national.sort_index()
    daily = weekly.reindex(weekly.index.union(days)).interpolate().bfill().ffill().reindex(days)
    daily[days > pd.Timestamp(cutoff)] = daily[days <= pd.Timestamp(cutoff)].iloc[-1]   # held after the cutoff
    return (daily.to_numpy() / 7) * INFECTIONS_PER_ADMISSION * TRAVEL_SHARE


# --------------------------------------------------------------- one run
def params_for(region: Region, theta) -> DiseaseParams:
    p = replace(region.base)
    p.beta_scale = float(theta.beta_scale)
    p.prior_immune = np.clip(region.base.prior_immune * theta.immune_scale, 0, 0.9)
    return p


def make_sim(region: Region, theta, pressure: np.ndarray, seed: int) -> Simulation:
    p = params_for(region, theta)
    cal = K.build(region.start, region.days, region.main_state, float(theta.season_amplitude))
    imm = I.build(region.net, p.prior_immune, region.start, region.start.year, region.main_state, seed)
    imp = pressure * float(theta.import_scale)
    return Simulation(region.net, p, cal, region.start, imm, imp, pool=region.pool, seed=seed)


def _delay_weights(p: DiseaseParams, max_days: int = 21) -> np.ndarray:
    k, m = p.onset_to_admit_shape, p.onset_to_admit_mean
    cdf = stats.gamma.cdf(np.arange(max_days + 1), k, scale=m / k)
    return np.diff(cdf) / cdf[-1]


def expected_admissions(region: Region, sim: Simulation) -> np.ndarray:
    """[days, pieces] expected admissions (before the multiplier), in real population units."""
    n = sim.rn
    j, _, _, _, sym, on, _, vac = (a[:n] for a in sim.rec)
    piece = region.piece_of[j]
    keep = sym & (piece >= 0)
    j, on, vac, piece = j[keep], on[keep], vac[keep], piece[keep]
    ag = region.net.age_group[j]
    ag = np.where(ag < 0, 2, ag)
    p = sim.params
    ph = p.hosp_per_symptomatic[ag] * np.where(vac, 1 - p.ve_hosp_extra, 1.0)
    w = _delay_weights(p)
    out = np.zeros((region.days + len(w), len(region.pieces)))
    d0 = np.floor(on).astype(int)
    keep = d0 < region.days          # onsets after the horizon (scheduled at the end) are dropped
    d0, piece, ph = d0[keep], piece[keep], ph[keep]
    for k, wk in enumerate(w):
        np.add.at(out, (d0 + k, piece), ph * wk)
    return out[:region.days] * region.piece_scale


def weekly(region: Region, daily: np.ndarray, weeks: pd.DatetimeIndex) -> np.ndarray:
    """Sum daily values over epiweeks (Sunday-Saturday) ending on `weeks`."""
    dates = pd.date_range(region.start, periods=region.days)
    out = np.zeros((len(weeks), daily.shape[1]))
    for i, w in enumerate(weeks):
        m = (dates > w - pd.Timedelta(days=7)) & (dates <= w)
        out[i] = daily[m].sum(0)
    return out


def score(mu: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Log likelihood with the per-piece multiplier set to its posterior mean; returns (ll, multipliers)."""
    ok = ~np.isnan(y)
    a, b = MULT_PRIOR
    mult = (np.where(ok, y, 0).sum(0) + a) / (np.where(ok, mu, 0).sum(0) + b)
    m = np.maximum(mu * mult, 1e-6)
    r = NB_SIZE
    ll = (special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1)
          + r * np.log(r / (r + m)) + y * np.log(m / (r + m)))
    return float(np.nansum(ll)), mult


# --------------------------------------------------------------- fitting
@dataclass
class Fit:
    region: str
    cutoff: dt.date
    theta: pd.DataFrame          # all draws, with log likelihood and weight
    kept: pd.DataFrame           # resampled draws (with seeds and multipliers)
    weeks: pd.DatetimeIndex
    observed: np.ndarray         # [weeks, pieces]
    ess: float


def fit(region: Region, observed: pd.DataFrame, national: pd.Series, cutoff: dt.date,
        n_draws: int = 200, n_keep: int = 20, seed: int = 1, prior: Prior | None = None) -> Fit:
    rng = np.random.default_rng(seed)
    prior = prior or Prior()
    th = prior.draw(rng, n_draws)
    th["seed"] = rng.integers(1, 2**31 - 1, n_draws)
    tgt = targets(region, observed, cutoff)
    y = tgt.to_numpy()
    pressure = import_pressure(region, national, cutoff)
    day_cut = (cutoff - region.start).days + 1
    lls, mults = [], []
    for i, t in enumerate(th.itertuples()):
        sim = make_sim(region, t, pressure, int(t.seed))
        sim.run_to(day_cut)
        mu = weekly(region, expected_admissions(region, sim), tgt.index)
        ll, mult = score(mu, y)
        lls.append(ll); mults.append(mult)
        if (i + 1) % max(n_draws // 10, 1) == 0:
            log.info("%s: %d/%d draws", region.group, i + 1, n_draws)
    th["loglik"] = lls
    w = np.exp(th.loglik - th.loglik.max())
    th["weight"] = w / w.sum()
    for k, s in enumerate(region.pieces):
        th[f"mult_{s}"] = [m[k] for m in mults]
    ess = float(1 / (th.weight**2).sum())
    idx = rng.choice(len(th), n_keep, replace=True, p=th.weight)
    kept = th.iloc[idx].reset_index(drop=True)
    log.info("%s: ESS %.1f of %d draws; kept %d (%d distinct)", region.group, ess, n_draws, n_keep, len(set(idx)))
    return Fit(region.group, cutoff, th, kept, tgt.index, y, ess)


def forecast(region: Region, fit_: Fit, national: pd.Series, n_cont: int = 5, seed: int = 2) -> dict:
    """Expected daily admissions per piece for each trajectory: {'daily': [traj, days, pieces], ...}."""
    rng = np.random.default_rng(seed)
    pressure = import_pressure(region, national, fit_.cutoff)
    day_cut = (fit_.cutoff - region.start).days + 1
    trajs, meta = [], []
    for i, t in enumerate(fit_.kept.itertuples()):
        mult = np.array([getattr(t, f"mult_{s}") for s in region.pieces])
        for c in range(n_cont):
            sim = make_sim(region, t, pressure, int(t.seed))
            sim.run_to(day_cut)
            sim.reseed(int(rng.integers(1, 2**31 - 1)))
            sim.run_to(region.days)
            trajs.append(expected_admissions(region, sim) * mult)
            meta.append((i, c))
    return {"daily": np.stack(trajs), "dates": pd.date_range(region.start, periods=region.days),
            "pieces": region.pieces, "meta": meta}
