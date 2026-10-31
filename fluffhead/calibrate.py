"""Calibrating a region to admissions data, and forecasting from the fitted runs.

Targets. A region's counties can span several states; each (region, state) part is
a "piece". A piece's weekly target is its state's admissions times the piece's
share of the state's flu activity that week: its share of the state's population,
reweighted by NSSP flu ED visits by sub-state region (ed.py; population share where
there are no ED data). State data are nowcast for the last weeks.

Model admissions. Rather than the engine's random admissions, each symptomatic
resident infection contributes its admission probability (age, vaccination),
spread over the onset-to-admission delay. Each piece then gets a multiplier
(season severity x NHSN reporting in that state): the ratio of its data to the
model, with a log-normal prior centred on 1 that enters the score, so a run whose
epidemic is the wrong size can't be rescued by a large multiplier. A constant
background (sporadic admissions outside the seasonal epidemic, the mean of the
piece's first BASELINE_WEEKS weeks of the season) is added on top.

Parameters. Four are fitted, each with a prior on an interpretable scale:
  r_now           effective reproduction number at the cutoff (current contacts,
                  vaccination, prior immunity, humidity). Each region measures how R
                  depends on transmissibility and immunity in a few short seeded runs
                  (RMap); a draw's beta_scale is solved from its r_now.
  immune_scale    multiplies the age-specific prior immunity.
  travel_share    iota in exposure.py: the average resident's share of infection risk met
                  outside the region (travel, visitors), applied to national incidence and
                  spread over people by their exposure (airport traffic, age, visitor-facing
                  work); commuters across the region boundary add a fixed share on top.
                  One tight prior for all regions.
  humidity_floor  rho in weather.factor: transmissibility at very high humidity relative
                  to dry air (Shaman et al. 2010 best fits: R0min/R0max ~0.26-0.5). It
                  sets how much R changes as humidity changes after the cutoff.

Fitting (adaptive importance sampling). Round 1 draws from the prior; each later
round draws from a Gaussian fitted to the weighted draws so far (in a transformed
space where every prior is unbounded), mixed with the prior. All draws are pooled
with weights likelihood x prior / proposal. If the pooled effective sample size
is still below the target, the likelihood is tempered (raised to a power phi < 1)
until it reaches it, and phi is reported. K runs are kept by weighted resampling.

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
from . import ed as ED
from . import exposure as X
from . import immunity as I
from . import weather as W
from .disease import DiseaseParams
from .engine import Simulation
from .network import N_AGE, EdgePool, Network, choose_mode

log = logging.getLogger(__name__)

SEASON_START_MONTH = 8          # simulations start Aug 1
NB_SIZE = 20.0                  # negative binomial size for weekly counts (overdispersion)
MULT_LOG_SD = 0.5               # log-normal prior on the per-piece admission multiplier (median 1)
N_ROUNDS = 3                    # adaptive importance sampling rounds
PRIOR_MIX = 0.2                 # share of each later round drawn from the prior (defensive)
PROPOSAL_INFLATE = 2.0          # proposal covariance = this x weighted covariance of the draws so far
MIN_ESS = 20.0                  # temper the likelihood until the effective sample size reaches this
BASELINE_WEEKS = 4              # weeks at the season start (August) that set each piece's background admissions
HALF_LIFE_WEEKS = 4.0           # weekly likelihood terms are down-weighted with this half-life back from the cutoff
INFECTIONS_PER_ADMISSION = 150  # converts national admissions per capita to infection incidence
R_GRID_X = (1.25, 2.0, 3.0, 4.5, 6.5)   # RMap: beta_scale x seasonal factor
R_GRID_IMMUNE = (0.5, 1.0, 1.5)         # RMap: immune_scale
R_RUN_DAYS = 21                 # RMap runs: seeded at the cutoff, run this long
R_COHORT_DAYS = 7               # RMap: R = offspring per first-generation case infected in this window
NOW_DAYS = 7                    # humidity at the cutoff: mean seasonal factor over this many days
FALLBACK_AMPLITUDE = 0.2        # cosine seasonality when no humidity data


@dataclass
class Prior:
    """Uniform (or log-uniform) priors on the calibrated parameters."""
    r_now: tuple = (0.8, 1.5)               # seasonal influenza R ~1.2-1.4 in growth (Biggerstaff 2014)
    immune_scale: tuple = (0.5, 1.5)        # multiplies DiseaseParams.prior_immune
    travel_share: tuple = (0.01, 0.04)      # log-uniform
    humidity_floor: tuple = (0.25, 0.6)
    log_scale: tuple = ("travel_share",)

    @property
    def names(self) -> list[str]:
        return ["r_now", "immune_scale", "travel_share", "humidity_floor"]

    def _bounds(self, k):
        lo, hi = getattr(self, k)
        return (np.log(lo), np.log(hi)) if k in self.log_scale else (lo, hi)

    def to_z(self, th: pd.DataFrame) -> np.ndarray:
        """Parameters -> unbounded space: logit of the position within the (log) prior range."""
        cols = []
        for k in self.names:
            lo, hi = self._bounds(k)
            x = np.log(th[k].to_numpy()) if k in self.log_scale else th[k].to_numpy()
            u = np.clip((x - lo) / (hi - lo), 1e-9, 1 - 1e-9)
            cols.append(np.log(u / (1 - u)))
        return np.column_stack(cols)

    def from_z(self, z: np.ndarray) -> pd.DataFrame:
        out = {}
        for i, k in enumerate(self.names):
            lo, hi = self._bounds(k)
            x = lo + (hi - lo) / (1 + np.exp(-z[:, i]))
            out[k] = np.exp(x) if k in self.log_scale else x
        return pd.DataFrame(out)

    @staticmethod
    def log_density_z(z: np.ndarray) -> np.ndarray:
        """Log density in z of the (log-)uniform prior: each u = sigmoid(z) is uniform."""
        return np.sum(-z - 2 * np.log1p(np.exp(-z)), axis=1)

    def draw(self, rng, n) -> pd.DataFrame:
        u = rng.uniform(0, 1, (n, len(self.names)))
        return self.from_z(np.log(u / (1 - u)))


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
    activity_share: pd.DataFrame | None = None   # [season weeks, pieces] share of state flu activity
    humidity: W.Humidity | None = None           # daily specific humidity
    import_travel: np.ndarray | None = None      # per-person importation weights (exposure.py)
    import_commute: np.ndarray | None = None
    community_mult: np.ndarray | None = None     # per-tract community rate multiplier
    cutoff: dt.date | None = None
    r_map: RMap | None = None

    def shares_for(self, dates) -> np.ndarray:
        if self.activity_share is None:
            return np.tile(self.piece_share, (len(dates), 1))
        return self.activity_share.reindex(dates).fillna(pd.Series(self.piece_share, index=self.pieces)).to_numpy()

    def forecast_share(self) -> np.ndarray:
        """Share of each piece's state that the piece carries in forecast weeks."""
        return self.piece_share if self.activity_share is None else self.activity_share.iloc[-1].to_numpy()


def season_start(day: dt.date) -> dt.date:
    y = day.year if day.month >= SEASON_START_MONTH else day.year - 1
    return dt.date(y, SEASON_START_MONTH, 1)


def setup(group: str, net: Network, ref_date: dt.date, horizon_end: dt.date, locations: pd.DataFrame,
          mode: str | None = None, n_parallel: int = 1, net_seed: int = 1,
          activity: pd.DataFrame | None = None) -> Region:
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
    act_share = None
    if activity is not None and not activity.empty:
        first = start + dt.timedelta(days=(5 - start.weekday()) % 7)
        weeks = pd.date_range(first, horizon_end, freq="W-SAT")
        act_share = pd.DataFrame(ED.piece_shares(activity, [list(mine[mine.state == s].fips) for s in pieces],
                                                 pieces, weeks, np.array(share)), index=weeks, columns=pieces)
        log.info("%s: share of state flu activity (ED-weighted / population): %s", group,
                 ", ".join(f"{s} {a:.3f}/{p:.3f}" for s, a, p in zip(pieces, act_share.iloc[-1], share)))
    try:
        humidity = W.region_humidity(group, start, days)
    except Exception as e:      # no weather: cosine seasonality
        log.warning("%s: humidity unavailable (%s); cosine seasonality", group, e)
        humidity = None
    base = DiseaseParams()
    try:
        travel, commute = X.import_weights(group, net)
        cmult = X.community_multiplier(net, base.community_density_gamma)
        size = np.diff(net.tract_ptr)
        log.info("%s: community density multiplier: 10/50/90%% of people %s", group,
                 np.round(np.quantile(np.repeat(cmult, size), [0.1, 0.5, 0.9]), 2))
    except Exception as e:      # missing exposure data: uniform importation and contact
        log.warning("%s: exposure data unavailable (%s); uniform importation and community contact", group, e)
        travel, commute, cmult = net.resident.astype(float), np.zeros(net.n), None
    return Region(group, net, start, days, main, pieces, piece_of, np.array(scale), np.array(share),
                  EdgePool.create(net, mode, net_seed), base=base, activity_share=act_share, humidity=humidity,
                  cutoff=ref_date, import_travel=travel, import_commute=commute, community_mult=cmult)


def targets(region: Region, observed: pd.DataFrame, cutoff: dt.date) -> pd.DataFrame:
    """Weekly target per piece: state admissions x the piece's share of the state's flu activity."""
    obs = observed[(observed.location.isin(region.pieces)) & (observed.date >= pd.Timestamp(region.start))
                   & (observed.date <= pd.Timestamp(cutoff))]
    w = obs.pivot_table(index="date", columns="location", values="value").reindex(columns=region.pieces)
    return w * region.shares_for(w.index)


def import_pressure(region: Region, national: pd.Series, cutoff: dt.date) -> np.ndarray:
    """National infections per capita per day (before travel_share), from admissions per capita."""
    days = pd.date_range(region.start, periods=region.days)
    weekly = national.sort_index()
    daily = weekly.reindex(weekly.index.union(days)).interpolate().bfill().ffill().reindex(days)
    daily[days > pd.Timestamp(cutoff)] = daily[days <= pd.Timestamp(cutoff)].iloc[-1]   # held after the cutoff
    return (daily.to_numpy() / 7) * INFECTIONS_PER_ADMISSION


# ------------------------------------------------------------ R and season
def season(region: Region, rho: float, q: np.ndarray | None = None) -> np.ndarray:
    """Daily seasonal factor in (0, 1] from humidity (`q`, else the region's central path)."""
    if region.humidity is None:
        return np.array([K.seasonality(region.start + dt.timedelta(days=i), FALLBACK_AMPLITUDE)
                         for i in range(region.days)])
    return W.factor(region.humidity.path() if q is None else q, rho)


def season_now(region: Region, rho: float) -> float:
    """Mean seasonal factor over the NOW_DAYS days to the cutoff."""
    d = (region.cutoff - region.start).days
    return float(season(region, rho)[max(d - NOW_DAYS + 1, 0):d + 1].mean())


def _extrap(x0: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Linear interpolation, extended linearly past the ends."""
    if x0 <= xs[0]:
        return ys[0] + (x0 - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
    if x0 >= xs[-1]:
        return ys[-1] + (x0 - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(np.interp(x0, xs, ys))


@dataclass
class RMap:
    """Measured effective R at the cutoff on a grid of transmissibility x (beta_scale times the
    seasonal factor) and immune_scale; inverted in log-log space."""
    x: np.ndarray
    immune: np.ndarray
    r: np.ndarray                # [immune, x]

    def x_for(self, r: float, immune: float) -> float:
        lx = np.log(self.x)
        per = []
        for row in self.r:
            lr = np.log(np.maximum.accumulate(np.maximum(row, 0.05)) + 1e-9 * np.arange(len(row)))
            per.append(_extrap(np.log(r), lr, lx))
        return float(np.exp(_extrap(immune, self.immune, np.array(per))))


def _measure_r(region: Region, x: float, imm, p_imm: np.ndarray, d0: int, seed: int) -> float:
    p = replace(region.base)
    p.beta_scale = x
    p.prior_immune = p_imm
    days = d0 + R_RUN_DAYS
    cal = K.build(region.start, days, region.main_state, season=np.ones(days))
    n_seed = int(np.clip(region.net.n / 2000, 200, 1000))
    imp = np.zeros(days)
    imp[d0] = n_seed / max(region.net.resident.sum(), 1)    # seeds: residents at random
    sim = Simulation(region.net, p, cal, region.start, imm, imp, pool=region.pool, seed=seed,
                     community_mult=region.community_mult)
    sim.run_to(days)
    j, t, src = (a[:sim.rn] for a in sim.rec[:3])
    first = j[np.isin(src, j[src < 0]) & (t < d0 + R_COHORT_DAYS)]
    return float(np.isin(src, first).sum() / max(len(first), 1))


def r_map(region: Region, seed: int = 1) -> RMap:
    d0 = (region.cutoff - region.start).days + 1
    r = np.zeros((len(R_GRID_IMMUNE), len(R_GRID_X)))
    for a, s in enumerate(R_GRID_IMMUNE):
        p_imm = np.clip(region.base.prior_immune * s, 0, 0.9)
        imm = I.build(region.net, p_imm, region.start, region.start.year, region.main_state, seed + a)
        for b, x in enumerate(R_GRID_X):
            r[a, b] = _measure_r(region, x, imm, p_imm, d0, seed + 10 * a + b)
    log.info("%s: R at the cutoff by immune_scale (rows %s) and beta x season (%s):\n%s", region.group,
             R_GRID_IMMUNE, R_GRID_X, np.array2string(r, precision=2))
    return RMap(np.array(R_GRID_X), np.array(R_GRID_IMMUNE), r)


# --------------------------------------------------------------- one run
def beta_scale(region: Region, theta) -> float:
    """The beta_scale that gives the draw's r_now at the cutoff, given its immunity and humidity floor."""
    x = region.r_map.x_for(float(theta.r_now), float(theta.immune_scale))
    return x / season_now(region, float(theta.humidity_floor))


def params_for(region: Region, theta) -> DiseaseParams:
    p = replace(region.base)
    p.beta_scale = beta_scale(region, theta)
    p.prior_immune = np.clip(region.base.prior_immune * theta.immune_scale, 0, 0.9)
    return p


def make_sim(region: Region, theta, pressure: np.ndarray, seed: int, q: np.ndarray | None = None) -> Simulation:
    """A run for draw `theta`; `q` is the humidity path (default: the region's central path)."""
    p = params_for(region, theta)
    cal = K.build(region.start, region.days, region.main_state,
                  season=season(region, float(theta.humidity_floor), q))
    imm = I.build(region.net, p.prior_immune, region.start, region.start.year, region.main_state, seed)
    weight = float(theta.travel_share) * region.import_travel + region.import_commute
    return Simulation(region.net, p, cal, region.start, imm, pressure, weight, pool=region.pool, seed=seed,
                      community_mult=region.community_mult)


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


def baseline(y: np.ndarray) -> np.ndarray:
    """[pieces] weekly background admissions: mean of the first BASELINE_WEEKS reported weeks."""
    b = np.array([np.nanmean(c[~np.isnan(c)][:BASELINE_WEEKS]) if (~np.isnan(c)).any() else 0.0 for c in y.T])
    return np.nan_to_num(b)


def score(mu: np.ndarray, y: np.ndarray, base: np.ndarray) -> tuple[float, np.ndarray]:
    """Log likelihood (recent weeks weighted more, half-life HALF_LIFE_WEEKS) plus the
    multipliers' log prior, with each piece's multiplier set to the same-weighted
    data / model ratio; returns (score, multipliers)."""
    ok = ~np.isnan(y)
    age = np.arange(len(y))[::-1, None]                              # weeks before the cutoff
    wt = 0.5 ** (age / HALF_LIFE_WEEKS) * ok
    excess = np.clip(np.where(ok, y, 0) - base, 0, None)
    mult = np.clip(((wt * excess).sum(0) + 1) / ((wt * np.where(ok, mu, 0)).sum(0) + 1), 1e-3, 1e3)
    m = np.maximum(mu * mult + base, 1e-6)
    r = NB_SIZE
    ll = (special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1)
          + r * np.log(r / (r + m)) + y * np.log(m / (r + m)))
    log_prior = -0.5 * (np.log(mult) / MULT_LOG_SD) ** 2          # normal prior on log multiplier
    return float(np.nansum(wt * ll) + log_prior.sum()), mult


# --------------------------------------------------------------- fitting
@dataclass
class Fit:
    region: str
    cutoff: dt.date
    theta: pd.DataFrame          # all draws, with score, round and weight
    kept: pd.DataFrame           # resampled draws (with seeds and multipliers)
    weeks: pd.DatetimeIndex
    observed: np.ndarray         # [weeks, pieces]
    ess: float
    phi: float = 1.0             # likelihood tempering power (1 = none)
    baseline: np.ndarray | None = None   # [pieces] weekly background admissions


def _mvn_logpdf(z: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return stats.multivariate_normal(mean, cov, allow_singular=True).logpdf(z).reshape(len(z))


def _weights(log_target: np.ndarray, phi: float, log_correction: np.ndarray) -> np.ndarray:
    lw = phi * log_target + log_correction
    w = np.exp(lw - lw.max())
    return w / w.sum()


def _temper(score_: np.ndarray, log_correction: np.ndarray, min_ess: float) -> tuple[float, np.ndarray]:
    """Largest phi in (0, 1] whose weights have ESS >= min_ess (bisection on log phi)."""
    ess = lambda w: 1 / (w**2).sum()
    w = _weights(score_, 1.0, log_correction)
    if ess(w) >= min_ess:
        return 1.0, w
    lo, hi = -12.0, 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if ess(_weights(score_, np.exp(mid), log_correction)) >= min_ess:
            lo = mid
        else:
            hi = mid
    return float(np.exp(lo)), _weights(score_, np.exp(lo), log_correction)


def _evaluate(region: Region, th: pd.DataFrame, y: np.ndarray, base: np.ndarray, weeks, pressure, day_cut: int,
              done: int, total: int) -> tuple[list, list]:
    scores, mults = [], []
    for i, t in enumerate(th.itertuples()):
        sim = make_sim(region, t, pressure, int(t.seed))
        sim.run_to(day_cut)
        sc, mult = score(weekly(region, expected_admissions(region, sim), weeks), y, base)
        scores.append(sc); mults.append(mult)
        if (done + i + 1) % max(total // 10, 1) == 0:
            log.info("%s: %d/%d draws", region.group, done + i + 1, total)
    return scores, mults


def fit(region: Region, observed: pd.DataFrame, national: pd.Series, cutoff: dt.date,
        n_draws: int = 300, n_keep: int = 20, seed: int = 1, prior: Prior | None = None,
        n_rounds: int = N_ROUNDS) -> Fit:
    rng = np.random.default_rng(seed)
    prior = prior or Prior()
    region.cutoff = cutoff
    region.r_map = r_map(region, seed)
    tgt = targets(region, observed, cutoff)
    y = tgt.to_numpy()
    base = baseline(y)
    pressure = import_pressure(region, national, cutoff)
    day_cut = (cutoff - region.start).days + 1
    sizes = [n_draws // n_rounds + (r < n_draws % n_rounds) for r in range(n_rounds)]
    rounds, z_all, logq_parts = [], [], []
    mean = cov = None
    for r, n in enumerate(sizes):
        if r == 0:
            z = prior.to_z(prior.draw(rng, n))
        else:
            from_prior = rng.random(n) < PRIOR_MIX
            z = rng.multivariate_normal(mean, cov, n)
            z[from_prior] = prior.to_z(prior.draw(rng, int(from_prior.sum())))
        th = prior.from_z(z)
        th["seed"] = rng.integers(1, 2**31 - 1, n)
        th["round"] = r + 1
        sc, mults = _evaluate(region, th, y, base, tgt.index, pressure, day_cut, sum(sizes[:r]), n_draws)
        th["score"] = sc
        for k, s in enumerate(region.pieces):
            th[f"mult_{s}"] = [m[k] for m in mults]
        rounds.append(th); z_all.append(z)
        logq_parts.append((mean, cov))
        # importance weights so far: target x prior / proposal (each draw's own round's proposal)
        allz = np.vstack(z_all)
        allth = pd.concat(rounds, ignore_index=True)
        logp = Prior.log_density_z(allz)
        logq = np.concatenate([logp[:sizes[0]]] + [
            np.logaddexp(np.log(PRIOR_MIX) + logp[sum(sizes[:i]):sum(sizes[:i + 1])],
                         np.log(1 - PRIOR_MIX) + _mvn_logpdf(allz[sum(sizes[:i]):sum(sizes[:i + 1])], m_, c_))
            for i, (m_, c_) in enumerate(logq_parts[1:], start=1)])
        phi, w = _temper(allth.score.to_numpy(), logp - logq, MIN_ESS)
        mean = (w[:, None] * allz).sum(0)
        d = allz - mean
        cov = PROPOSAL_INFLATE * (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) + 1e-4 * np.eye(allz.shape[1])
        log.info("%s: round %d: ESS %.1f, phi %.3f", region.group, r + 1, 1 / (w**2).sum(), phi)
    th = allth
    th["weight"] = w
    th["beta_scale"] = [beta_scale(region, t) for t in th.itertuples()]
    ess = float(1 / (w**2).sum())
    idx = rng.choice(len(th), n_keep, replace=True, p=w)
    kept = th.iloc[idx].reset_index(drop=True)
    log.info("%s: ESS %.1f of %d draws (phi %.3f); kept %d (%d distinct); multipliers %s", region.group, ess,
             n_draws, phi, n_keep, len(set(idx)),
             ", ".join(f"{s} {np.exp((w * np.log(th[f'mult_{s}'])).sum()):.2f}" for s in region.pieces))
    log.info("%s: weighted means: %s", region.group,
             ", ".join(f"{k} {(w * th[k]).sum():.3g}" for k in prior.names + ["beta_scale"]))
    log.info("%s: background admissions per week %s", region.group,
             ", ".join(f"{s} {b:.1f}" for s, b in zip(region.pieces, base)))
    return Fit(region.group, cutoff, th, kept, tgt.index, y, ess, phi, base)


def forecast(region: Region, fit_: Fit, national: pd.Series, n_cont: int = 5, seed: int = 2) -> dict:
    """Expected daily admissions per piece for each trajectory: {'daily': [traj, days, pieces], ...}."""
    rng = np.random.default_rng(seed)
    pressure = import_pressure(region, national, fit_.cutoff)
    day_cut = (fit_.cutoff - region.start).days + 1
    trajs, meta = [], []
    bg = (fit_.baseline if fit_.baseline is not None else np.zeros(len(region.pieces))) / 7   # per day
    for i, t in enumerate(fit_.kept.itertuples()):
        mult = np.array([getattr(t, f"mult_{s}") for s in region.pieces])
        for c in range(n_cont):
            # each continuation its own weather after the forecast (identical up to the cutoff)
            q = region.humidity.path(rng) if region.humidity is not None else None
            sim = make_sim(region, t, pressure, int(t.seed), q)
            sim.run_to(day_cut)
            sim.reseed(int(rng.integers(1, 2**31 - 1)))
            sim.run_to(region.days)
            trajs.append(expected_admissions(region, sim) * mult + bg)
            meta.append((i, c))
    return {"daily": np.stack(trajs), "dates": pd.date_range(region.start, periods=region.days),
            "pieces": region.pieces, "meta": meta}
