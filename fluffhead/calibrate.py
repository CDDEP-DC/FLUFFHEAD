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

Parameters. Five are fitted, each with a prior on an interpretable scale:
  r_now           effective reproduction number behind the latest data: contacts,
                  vaccination and prior immunity at the cutoff, with the mean humidity
                  factor of the ANCHOR_DAYS days ending ANCHOR_LAG days before the cutoff
                  (the infections the last weeks' admissions reflect, so one unusually
                  dry or humid week just before the cutoff doesn't set the level). Each region measures how R
                  depends on transmissibility and immunity in a few short seeded runs
                  (RMap); a draw's beta_scale is solved from its r_now.
  immune_scale    multiplies the age-specific prior immunity.
  ve_infection    the vaccine's effect on susceptibility. Little of the fitted season has
                  vaccine in it, so data before November barely move this prior: it mostly
                  carries the uncertainty into the forecast. r_now is held at the draw's value
                  whatever its ve_infection (beta_scale).
  travel_share    iota in exposure.py: the average resident's share of infection risk met
                  outside the region (travel, visitors), applied to national incidence and
                  spread over people by their exposure (airport traffic, age, visitor-facing
                  work); commuters across the region boundary add a fixed share on top.
                  One tight prior for all regions.
  humidity_floor  rho in weather.factor: transmissibility at very high humidity relative
                  to dry air. Shaman et al. 2010 fitted 0.26-0.5 on 1972-2002 mortality, but
                  the observational association is modest. The prior is narrow (0.4-0.6, a
                  winter rise in R of about x1.2-1.3) because data before November can't
                  identify rho: humidity varies little before the cutoff, so a wide prior let
                  fits drift to no effect (rho ~0.8-1) and forecasts with almost no winter
                  rise, e.g. Maryland peaks a third of any recent season's.

Holiday rise. FORCING multiplies transmission by date, on top of humidity and the school and work
calendar: x1.2 from Dec 10 to Dec 31. Backtests on 2023-24 to 2025-26 (weekly cutoffs, 1-4 weeks
ahead) put it there: without it the fits under-call the December climb, and ending it Dec 24 made
the forecasts level off a week before the real peaks. The holiday lift is the same every year; it
is a calendar effect, not a fitted one.

Score. Negative binomial log likelihood of the weekly targets (recent weeks weighted
more), plus the multipliers' prior, plus a growth term: the log-linear slope of the
region's total admissions over the last GROWTH_WEEKS weeks, model vs data, with sd
GROWTH_SD plus the counts' Poisson noise. Weekly NB noise (~22% at size 20) is treated
as independent from week to week, which lets a run that grows too fast match the level
with a multiplier and overshoot only the last weeks; the slope term stops that.

Fitting (adaptive importance sampling). Round 1 draws from the prior; each later
round draws from a Gaussian fitted to the weighted draws so far (in a transformed
space where every prior is unbounded), mixed with the prior. All draws are pooled
with weights likelihood x prior / proposal. If the pooled effective sample size
is still below the target, the likelihood is tempered (raised to a power phi < 1)
until it reaches it, and phi is reported. K runs are kept by weighted resampling.

Strains (strains.py). With a variant, r_now is the variant's R where it leads at the
cutoff (the origin and the regions a reported share applies to; the dominant strain's
transmissibility is then solved from it, see beta_scale) and the dominant strain's R
elsewhere. The variant's start date, immune escape, transmissibility advantage and
vaccine-effect loss are fitted where it is seeded (four more parameters), and taken from
the origin's trajectories elsewhere. Imports
split between the strains weekly; an optional reported variant share adds a score term.

Forecast. Each kept run is replayed to the cutoff from its seed (cheap, and
identical) and continued M times with new seeds. After the cutoff the national import
pressure follows the run's own epidemic: each week it is the cutoff level times the
run's resident infections in the past week over those in the week to the cutoff (the
region standing in for the nation), so imports neither freeze nor grow on their own. Trajectories are expected
admissions; observation noise is added when quantiles are formed.
"""
from __future__ import annotations

import datetime as dt
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import special, stats

from . import calendar as K
from . import config as C
from . import ed as ED
from . import exposure as X
from . import immunity as I
from . import strains as ST
from . import weather as W
from .disease import DiseaseParams
from .engine import Simulation, StrainParams
from .network import N_AGE, EdgePool, Network, choose_mode

log = logging.getLogger(__name__)

SEASON_START_MONTH = 8          # simulations start Aug 1
NB_SIZE = 20.0                  # negative binomial size for weekly counts (overdispersion)
MULT_LOG_SD = 0.5               # log-normal prior on the per-piece admission multiplier (median 1)
MULT_CAP = 1e3                  # hard limit: multipliers stay within [1/MULT_CAP, MULT_CAP]
N_ROUNDS = 3                    # adaptive importance sampling rounds
VARIANT_DRAWS_FACTOR = 4        # where the variant's parameters are fitted (up to 9 in all): this many times the draws...
VARIANT_ROUNDS = 8              # ...in this many rounds
PRIOR_MIX = 0.2                 # share of each later round drawn from the prior (defensive)
PROPOSAL_INFLATE = 2.0          # proposal covariance = this x weighted covariance of the draws so far
MIN_ESS = 20.0                  # temper the likelihood until the effective sample size reaches this
BASELINE_WEEKS = 4              # weeks at the season start (August) that set each piece's background admissions
HALF_LIFE_WEEKS = 4.0           # weekly likelihood terms are down-weighted with this half-life back from the cutoff
GROWTH_WEEKS = 4                # growth term: log-linear slope of the region's admissions over the last weeks
GROWTH_SD = 0.05                # its extra sd (per week, log scale) beyond count noise; ~ x1.05/week
START_PSEUDO = 10.0             # start_correction: admissions added to both sides of the ratio, so small counts stay near 1
IMPORT_STEP_DAYS = 7            # forecast: national import pressure updated from the run's own incidence this often
IMPORT_MIN_REF = 20             # ...if the run had at least this many resident infections in the week to the cutoff
INFECTIONS_PER_ADMISSION = 75   # converts national admissions per capita to import pressure (backtests: 75 beat 150 and 37.5)
R_GRID_X = (1.25, 2.0, 3.0, 4.5, 6.5)   # RMap: beta_scale x seasonal factor
R_GRID_IMMUNE = (0.5, 1.0, 1.5)         # RMap: immune_scale
R_GRID_IMMUNE_VARIANT = (0.1, 0.3)      # ...plus these with a variant (immune_scale x (1 - escape))
R_YOUNG_POINTS = 3                       # RMap: escape-under-25 grid points (the "young" mechanism)
R_RUN_DAYS = 21                 # RMap runs: seeded at the cutoff, run this long
R_COHORT_DAYS = 7               # RMap: R = offspring per first-generation case infected in this window
ANCHOR_DAYS = 14                # r_now's humidity: mean seasonal factor over this many days...
MULT_CENTER = None              # log prior centre of the multipliers, per piece (backtests); None for 0
DIAG_DAYS = 14                  # per-draw diagnostics (draws csv): the days to the cutoff
ANCHOR_LAG = 7                  # ...ending this many days before the cutoff
FORCING: tuple = (((12, 10), (12, 31), 1.2),)   # ((month, day), (month, day), multiplier) of transmission by date:
                                # the holiday rise (backtests: Dec 10-31 x1.2 beat Dec 10-24 and none)
FALLBACK_AMPLITUDE = 0.2        # cosine seasonality when no humidity data


@dataclass
class Prior:
    """Uniform (or log-uniform) priors on the calibrated parameters."""
    r_now: tuple = (1.1, 1.3)               # seasonal influenza R ~1.2-1.4 in growth (Biggerstaff 2014); 0.9-1.3 forecast worse
    immune_scale: tuple = (0.5, 1.5)        # multiplies DiseaseParams.prior_immune
    travel_share: tuple = (0.01, 0.04)      # log-uniform
    humidity_floor: tuple = (0.4, 0.6)      # 1 would mean humidity has no effect
    ve_infection: tuple = (0.30, 0.65)      # vaccine effect on susceptibility
    variant_escape: tuple = ST.ESCAPE_PRIOR          # strains.py; fitted only where the variant is seeded
    variant_escape_young: tuple = (0.3, 0.7)         # the "young" mechanism
    variant_advantage: tuple = ST.ADVANTAGE_PRIOR
    variant_ve_loss: tuple = ST.VE_LOSS_PRIOR
    variant_start: tuple = (0.0, 60.0)               # days since the season start (set per region)
    variant_intro: tuple = ST.INTRO_PRIOR            # log-uniform
    extra: tuple = ()                                # variant parameters fitted in this region
    log_scale: tuple = ("travel_share", "variant_intro")

    @property
    def names(self) -> list[str]:
        return ["r_now", "immune_scale", "travel_share", "humidity_floor", "ve_infection"] + list(self.extra)

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
    scenario: ST.Scenario = field(default_factory=ST.Scenario)
    origin_track: ST.OriginTrack | None = None   # the variant's origin, for regions it spreads to
    link: float = 0.0                            # linkage to the origin (airlinks.py)
    asia: float = 0.0                            # arrivals per day from Japan / East Asia (airlinks.asia_arrivals)
    near: float = 0.0                            # origin: air share to the regions the reported share applies to
    national_path: np.ndarray | None = None      # [days] national incidence / the week to the cutoff, after the cutoff
    draw_workers: int = 1                        # processes for the region's runs (pmap)

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
          activity: pd.DataFrame | None = None, scenario: ST.Scenario | None = None,
          weather_as_of: dt.date | None = None, weather_today: dt.date | None = None) -> Region:
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
        humidity = W.region_humidity(group, start, days, today=weather_today, as_of=weather_as_of)
    except Exception as e:      # no weather: cosine seasonality
        log.warning("%s: humidity unavailable (%s); cosine seasonality", group, e)
        humidity = None
    base = DiseaseParams()
    scenario = scenario or ST.Scenario()
    base.hosp_per_symptomatic = base.hosp_per_symptomatic * scenario.severity()
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
                  cutoff=ref_date, import_travel=travel, import_commute=commute, community_mult=cmult,
                  scenario=scenario)


def region_prior(region: Region, immune_scale: tuple = (0.5, 1.5)) -> Prior:
    """The region's prior: the variant's parameters are added where it is seeded."""
    pr = Prior(immune_scale=immune_scale)
    if region.scenario.seeded_in(region.group):
        day_cut = (region.cutoff - region.start).days + 1
        mech = ST.MECHANISMS[region.scenario.mechanism]
        fitted = {k: v for k, v in mech.items() if isinstance(v, tuple)}
        pr = replace(pr, variant_start=(0.0, float(max(day_cut - ST.START_MIN_LEAD, 1))), **fitted,
                     extra=tuple(fitted) + ("variant_start", "variant_intro"))
    return pr


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
    return W.factor(region.humidity.path() if q is None else q, rho) * _forcing(region)


def _forcing(region: Region) -> np.ndarray:
    """Daily multiplier from FORCING, scaled to at most 1 (calendars thin events; beta_scale makes up for it)."""
    f = np.ones(region.days)
    days = [region.start + dt.timedelta(days=i) for i in range(region.days)]
    for (m0, d0), (m1, d1), x in FORCING:
        y = region.start.year
        a, b = dt.date(y, m0, d0), dt.date(y + (m1 < m0), m1, d1)
        f[[a <= d <= b for d in days]] *= x
    return f / f.max()


def season_now(region: Region, rho: float) -> float:
    """Mean seasonal factor over the ANCHOR_DAYS days ending ANCHOR_LAG days before the cutoff."""
    d = (region.cutoff - region.start).days - ANCHOR_LAG
    return float(season(region, rho)[max(d - ANCHOR_DAYS + 1, 0):max(d + 1, 1)].mean())


def _extrap(x0: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Linear interpolation, extended linearly past the ends."""
    if x0 <= xs[0]:
        return ys[0] + (x0 - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
    if x0 >= xs[-1]:
        return ys[-1] + (x0 - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(np.interp(x0, xs, ys))


def _invert_row(lr0: float, lr: np.ndarray, lx: np.ndarray) -> float:
    """log x for log R = lr0 on one row of the R map. Past the row's ends it follows the slope of the
    whole row, not of the last segment: in a small region noise can leave that segment flat, and
    its slope would send x to infinity. Bounded at a factor of 4 past the grid."""
    if lr[0] < lr0 < lr[-1]:
        return float(np.interp(lr0, lr, lx))
    slope = (lx[-1] - lx[0]) / max(lr[-1] - lr[0], 0.1)
    v = lx[-1] + (lr0 - lr[-1]) * slope if lr0 >= lr[-1] else lx[0] + (lr0 - lr[0]) * slope
    return float(np.clip(v, lx[0] - np.log(4), lx[-1] + np.log(4)))


@dataclass
class RMap:
    """Measured effective R at the cutoff on a grid of transmissibility x (beta_scale times the
    seasonal factor) and immune_scale; inverted in log-log space."""
    x: np.ndarray
    immune: np.ndarray
    r: np.ndarray                # [immune, x]
    vaccinated: float = 0.0      # share of residents vaccinated by the cutoff
    young: tuple | None = None   # (escape under 25 grid, equivalent immune multiplier) for escape by age

    def variant_immunity(self, escape: float, escape_young: float) -> float:
        """Multiplier of immune_scale for the variant's immunity: 1 - escape when escape is the same
        at all ages; otherwise the uniform multiplier with the same R (measured in r_map, at the
        mechanism's escape from 25; assumes it scales with immune_scale)."""
        if self.young is None or abs(escape_young - escape) < 1e-12:
            return 1.0 - escape
        return float(np.interp(escape_young, *self.young))

    def immune_for(self, r: float, b: int) -> float:
        """The immune_scale at which column b (x = self.x[b]) gives R = r."""
        col = np.maximum.accumulate(self.r[::-1, b])            # R rising as immunity falls
        return float(np.interp(r, col + 1e-9 * np.arange(len(col)), self.immune[::-1]))

    def x_for(self, r: float, immune: float) -> float:
        lx = np.log(self.x)
        per = []
        for row in self.r:
            lr = np.log(np.maximum.accumulate(np.maximum(row, 0.05)) + 1e-9 * np.arange(len(row)))
            per.append(_invert_row(np.log(r), lr, lx))
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


_JOB = None


def _call(i: int):
    return _JOB(i)


def pmap(fn, n: int, workers: int) -> list:
    """[fn(i) for i in range(n)], over `workers` forked processes when workers > 1. The runs of a fit
    round, an R map or a forecast are independent (each its own seed), so the results are the same
    as in one process. fn and the region it uses reach the workers by fork (Linux; on macOS fork is
    used too, but the default there is one process)."""
    global _JOB
    if workers <= 1 or n <= 1:
        return [fn(i) for i in range(n)]
    _JOB = fn
    try:
        with ProcessPoolExecutor(min(workers, n), mp_context=mp.get_context("fork")) as ex:
            return list(ex.map(_call, range(n)))
    finally:
        _JOB = None


def r_map(region: Region, seed: int = 1) -> RMap:
    d0 = (region.cutoff - region.start).days + 1
    grid = (R_GRID_IMMUNE_VARIANT if region.scenario.strains == 2 else ()) + R_GRID_IMMUNE
    def row(a):
        p_imm = np.clip(region.base.prior_immune * grid[a], 0, 0.9)
        imm = I.build(region.net, p_imm, region.start, region.start.year, region.main_state, seed + a)
        return [_measure_r(region, x, imm, p_imm, d0, seed + 10 * a + b) for b, x in enumerate(R_GRID_X)]
    mech = ST.MECHANISMS[region.scenario.mechanism]
    young = (region.scenario.variant_led(region.group) and "variant_escape_young" in mech)
    if young:                    # rows with escape by age (immune_scale 1), measured alongside
        yv = mech["variant_escape_young"]
        y_grid = tuple(np.linspace(*yv, R_YOUNG_POINTS)) if isinstance(yv, tuple) else (float(yv),)
        e_old = mech["variant_escape"]
        assert not isinstance(e_old, tuple), "escape by age needs a fixed escape from 25"
    else:
        y_grid = ()

    def row(a):
        if a < len(grid):
            p_imm = np.clip(region.base.prior_immune * grid[a], 0, 0.9)
        else:
            p_imm = np.clip(region.base.prior_immune * (1 - ST.escape_by_age(e_old, y_grid[a - len(grid)])), 0, 0.9)
        imm = I.build(region.net, p_imm, region.start, region.start.year, region.main_state, seed + a)
        return [_measure_r(region, x, imm, p_imm, d0, seed + 10 * a + b) for b, x in enumerate(R_GRID_X)]
    rows = np.array(pmap(row, len(grid) + len(y_grid), region.draw_workers))
    r = rows[:len(grid)]
    p_imm = np.clip(region.base.prior_immune * grid[-1], 0, 0.9)
    imm = I.build(region.net, p_imm, region.start, region.start.year, region.main_state, seed + len(grid) - 1)
    res = region.net.resident
    vac = float(((imm.vacc_day[res] >= 0) & (imm.vacc_day[res] < d0)).mean())
    log.info("%s: R at the cutoff by immune_scale (rows %s) and beta x season (%s):\n%s; %.0f%% vaccinated",
             region.group, grid, R_GRID_X, np.array2string(r, precision=2), 100 * vac)
    rm = RMap(np.array(R_GRID_X), np.array(grid), r, vac)
    if young:
        # equivalent uniform multiplier: per column, the immune_scale with the same R; median over columns
        m = [float(np.median([rm.immune_for(ry[b], b) for b in range(len(R_GRID_X))])) for ry in rows[len(grid):]]
        rm.young = (np.array(y_grid), np.array(m))
        log.info("%s: escape by age (from 25: %.2f): under-25 escape %s -> R as with immune_scale x %s; R rows\n%s",
                 region.group, e_old, np.round(y_grid, 2), np.round(m, 3),
                 np.array2string(rows[len(grid):], precision=2))
    return rm


# --------------------------------------------------------------- one run
def beta_scale(region: Region, theta) -> float:
    """The beta_scale that gives the draw's r_now (humidity of the anchor window), given its immunity
    and humidity floor. Where the variant leads (strains.py), r_now is the variant's R: its
    transmissibility is solved at its own immunity (immune_scale x (1 - escape)), after taking out
    the extra susceptibility of the vaccinated it gains from the vaccine-effect loss, and the
    dominant strain's is that over (1 + advantage)."""
    r, imm, adv = float(theta.r_now), float(theta.immune_scale), 0.0
    vac = region.r_map.vaccinated
    cv = vac * float(theta.ve_infection)     # share of residents the draw's vaccine protects
    loss = 0.0
    if region.scenario.variant_led(region.group):
        esc, adv, loss, _, esc_y = variant_of(region, theta)
        imm *= region.r_map.variant_immunity(esc, esc_y)
    r *= (1 - vac * region.base.ve_infection) / (1 - cv * (1 - loss))    # to the RMap's vaccine effect
    x = region.r_map.x_for(r, imm) / (1 + adv)
    return x / season_now(region, float(theta.humidity_floor))


def params_for(region: Region, theta) -> DiseaseParams:
    p = replace(region.base)
    p.beta_scale = beta_scale(region, theta)
    p.prior_immune = np.clip(region.base.prior_immune * theta.immune_scale, 0, 0.9)
    p.ve_infection = float(theta.ve_infection)
    return p


def variant_of(region: Region, theta) -> tuple | None:
    """(escape, advantage, vaccine-effect loss, origin trajectory or None, escape under 25) of the
    draw's variant; None with one strain."""
    sc = region.scenario
    if sc.strains == 1:
        return None
    if sc.seeded_in(region.group):
        mech = ST.MECHANISMS[sc.mechanism]
        val = lambda k: float(getattr(theta, k)) if isinstance(mech[k], tuple) else float(mech[k])
        esc = val("variant_escape")
        young = val("variant_escape_young") if "variant_escape_young" in mech else esc
        return esc, val("variant_advantage"), val("variant_ve_loss"), None, young
    tr = region.origin_track
    i = tr.pick(theta.seed)
    young = tr.escape_young[i] if tr.escape_young is not None else tr.escape[i]
    return float(tr.escape[i]), float(tr.advantage[i]), float(tr.ve_loss[i]), i, float(young)


def variant_timing(region: Region, theta) -> tuple[int, float]:
    """(start day, share of arriving East Asia travellers infected) of the draw's variant."""
    if region.scenario.seeded_in(region.group):
        return int(round(float(theta.variant_start))), float(theta.variant_intro)
    tr = region.origin_track
    i = tr.pick(theta.seed)
    return int(round(float(tr.start[i]))), float(tr.intro[i])


def _people_per_agent(region: Region) -> float:
    n = np.bincount(region.piece_of[region.piece_of >= 0], minlength=len(region.pieces))
    return float((n * region.piece_scale).sum() / max(n.sum(), 1))


def make_sim(region: Region, theta, pressure: np.ndarray, seed: int, q: np.ndarray | None = None) -> Simulation:
    """A run for draw `theta`; `q` is the humidity path (default: the region's central path).
    Imports are set as the run advances (advance())."""
    p = params_for(region, theta)
    cal = K.build(region.start, region.days, region.main_state,
                  season=season(region, float(theta.humidity_floor), q))
    v = variant_of(region, theta)
    imm = I.build(region.net, p.prior_immune, region.start, region.start.year, region.main_state, seed,
                  escape=None if v is None else ST.escape_by_age(v[0], v[4]))
    weight = float(theta.travel_share) * region.import_travel + region.import_commute
    strains, imp = None, pressure
    if v is not None:
        keep = 1.0 - v[2]
        strains = StrainParams(rel_beta=(1.0, 1.0 + v[1]), ve_inf=(p.ve_infection, p.ve_infection * keep),
                               ve_hosp=(p.ve_hosp_extra, p.ve_hosp_extra * keep), cross=ST.CROSS)
        imp = np.vstack([pressure, np.zeros_like(pressure)])
    return Simulation(region.net, p, cal, region.start, imm, imp, weight, pool=region.pool, seed=seed,
                      community_mult=region.community_mult, strains=strains)


def _delay_weights(p: DiseaseParams, max_days: int = 21) -> np.ndarray:
    k, m = p.onset_to_admit_shape, p.onset_to_admit_mean
    cdf = stats.gamma.cdf(np.arange(max_days + 1), k, scale=m / k)
    return np.diff(cdf) / cdf[-1]


def expected_admissions(region: Region, sim: Simulation) -> np.ndarray:
    """[days, pieces] expected admissions (before the multiplier), in real population units."""
    n = sim.rn
    j, _, _, _, sym, on, _, vac, strain = (a[:n] for a in sim.rec)
    piece = region.piece_of[j]
    keep = sym & (piece >= 0)
    j, on, vac, piece, strain = j[keep], on[keep], vac[keep], piece[keep], strain[keep]
    ag = region.net.age_group[j]
    ag = np.where(ag < 0, 2, ag)
    p = sim.params
    st = sim.strains
    veh = np.array(st.ve_hosp if st.ve_hosp is not None else [p.ve_hosp_extra] * st.n)
    ph = p.hosp_per_symptomatic[ag] * np.where(vac, 1 - veh[strain], 1.0)
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


def _slope(v: np.ndarray) -> float:
    x = np.arange(len(v)) - (len(v) - 1) / 2
    return float((x * np.log(v)).sum() / (x**2).sum())


def _recent(y: np.ndarray, m: np.ndarray | None = None):
    """Region totals over the last GROWTH_WEEKS fully reported weeks: (data, model, slope count variance)."""
    ok = ~np.isnan(y).any(1)
    yt = np.where(np.isnan(y), 0, y).sum(1)[ok][-GROWTH_WEEKS:]
    mt = None if m is None else m.sum(1)[ok][-GROWTH_WEEKS:]
    if len(yt) < 3 or yt.min() < 1:
        return yt, mt, None
    x = np.arange(len(yt)) - (len(yt) - 1) / 2
    return yt, mt, (x**2 / yt).sum() / (x**2).sum() ** 2            # slope variance from Poisson counts


def growth_term(m: np.ndarray, y: np.ndarray) -> float:
    """Log density of the model's recent growth rate given the data's (see the module docstring)."""
    yt, mt, v = _recent(y, m)
    if v is None:
        return 0.0
    return -0.5 * (_slope(mt) - _slope(yt)) ** 2 / (GROWTH_SD**2 + v)


def share_term(region: Region, sim: Simulation) -> float:
    """Log density of the reported variant share given the run's (strains.py), where it applies."""
    sc = region.scenario
    if not sc.share_applies(region.group):
        return 0.0
    d1 = (sc.variant_share_date - region.start).days + 1
    if d1 > sim.day:
        return 0.0
    n1, n2 = _infections_by_strain(region, sim, d1 - ST.SHARE_WINDOW_DAYS, d1)
    m = np.clip(n2 / max(n1 + n2, 1), 1e-4, 1 - 1e-4)
    o = np.clip(sc.variant_share, 1e-4, 1 - 1e-4)
    return float(-0.5 * ((special.logit(m) - special.logit(o)) / ST.SHARE_LOGIT_SD) ** 2)


def score(mu: np.ndarray, y: np.ndarray, base: np.ndarray) -> tuple[float, np.ndarray]:
    """Log likelihood (recent weeks weighted more, half-life HALF_LIFE_WEEKS), the
    multipliers' log prior and the growth term, with each piece's multiplier set to the
    same-weighted data / model ratio; returns (score, multipliers)."""
    ok = ~np.isnan(y)
    age = np.arange(len(y))[::-1, None]                              # weeks before the cutoff
    wt = 0.5 ** (age / HALF_LIFE_WEEKS) * ok
    excess = np.clip(np.where(ok, y, 0) - base, 0, None)
    mult = np.clip(((wt * excess).sum(0) + 1) / ((wt * np.where(ok, mu, 0)).sum(0) + 1), 1 / MULT_CAP, MULT_CAP)
    m = np.maximum(mu * mult + base, 1e-6)
    r = NB_SIZE
    ll = (special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1)
          + r * np.log(r / (r + m)) + y * np.log(m / (r + m)))
    log_prior = -0.5 * ((np.log(mult) - (0 if MULT_CENTER is None else MULT_CENTER)) / MULT_LOG_SD) ** 2   # normal prior on log multiplier
    return float(np.nansum(wt * ll) + log_prior.sum() + growth_term(m, y)), mult


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
              done: int, total: int) -> tuple[list, list, list]:
    rows = list(th.itertuples())

    def one(i):
        t = rows[i]
        sim = make_sim(region, t, pressure, int(t.seed))
        advance(region, sim, t, pressure, day_cut, day_cut)
        sc, mult = score(weekly(region, expected_admissions(region, sim), weeks), y, base)
        st = share_term(region, sim)
        if region.draw_workers <= 1 and (done + i + 1) % max(total // 10, 1) == 0:
            log.info("%s: %d/%d draws", region.group, done + i + 1, total)
        return sc + st, mult, {"score_data": sc, "score_share": st, **_diagnostics(region, sim, day_cut)}
    out = pmap(one, len(rows), region.draw_workers)
    if region.draw_workers > 1:
        log.info("%s: %d/%d draws (%d processes)", region.group, done + len(rows), total, region.draw_workers)
    return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]


def _diagnostics(region: Region, sim: Simulation, day_cut: int) -> dict:
    """Over the DIAG_DAYS days to the cutoff: resident infections, the share imported, the variant's share."""
    j, t, src, s = sim.rec[0][:sim.rn], sim.rec[1][:sim.rn], sim.rec[2][:sim.rn], sim.rec[8][:sim.rn]
    m = (t >= day_cut - DIAG_DAYS) & (t < day_cut) & region.net.resident[j]
    n = max(int(m.sum()), 1)
    return {"infections_recent": int(m.sum()), "imported_recent": float((m & (src < 0)).sum() / n),
            "variant_recent": float((m & (s == 1)).sum() / n)}


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
        sc, mults, diags = _evaluate(region, th, y, base, tgt.index, pressure, day_cut, sum(sizes[:r]), n_draws)
        th["score"] = sc
        for k in diags[0]:
            th[k] = [dg[k] for dg in diags]
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


def _resident_infections(region: Region, sim: Simulation, d0: int, d1: int) -> int:
    return sum(_infections_by_strain(region, sim, d0, d1))


def _infections_by_strain(region: Region, sim: Simulation, d0: int, d1: int) -> tuple[int, int]:
    j, t, s = sim.rec[0][:sim.rn], sim.rec[1][:sim.rn], sim.rec[8][:sim.rn]
    m = (t >= d0) & (t < d1) & region.net.resident[j]
    n2 = int((m & (s == 1)).sum())
    return int(m.sum()) - n2, n2


def advance(region: Region, sim: Simulation, theta, pressure: np.ndarray, d1: int, day_cut: int) -> None:
    """Run to day d1, setting imports a week at a time. Total pressure: national to the cutoff,
    then following the run's own resident infections (cutoff level x past week / week to the
    cutoff; held if fewer than IMPORT_MIN_REF). With a variant, split between the strains by the
    variant's share among those travellers meet, plus its seed infections (strains.py)."""
    v = variant_of(region, theta)
    if v is None and d1 <= day_cut:          # one strain, before the cutoff: national pressure as set
        sim.run_to(d1)
        return
    seeded = region.scenario.seeded_in(region.group)
    sim.imp = sim.imp.copy()
    W_ = sim.imp_cum[-1] if len(sim.imp_cum) else 0.0
    if v is not None:
        start, intro = variant_timing(region, theta)
        abroad = intro * region.asia / _people_per_agent(region)     # model infections per day from abroad
    ref = level = None
    d = sim.day
    while d < d1:
        nxt = min(d + IMPORT_STEP_DAYS, d1)
        if d < day_cut:
            nxt = min(nxt, day_cut)
            total = pressure[d:nxt]
        else:
            if ref is None:
                ref, level = _resident_infections(region, sim, day_cut - 7, day_cut), pressure[day_cut - 1]
            if region.national_path is not None:      # a national forecast (e.g. a first pass over all regions)
                f = float(region.national_path[d:nxt].mean())
            else:
                f = _resident_infections(region, sim, d - 7, d) / ref if ref >= IMPORT_MIN_REF else 1.0
            total = np.full(nxt - d, level * f)
        if v is None:
            sim.imp[0, d:nxt] = total
        else:
            n1, n2 = _infections_by_strain(region, sim, d - 7, d)
            own = n2 / (n1 + n2) if n1 + n2 > 0 else 0.0
            if not seeded:            # from the origin by linkage, from elsewhere as the region's own mix
                s2 = region.link * region.origin_track.share[v[3], max(d - 1, 0)] + (1 - region.link) * own
            elif region.scenario.origin == ST.EVERYWHERE:
                s2 = own              # seeded everywhere: the region stands in for the nation
            else:                     # the origin: its variant zone for `near`, the dominant strain elsewhere
                s2 = region.near * own
            s2 = min(max(s2, 0.0), 1.0)
            sim.imp[0, d:nxt] = total * (1 - s2)
            sim.imp[1, d:nxt] = total * s2
            if W_ > 0 and abroad > 0:
                a = max(d, start)
                if a < nxt:
                    sim.imp[1, a:nxt] += abroad / W_
            elif seeded and W_ > 0:
                a, b = max(d, start), min(nxt, start + ST.VARIANT_SEED_DAYS)
                if a < b:
                    sim.imp[1, a:b] += ST.VARIANT_SEEDS / ST.VARIANT_SEED_DAYS / W_
        sim.run_to(nxt)
        d = nxt


def start_correction(daily: np.ndarray, y: np.ndarray, day_cut: int, bg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start the forecast from the data. Fits run 10% or so above the last week (the score weights
    the whole recent curve), which would otherwise show as a jump from the nowcast to horizon 0.
    Each piece's epidemic after the cutoff is scaled by data / median model in the cutoff week:
    one factor for all trajectories, so their spread is kept. `daily` [traj, days, pieces], `y` the
    cutoff week's target per piece, `bg` background per day. Returns (corrected, factors)."""
    model = np.median(daily[:, day_cut - 7:day_cut].sum(1), axis=0)
    y = np.where(np.isnan(y), model, y)
    r = (y + START_PSEUDO) / (model + START_PSEUDO)
    out = daily.copy()
    out[:, day_cut:] = (out[:, day_cut:] - bg) * r + bg
    return out, r


def forecast(region: Region, fit_: Fit, national: pd.Series, n_cont: int = 5, seed: int = 2) -> dict:
    """Expected daily admissions per piece for each trajectory: {'daily': [traj, days, pieces], ...}."""
    rng = np.random.default_rng(seed)
    pressure = import_pressure(region, national, fit_.cutoff)
    day_cut = (fit_.cutoff - region.start).days + 1
    bg = (fit_.baseline if fit_.baseline is not None else np.zeros(len(region.pieces))) / 7   # per day
    rows = list(fit_.kept.itertuples())
    jobs = []                    # (kept draw, continuation, humidity path, seed after the cutoff), drawn in order
    for i in range(len(rows)):
        for c in range(n_cont):
            # each continuation its own weather after the forecast (identical up to the cutoff)
            q = region.humidity.path(rng) if region.humidity is not None else None
            jobs.append((i, c, q, int(rng.integers(1, 2**31 - 1))))

    def one(j):
        i, c, q, s = jobs[j]
        t = rows[i]
        mult = np.array([getattr(t, f"mult_{p}") for p in region.pieces])
        sim = make_sim(region, t, pressure, int(t.seed), q)
        advance(region, sim, t, pressure, day_cut, day_cut)
        sim.reseed(s)
        advance(region, sim, t, pressure, region.days, day_cut)
        v = variant_of(region, t)
        if v is None:
            return expected_admissions(region, sim) * mult + bg, None, None
        return (expected_admissions(region, sim) * mult + bg, ST.variant_share_daily(region, sim),
                v[:3] + variant_timing(region, t) + (v[4],))
    out = pmap(one, len(jobs), region.draw_workers)
    trajs = [o[0] for o in out]
    meta = [(i, c) for i, c, _, _ in jobs]
    vshare = [o[1] for o in out if o[1] is not None]
    vpar = [o[2] for o in out if o[2] is not None]
    daily = np.stack(trajs)
    last = fit_.weeks == pd.Timestamp(fit_.cutoff)
    if last.any():
        daily, r = start_correction(daily, fit_.observed[last][-1], day_cut, bg)
        log.info("%s: start correction by piece: %s", region.group, np.array2string(r, precision=2))
    out = {"daily": daily, "dates": pd.date_range(region.start, periods=region.days),
           "pieces": region.pieces, "meta": meta}
    if vshare:
        out["variant_share"] = np.stack(vshare)
        (out["variant_escape"], out["variant_advantage"], out["variant_ve_loss"], out["variant_start"],
         out["variant_intro"], out["variant_escape_young"]) = (np.array(x, float) for x in zip(*vpar))
    return out
