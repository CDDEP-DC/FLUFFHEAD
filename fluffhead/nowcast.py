# Vendored from flusight-fm (flusight_fm/nowcast.py, commit 4c5e8a5, CC0-1.0). Keep in sync by hand;
# FLUFFHEAD does not depend on the flusight-fm package (it needs torch).
"""Backfill nowcast: estimate what the most recent, still-incomplete weeks will be revised to.

For each of the last NOWCAST_WEEKS reported weeks, two estimates are combined,
each a normal distribution on log(value + offset):

1. The latest report, adjusted by a revision model learned from past versions of
   the hub's target data. Each version's report is compared with the value the
   newest version has once that week is REVISION_SETTLE_WEEKS old, separately for
   each lag (0 = newest week). The model (NOWCAST_MODEL, per dataset) is one of:

   - "none":        no shift; only the revision uncertainty is used
   - "constant":    a fixed log shift per lag
   - "size":        shift linear in log(report)
   - "size+growth": also depends on the week-over-week change in the report, since a
                    sharp apparent drop is often an incomplete report

   plus a per-location intercept shrunk toward zero (REVISION_SHRINK). The
   revision s.d. on the log scale is a power law in the report's size,
   c * ((report + offset) / offset)^-k (NOWCAST_VARIANCE = "power"; "constant"
   uses one s.d. per lag). With k near 0.2 for hospital admissions, revisions in
   counts grow like count^0.8: less than proportionally, at every size.
   See `flusight revisions` for the analysis behind these choices.
2. Our own earlier forecast for that week (from last week's submission), if
   there is one. It is weighted by precision and scaled by PRIOR_WEIGHT.

The models are then run on the series with these weeks replaced by the nowcast
medians, and horizon -1 is submitted from the nowcast distribution.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

GROWTH_CLIP = 1.5      # |log week-over-week change| beyond this is treated as this
CALIBRATION = 0.8      # the s.d. is scaled so this share of past revisions falls in the central interval
MODEL_KINDS = ("none", "constant", "size", "size+growth")


def _growth(rep, prev, offset):
    g = np.log(np.asarray(rep, float) + offset) - np.log(np.asarray(prev, float) + offset)
    return np.clip(np.nan_to_num(g, nan=0.0), -GROWTH_CLIP, GROWTH_CLIP)


def revision_errors(vintages: dict[dt.date, pd.DataFrame], as_of: dt.date, offset: float,
                    lags: int | None = None, truth: pd.DataFrame | None = None, settle: int | None = None,
                    start: str | None = None) -> pd.DataFrame:
    """One row per (version, location, lag): the report, the week-over-week growth it showed, and
    err = log(settled value + offset) - log(report + offset).

    By default only versions from REVISION_START to `as_of` are used, and the "settled" value is
    the newest of those versions' value for weeks at least REVISION_SETTLE_WEEKS old. Pass `truth`
    to compare against other data (e.g. the latest data, for evaluation)."""
    lags = C.NOWCAST_WEEKS if lags is None else lags
    settle = C.REVISION_SETTLE_WEEKS if settle is None else settle
    start = pd.Timestamp(start or C.REVISION_START).date()
    avail = {r: df.dropna(subset=["value"]) for r, df in vintages.items() if start <= r <= as_of}
    cols = ["ref", "location", "date", "lag", "rep", "growth", "fin", "err"]
    if len(avail) < 3 and truth is None:
        return pd.DataFrame(columns=cols)
    if truth is None:
        truth = avail.pop(max(avail))
    settled = truth["date"].max() - pd.Timedelta(weeks=settle)
    fin = truth.dropna(subset=["value"]).drop_duplicates(["location", "date"], keep="last").set_index(["location", "date"])["value"]

    rows = []
    for ref, df in avail.items():
        wide = df.drop_duplicates(["location", "date"], keep="last").pivot(index="date", columns="location", values="value")
        last = wide.index.max()
        for lag in range(lags):
            d = last - pd.Timedelta(weeks=lag)
            if d > settled or d not in wide.index:
                continue
            rep = wide.loc[d].dropna()
            before = d - pd.Timedelta(weeks=1)
            prev = wide.loc[before].reindex(rep.index) if before in wide.index else pd.Series(np.nan, index=rep.index)
            f = fin.reindex(pd.MultiIndex.from_product([rep.index, [d]])).to_numpy()
            rows.append(pd.DataFrame({"ref": ref, "location": rep.index, "date": d, "lag": lag, "rep": rep.to_numpy(),
                                      "growth": _growth(rep, prev, offset), "fin": f}))
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.concat(rows, ignore_index=True).dropna(subset=["fin"])
    return out.assign(err=np.log(out["fin"] + offset) - np.log(out["rep"] + offset))[cols]


def _huber(X: np.ndarray, y: np.ndarray, c: float = 1.345, iters: int = 30) -> np.ndarray:
    """Robust linear regression (Huber loss, iteratively reweighted least squares)."""
    w = np.ones(len(y))
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r / s)
        w = np.where(u <= c, 1.0, c / u)
    return beta


def _nnls2(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with two non-negative coefficients."""
    best, best_loss = np.zeros(2), np.inf
    full = np.linalg.lstsq(A, y, rcond=None)[0]
    candidates = [full] if (full >= 0).all() else []
    for j in range(2):
        b = np.zeros(2)
        b[j] = max(float(A[:, j] @ y / (A[:, j] @ A[:, j])), 0.0)
        candidates.append(b)
    for b in candidates:
        loss = float(np.sum((y - A @ b) ** 2))
        if loss < best_loss:
            best, best_loss = b, loss
    return best


def _gauss_loglik(r2: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(var, 1e-8)
    return float(np.sum(-0.5 * (np.log(2 * np.pi * var) + r2 / var)))


def fit_power_variance(rep: np.ndarray, r2: np.ndarray, offset: float) -> tuple[float, float]:
    """(c, k) maximizing the normal likelihood of squared residuals r2 under
    var = c * ((rep + offset) / offset)^(-2k), with k profiled over a grid in [0, 1]."""
    size = (np.asarray(rep, float) + offset) / offset
    best = (float(r2.mean()), 0.0, -np.inf)
    for k in np.linspace(0, 1, 101):
        f = size ** (-2 * k)
        c = float(np.mean(r2 / f))   # maximum-likelihood c for this k
        ll = _gauss_loglik(r2, c * f)
        if ll > best[2]:
            best = (c, float(k), ll)
    return best[0], best[1]


@dataclass
class LagModel:
    """Revision model for one lag, on the log(value + offset) scale."""
    kind: str
    offset: float
    beta: np.ndarray                 # intercept, then size, growth-down, growth-up coefficients as used
    center: float = 0.0              # log size the size coefficient is centered on
    size_range: tuple = (-np.inf, np.inf)
    loc_effect: dict = field(default_factory=dict)
    var: tuple = (C.NOWCAST_FALLBACK_SD ** 2, 0.0)   # (c, k): variance = c * ((report + offset) / offset)^(-2k)
    scale: float = 1.0
    n: int = 0

    def _design(self, rep, growth) -> np.ndarray:
        rep, growth = np.atleast_1d(np.asarray(rep, float)), np.atleast_1d(np.asarray(growth, float))
        cols = [np.ones(len(rep))]
        if self.kind in ("size", "size+growth"):
            z = np.clip(np.log(rep + self.offset), *self.size_range)
            cols.append(z - self.center)
        if self.kind == "size+growth":
            cols += [np.minimum(growth, 0), np.maximum(growth, 0)]
        return np.column_stack(cols)

    def mean(self, rep, growth, locations=None) -> np.ndarray:
        m = self._design(rep, growth) @ self.beta
        if locations is not None:
            m = m + np.array([self.loc_effect.get(loc, 0.0) for loc in np.atleast_1d(locations)])
        return m

    def raw_sd(self, rep) -> np.ndarray:
        size = (np.maximum(np.atleast_1d(np.asarray(rep, float)), 0) + self.offset) / self.offset
        return np.maximum(np.sqrt(self.var[0] * size ** (-2 * self.var[1])), C.REVISION_MIN_SD)

    def sd(self, rep) -> np.ndarray:
        return self.scale * self.raw_sd(rep)


def fit_lag(g: pd.DataFrame, kind: str, offset: float, variance: str | None = None) -> LagModel:
    """Fit one lag's revision model to rows of `revision_errors`."""
    variance = variance or C.NOWCAST_VARIANCE
    y = g["err"].to_numpy(float)
    z = np.log(g["rep"].to_numpy(float) + offset)
    m = LagModel(kind=kind, offset=offset, beta=np.zeros(1), center=float(np.median(z)),
                 size_range=(float(z.min()), float(z.max())), n=len(g))
    if kind == "none":
        res = y
    else:
        n_coef = {"constant": 1, "size": 2, "size+growth": 4}[kind]
        m.beta = np.zeros(n_coef)
        X = m._design(g["rep"], g["growth"])
        m.beta = np.array([np.median(y)]) if kind == "constant" else _huber(X, y)
        res = y - X @ m.beta
        for loc, idx in g.groupby("location").indices.items():
            w = len(idx) / (len(idx) + C.REVISION_SHRINK)
            m.loc_effect[loc] = w * float(np.median(res[idx]))
        res = res - g["location"].map(m.loc_effect).to_numpy()
    lo, hi = np.quantile(res, [0.025, 0.975])
    r2 = np.clip(res, lo, hi) ** 2
    m.var = fit_power_variance(g["rep"].to_numpy(float), r2, offset) if variance == "power" else (float(r2.mean()), 0.0)
    # z of the CALIBRATION central interval, e.g. 1.2816 for 80%
    z_cal = {0.5: 0.6745, 0.8: 1.2816, 0.9: 1.6449}[CALIBRATION]
    m.scale = float(np.quantile(np.abs(res) / m.raw_sd(g["rep"]), CALIBRATION) / z_cal) or 1.0
    return m


def fit_revisions(vintages, as_of, offset, kind: str = "constant") -> dict[int, LagModel]:
    """{lag: LagModel} learned from data versions available on `as_of`."""
    errs = revision_errors(vintages, as_of, offset)
    models = {int(lag): fit_lag(g, kind, offset) for lag, g in errs.groupby("lag") if len(g) >= 20}
    if models:
        def show(m: LagModel):
            sizes = [s for s in (10, 100, 1000) if np.log(s + offset) <= m.size_range[1]] if offset >= 1 else [0.005, 0.02, 0.08]
            return ", ".join(f"{s:g}: {float(m.mean(s, 0.0)[0]):+.3f} (sd {float(m.sd(s)[0]):.3f})" for s in sizes)
        log.info("Revision model (%s): %s", kind, "; ".join(f"lag {lag} [{show(m)}] n={m.n}" for lag, m in models.items()))
    else:
        log.info("No revision history available; nowcast uses the reports as they are")
    return models


def prior_forecasts(prior_dir: Path | None, target: str, dates, offset: float) -> dict:
    """{(location, date): (mu, sd)} on the log scale, from earlier submissions' horizon-0 forecasts."""
    out = {}
    if prior_dir is None or C.PRIOR_WEIGHT <= 0:
        return out
    folder = Path(prior_dir) / "model-output" / C.model_id()
    for d in dates:
        f = folder / f"{pd.Timestamp(d).date()}-{C.model_id()}.csv"
        if not f.exists():
            continue
        sub = pd.read_csv(f, dtype={"location": str, "output_type_id": str})
        sub = sub[(sub["target"] == target) & (sub["horizon"] == 0) & (sub["output_type"] == "quantile")]
        wide = sub.assign(q=sub["output_type_id"].astype(float)).pivot_table(index="location", columns="q", values="value")
        if not {0.25, 0.5, 0.75} <= set(wide.columns):
            continue
        lq = np.log(wide[[0.25, 0.5, 0.75]].clip(lower=0) + offset)
        sd = ((lq[0.75] - lq[0.25]) / 1.349).clip(lower=1e-3)
        for loc in wide.index:
            out[loc, pd.Timestamp(d)] = (float(lq.at[loc, 0.5]), float(sd[loc]))
    return out


def nowcast(series: dict[str, pd.Series], models: dict[int, LagModel], prior: dict, offset: float, adjust: bool = True):
    """Returns (revised series, table of nowcasts for the last NOWCAST_WEEKS weeks)."""
    rows, revised = [], {}
    for loc, s in series.items():
        s2 = s.copy()
        for lag in range(min(C.NOWCAST_WEEKS, len(s))):
            d, rep = s.index[-1 - lag], float(s.iloc[-1 - lag])
            prev = float(s.iloc[-2 - lag]) if len(s) > lag + 1 else np.nan
            m = models.get(lag)
            growth = float(_growth(rep, prev, offset))
            shift = float(m.mean(rep, growth, [loc])[0]) if m is not None and adjust else 0.0
            sd = float(m.sd(rep)[0]) if m is not None else C.NOWCAST_FALLBACK_SD
            mu = np.log(rep + offset) + shift
            p = prior.get((loc, d)) if adjust else None
            if p is not None:
                pm, psd = p[0], p[1] / np.sqrt(C.PRIOR_WEIGHT)
                prec = 1 / sd**2 + 1 / psd**2
                mu, sd = (mu / sd**2 + pm / psd**2) / prec, np.sqrt(1 / prec)
            med = max(float(np.exp(mu) - offset), 0.0)
            if adjust:
                s2.loc[d] = med
            rows.append(dict(location=loc, date=d, lag=lag, reported=rep, median=med, mu=mu, sd=sd, shift=shift,
                             growth=growth, n_revisions=m.n if m is not None else 0,
                             prior_median=float(np.exp(p[0]) - offset) if p else np.nan))
        revised[loc] = s2
    return revised, pd.DataFrame(rows)
