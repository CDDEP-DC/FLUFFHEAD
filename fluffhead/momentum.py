"""Momentum correction: an alternate forecast shifted by how far ED visits lead admissions.

FLUFFHEAD's 1-4 week errors follow the recent growth of each state's flu ED visits relative
to its admissions: when ED has flattened while admissions still climb, the forecast runs too
high (admissions lag ED by about a week), and the reverse when ED speeds up. A ridge regression
predicts each state's forecast error on the log(x + 1) scale from

    h            weeks after the last reported week (1-4; hub horizons 0-3)
    own_growth   the forecast's median at h minus the last reported week, log(x + 1)
    adm, ed      2-week average weekly log growth of admissions, log(x + 1), and of the
                 flu % of ED visits, log(% + 0.05), to the last reported week
    adm_acc, ed_acc   their change from the week before

and the state's samples are shifted by minus that prediction, capped at CAP. Weeks after the
4th keep the 4th week's shift, so the peak targets don't jump.

Fitted on the live-settings backtest on 2023 networks (data/momentum/training.csv: 501 cases,
2023-24 to 2025-26, revised data). Held-out season WIS 0.88 of the uncorrected forecast with
revised inputs. With inputs as known at each forecast it holds only when ED for the last
reported week is in hand: 0.88 in 2025-26 (NSSP released Wednesday mornings), 1.15 in 2024-25
(released Fridays, a week late). So a state is corrected only when its ED has that week, and
only with at least MIN_ADMISSIONS admissions last week.

The alternate goes to forecasts/alternates/momentum/ with its own review report; the hub
submission is untouched.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)
DIR = C.REPO / "data" / "momentum"
TRAIN = DIR / "training.csv"
COEF = DIR / "coef.json"
OUT_DIR = C.FORECAST_DIR / "alternates" / "momentum"
ENABLED = True        # pipeline.run also writes the alternate
INPUTS = ["h", "own_growth", "adm", "ed", "adm_acc", "ed_acc"]
ALPHA = 10.0          # ridge penalty on standardized inputs
CAP = 0.4             # largest shift, log(x + 1) scale (x1.5)
ED_OFFSET = 0.05      # log(% + 0.05)
MAX_H = 4
MIN_ADMISSIONS = 20   # states below this last week are left alone (backtest: no gain there, 0.98 log / 1.00 counts;
                      # 20-100: 0.83 / 0.91; 100+: 0.79 / 0.76)


def fit(train: Path = TRAIN, alpha: float = ALPHA) -> dict:
    """Ridge on standardized inputs (population sd), intercept unpenalized; coefficients on the raw scale."""
    d = pd.read_csv(train)
    X, y = d[INPUTS].to_numpy(float), d.err.to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    zc, yc = Z - Z.mean(0), y - y.mean()
    b = np.linalg.solve(zc.T @ zc + alpha * np.eye(len(INPUTS)), zc.T @ yc)
    beta = b / sd
    icpt = y.mean() - (beta * mu).sum()
    return {"intercept": float(icpt), "coef": dict(zip(INPUTS, map(float, beta))), "alpha": alpha, "cap": CAP,
            "n": int(len(d)), "seasons": sorted(map(int, d.season.unique())), "fitted": str(dt.date.today())}


def coefficients() -> dict:
    if not COEF.exists():
        c = fit()
        COEF.parent.mkdir(parents=True, exist_ok=True)
        COEF.write_text(json.dumps(c, indent=1))
    return json.loads(COEF.read_text())


def _growth(s: pd.Series, last: pd.Timestamp, offset: float) -> tuple[float, float] | None:
    """(2-week average weekly log growth, its change from the week before) to `last`; None unless the
    four weeks to `last` are all there."""
    v = s.reindex(pd.date_range(end=last, periods=4, freq="7D"))
    if v.isna().any():
        return None
    l = np.log(v.clip(lower=0).to_numpy(float) + offset)
    g0, g1 = (l[3] - l[1]) / 2, (l[2] - l[0]) / 2
    return float(g0), float(g0 - g1)


def state_ed(nssp: pd.DataFrame) -> pd.DataFrame:
    """State rows of an ed.fetch frame: date, location (state FIPS), pct."""
    s = nssp[nssp.trend_source == "State"]
    return pd.DataFrame({"date": s.date, "location": s.fips.astype(str).str.zfill(5).str[:2],
                         "pct": pd.to_numeric(s.percent_visits_influenza, errors="coerce")}).dropna()


def inputs(reported: pd.DataFrame, ed: pd.DataFrame, last_obs: pd.Timestamp) -> dict[str, dict]:
    """{state: {adm, adm_acc, ed, ed_acc}} from reported admissions (as posted, not nowcast) and state
    ED; a state is left out unless both series reach last_obs."""
    out = {}
    adm = reported.pivot_table(index="date", columns="location", values="value")
    e = ed.pivot_table(index="date", columns="location", values="pct")
    for loc in adm.columns:
        if loc == "US" or loc not in e.columns:
            continue
        a, b = _growth(adm[loc], last_obs, C.LOG_OFFSET), _growth(e[loc], last_obs, ED_OFFSET)
        if a and b:
            out[loc] = {"adm": a[0], "adm_acc": a[1], "ed": b[0], "ed_acc": b[1]}
    return out


def shifts(samples: dict[str, np.ndarray], weeks: pd.DatetimeIndex, last_obs: pd.Timestamp,
           last_value: dict[str, float], x: dict[str, dict], coef: dict) -> dict[str, list[float]]:
    """{state: [shift at h = 1..4]} on the log(x + 1) scale (positive: the forecast is lowered)."""
    i0 = int(np.searchsorted(weeks, last_obs))
    out = {}
    for loc, f in x.items():
        if loc not in samples or not last_value.get(loc, 0) >= MIN_ADMISSIONS:
            continue
        s = []
        for h in range(1, MAX_H + 1):
            q2 = np.log1p(np.median(samples[loc][:, i0 + h]))
            z = {**f, "h": h, "own_growth": q2 - np.log1p(last_value[loc])}
            v = coef["intercept"] + sum(coef["coef"][k] * z[k] for k in INPUTS)
            s.append(float(np.clip(v, -coef["cap"], coef["cap"])))
        out[loc] = s
    return out


def apply(samples: dict[str, np.ndarray], weeks: pd.DatetimeIndex, last_obs: pd.Timestamp,
          sh: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Copies of the state samples [samples, weeks] with the shifts applied after last_obs."""
    i0 = int(np.searchsorted(weeks, last_obs))
    out = {}
    for loc, x in samples.items():
        if loc not in sh:
            out[loc] = x
            continue
        f = np.zeros(len(weeks))
        for h, v in enumerate(sh[loc], start=1):
            if i0 + h < len(weeks):
                f[i0 + h] = v
        f[i0 + MAX_H + 1:] = sh[loc][-1]
        out[loc] = np.expm1(np.log1p(np.maximum(x, 0)) - f[None, :])
    return out
