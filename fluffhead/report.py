"""Pre-submission review report: one self-contained HTML page per reference date.

Written to forecasts/reports/<ref>-report.html (works offline, no external
scripts), with the data behind it in <ref>-data.json. It shows:

    checks          validation errors, missing or poorly fitted regions, large
                    nowcast revisions, big jumps from last week's forecast, low coverage
    locations       reported data, nowcast, forecast fan to the end of the season,
                    last week's forecast, rate-change and peak distributions
    small multiples every location at a glance
    regions         each simulation group's fit to its share of the data, parameter
                    estimates, effective sample size, network mode
    table           headline numbers for every location
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
TEMPLATE = Path(__file__).with_name("report_template.html")
LEVELS = [0.025, 0.25, 0.5, 0.75, 0.975]
PARAMS = ["r_now", "immune_scale", "travel_share", "humidity_floor", "beta_scale"]
MIN_ESS = 5.0
MIN_PHI = 0.5     # below this the likelihood was tempered hard: the model can't follow the data closely


def _iso(ts) -> str:
    return str(pd.Timestamp(ts).date())


def _r(v, nd=1):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def _wq(x: np.ndarray, w: np.ndarray, qs) -> list:
    """Weighted quantiles."""
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(np.interp(q, cw, x[o])) for q in qs]


def _previous(ref: dt.date) -> pd.DataFrame:
    f = C.FORECAST_DIR / "model-output" / C.model_id() / f"{ref - dt.timedelta(weeks=1)}-{C.model_id()}.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, dtype={"location": str, "output_type_id": str})
    return df[(df.target == C.HOSP_TARGET) & (df.output_type_id == "0.5")]


def _regions(region_dir: Path, weeks: pd.DatetimeIndex) -> list[dict]:
    from .submission import region_weekly
    groups = pd.read_csv(C.REGIONS_DIR / "groups.csv").set_index("group")
    out = []
    for g, row in groups.iterrows():
        f = region_dir / f"{g}.npz"
        rec = {"group": g, "name": row["name"], "pop": int(row["pop"]), "states": row.states, "built": f.exists()}
        if f.exists():
            z = np.load(f, allow_pickle=False)
            meta = json.loads((region_dir / f"{g}.json").read_text())
            W = region_weekly({"daily": z["daily"].astype(float), "dates": pd.DatetimeIndex(z["dates"])}, weeks).sum(2)
            obs_w = pd.DatetimeIndex(z["weeks"])
            obs = pd.Series(np.nansum(z["observed"], 1), index=obs_w).reindex(weeks)
            draws = pd.read_csv(region_dir / f"{g}-draws.csv")
            w = draws.weight.to_numpy()
            params = {p: [_r(v, 3) for v in _wq(draws[p].to_numpy(), w, [0.05, 0.5, 0.95])] for p in PARAMS if p in draws}
            for c in [c for c in draws if c.startswith("mult_")]:
                params[c] = [_r(v, 2) for v in _wq(draws[c].to_numpy(), w, [0.05, 0.5, 0.95])]
            q = np.quantile(W, LEVELS, axis=0)
            rec.update(ess=_r(meta["ess"]), phi=_r(meta.get("phi", 1.0), 2), mode=meta["mode"], n=meta["n"], params=params,
                       observed=[_r(v) for v in obs], bands={str(l): [_r(v) for v in q[i]] for i, l in enumerate(LEVELS)})
        out.append(rec)
    return out


def build_data(wk, df: pd.DataFrame, noisy: dict, weeks: pd.DatetimeIndex, problems: list, region_dir: Path) -> dict:
    ref, last_obs = wk.ref_date, wk.last_obs
    locs = wk.locations.set_index("location")
    prev = _previous(ref)
    obs = wk.revised.pivot_table(index="date", columns="location", values="value").reindex(weeks)
    now = wk.nowcasts
    checks = [{"level": "error", "text": p} for p in problems]
    checks += [{"level": "warning", "text": w} for w in wk.warnings]

    regions = _regions(region_dir, weeks)
    missing = [r["name"] for r in regions if not r["built"]]
    if missing:
        checks.append({"level": "warning", "text": f"{len(missing)} of {len(regions)} regions have no results this week: "
                                                    + "; ".join(missing[:8]) + (" …" if len(missing) > 8 else "")})
    for r in regions:
        if r.get("ess") is not None and r["ess"] < MIN_ESS:
            checks.append({"level": "warning", "text": f"{r['name']}: effective sample size {r['ess']:.1f} — the fit "
                                                        "rests on very few draws; consider more draws"})
        if r.get("phi") is not None and r["phi"] < MIN_PHI:
            checks.append({"level": "warning", "text": f"{r['name']}: likelihood tempered to phi {r['phi']:.2f} — no "
                                                        "draw follows the data closely; the forecast is widened to compensate"})
        for k, v in (r.get("params") or {}).items():
            if k.startswith("mult_") and v[1] is not None and not 0.4 <= v[1] <= 2.5:
                checks.append({"level": "info", "text": f"{r['name']}: admission multiplier for state {k[5:]} is "
                                                        f"{v[1]:.2f} (prior median 1)"})
    absent = sorted(set(locs.index) - set(noisy))
    if absent:
        checks.append({"level": "warning", "text": "No forecast for " + ", ".join(locs.loc[a, "location_name"] for a in absent)})
    big = now[(now.lag == 0) & (now.reported >= 20) & ((now["median"] / now.reported - 1).abs() > 0.15)]
    for _, r in big.iterrows():
        checks.append({"level": "info", "text": f"Nowcast revises {locs.loc[r.location, 'location_name']} {_iso(r.date)}: "
                                                f"reported {r.reported:.0f} → {r['median']:.0f}"})

    from .regions import coverage
    counties = pd.read_csv(C.REGIONS_DIR / "counties.csv", dtype={"fips": str, "state": str}, keep_default_na=False)
    built_groups = {r["group"] for r in regions if r["built"]}
    cov = coverage(counties.assign(group=counties.group.where(counties.group.isin(built_groups), "")))
    hub = df[(df.target == C.HOSP_TARGET) & (df.output_type == "quantile")]
    loc_data, table = [], []
    for loc, x in noisy.items():
        name = locs.loc[loc, "location_name"]
        q = np.quantile(x, LEVELS, axis=0)
        future = weeks >= last_obs
        bands = {str(l): [(_r(v) if f else None) for v, f in zip(q[i], future)] for i, l in enumerate(LEVELS)}
        med = hub[(hub.location == loc) & (hub.output_type_id == "0.5")].set_index("horizon").value
        pv = prev[prev.location == loc].set_index("horizon").value if len(prev) else pd.Series(dtype=float)
        for h in [0, 1, 2, 3]:
            if h in pv and h in med and pv[h] >= 20 and abs(med[h] / pv[h] - 1) > 0.5:
                checks.append({"level": "info", "text": f"{name} horizon {h}: median {med[h]:.0f} vs {pv[h]:.0f} last week"})
                break
        rc = df[(df.target == C.RATE_TARGET) & (df.location == loc)]
        rate = {int(h): {c: _r(v, 3) for c, v in zip(g.output_type_id, g.value)} for h, g in rc.groupby("horizon")}
        pw = df[(df.target == C.PEAK_WEEK_TARGET) & (df.location == loc)]
        ps = df[(df.target == C.PEAK_INC_TARGET) & (df.location == loc)].set_index("output_type_id").value
        nrow = now[(now.location == loc)].sort_values("date")
        state_cov = None if loc == "US" else (_r(cov.coverage.get(loc, 0) * 100, 0))
        if state_cov is not None and loc not in C.SCALED_STATES and state_cov < 40:
            checks.append({"level": "info", "text": f"{name}: only {state_cov:.0f}% of the population is simulated; "
                                                    "the rest is scaled from it"})
        loc_data.append({
            "location": loc, "name": name, "coverage": state_cov,
            "scaled_from": C.SCALED_STATES.get(loc),
            "observed": [_r(v) for v in (obs[loc] if loc in obs else pd.Series(index=weeks, dtype=float)).where(weeks <= last_obs)],
            "nowcast": [{"date": _iso(r.date), "reported": _r(r.reported), "median": _r(r.median)} for r in nrow.itertuples()],
            "bands": bands,
            "previous": {int(h): _r(v) for h, v in pv.items()},
            "median": {int(h): _r(v) for h, v in med.items()},
            "rate": rate,
            "peak_week": [{"week": w, "p": _r(p, 4)} for w, p in zip(pw.output_type_id, pw.value)],
            "peak_size": {k: _r(ps.get(k)) for k in ["0.05", "0.25", "0.5", "0.75", "0.95"]},
        })
        mode_week = pw.loc[pw.value.idxmax(), "output_type_id"] if len(pw) else None
        table.append({"location": loc, "name": name, "last": _r(obs[loc].get(last_obs) if loc in obs else None),
                      **{f"h{h}": _r(med.get(h)) for h in [0, 1, 2, 3]}, "peak_week": mode_week,
                      "peak_size": _r(ps.get("0.5")), "coverage": state_cov})
    order = ["US"] + sorted([l["location"] for l in loc_data if l["location"] != "US"],
                            key=lambda l: locs.loc[l, "location_name"])
    loc_data.sort(key=lambda d: order.index(d["location"]))
    table.sort(key=lambda d: order.index(d["location"]))
    built = [r for r in regions if r["built"]]
    return {
        "model": C.model_id(), "ref_date": str(ref), "last_obs": _iso(last_obs),
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "weeks": [_iso(w) for w in weeks],
        "summary": {"regions_built": len(built), "regions_total": len(regions),
                    "simulated_pop": int(sum(r["pop"] for r in built)),
                    "us_coverage": _r(cov.drop(list(C.SCALED_STATES), errors="ignore").simulated.sum()
                                      / cov.drop(list(C.SCALED_STATES), errors="ignore")["pop"].sum() * 100, 1),
                    "n_samples": int(len(next(iter(noisy.values())))) if noisy else 0},
        "checks": checks, "locations": loc_data, "table": table, "regions": regions,
    }


def render(payload: dict, out: Path) -> Path:
    blob = json.dumps(payload, default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", blob), encoding="utf-8")
    return out


def write(wk, df, noisy, weeks, problems, region_dir: Path) -> Path:
    payload = build_data(wk, df, noisy, weeks, problems, region_dir)
    d = C.FORECAST_DIR / "reports"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{wk.ref_date}-data.json").write_text(json.dumps(payload, default=str))
    return render(payload, d / f"{wk.ref_date}-report.html")
