"""Maryland products (not for the hub): a forecast for each Maryland hospital, and when Maryland
passes the Maryland Department of Health (MDH) trigger for masking in health care facilities, i.e.
when MDH SHOULD recommend masking if it acted on the forecast rather than waiting for the data.

MDH's trigger, from its clinician letter of 2025-01-14: "Maryland's statewide combined - COVID,
influenza and RSV - weekly respiratory virus-associated hospitalization rate (as calculated by CDC)
exceeds 10 hospitalizations per 100,000 residents", after which it recommends facility-wide source
control (universal masking in patient care areas) until the rate "has been below 10 hospitalizations
per 100,000 residents for two consecutive weeks". The rate is CDC RESP-NET's ("Combined", Maryland;
data.cdc.gov kvib-3txy). It is NOT flu alone. The target here is the settled rate (as it reads once
reports are complete). MDH acts on CDC's first release of a week, which reads low (the week ending
2025-01-04 was 10.2 in MDH's letter and is 17.0 now), so its announcements came 16-26 days after the
week the settled rate first passed 10.

The forecast, per path:
  flu   = FLUFFHEAD's Maryland NHSN admissions per 100,000 (each path's growth from the last report,
          times that report) x kappa, the FluSurv-NET / NHSN ratio of a past season (1.2-1.75 since 2022)
  COVID, RSV = analogs: the current level times the change over the same weeks of a past season
  crossing = the first week whose combined rate exceeds 10
The recommendation rule: recommend once the forecast gives at least P_RECOMMEND that the rate is over
10 in some week ending by LEAD_DAYS after the forecast is made (so masking is in place as it crosses).

Hospitals: each path's statewide admissions are split over hospitals in proportion to their share of
Maryland's confirmed COVID admissions, Oct 2022 - Apr 2024 (HHS facility data, healthdata.gov
anag-cw7u; reporting COVID was required, while its flu field is zero all season at the Johns Hopkins
and University of Maryland hospitals). The split is multinomial, so small hospitals get count noise.
Among the hospitals that did report flu, log flu and log COVID admissions correlate 0.67. No
hospital-level flu data exist after April 2024, so these forecasts cannot be scored.

    python -m fluffhead maryland --reference-date 2026-11-21 [--history]
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging

import numpy as np
import pandas as pd
import requests

from . import config as C
from . import hub_data as H

log = logging.getLogger(__name__)

MD = "24"
THRESHOLD = 10.0                 # combined weekly rate per 100,000
P_RECOMMEND = 0.5                # recommend once P(over 10 by LEAD_DAYS after the forecast) reaches this
LEAD_DAYS = 14
KAPPA_SD = 0.10                  # log scatter of kappa around the season drawn
PSEUDO = 0.3                     # added to analog levels so near-zero weeks don't give huge ratios
COVID_SEASONS = range(2022, 2030)    # after the pandemic waves
RSV_SEASONS = [s for s in range(2016, 2030) if s not in (2020, 2021)]
BACKTEST_VARIANT = "w_l31_n23"   # weekly backtest of the live configuration (Washington-Baltimore)
N_PATHS = 2000
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
UA = {"User-Agent": "FLUFFHEAD influenza forecasting (research)"}

RESPNET_URL = "https://data.cdc.gov/resource/kvib-3txy.csv"
NHSN_URL = "https://data.cdc.gov/resource/mpgq-jmmr.csv"
GAP_WEEKS, GAP_LEVEL = 4, 0.3    # COVID-NET zeros taken as data gaps (fill_covid_gaps)
HHS_FACILITY_URL = "https://healthdata.gov/resource/anag-cw7u.csv"
SHARES_PATH = C.DATA_DIR / "maryland" / "hospital_shares.csv"
CACHE = C.CACHE_DIR / "maryland"
OUT_DIR = C.REPO / "forecasts" / "maryland"

# MDH's masking recommendations (first of each season): date, and what it rested on
ANNOUNCED = {
    2023: ("2023-12-28", "MDH notice to clinicians, Dec 28, 2023 (reported by Eye On Annapolis, Dec 29)"),
    2024: ("2025-01-14", "MDH clinician letter, Jan 14, 2025: week ending Jan 4, rate 10.2"),
    2025: ("2026-01-05", "MDH news release, Jan 5, 2026: dashboard week ending Dec 26-27"),
}


def season_of(d) -> int:
    d = pd.Timestamp(d)
    return d.year if d.month >= 8 else d.year - 1


# ------------------------------------------------------------------------------------------ data
def _download(url: str, params: dict, path, max_age_days: float = 0.5) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = path.exists() and (dt.datetime.now().timestamp() - path.stat().st_mtime) < max_age_days * 86400
    if not fresh:
        try:
            r = requests.get(url, params=params, headers=UA, timeout=120)
            r.raise_for_status()
            H._write_atomic(path, r.text)
        except requests.RequestException:
            if not path.exists():
                raise
            log.warning("could not download %s; using the cached copy", url)
    return pd.read_csv(path)


def respnet() -> pd.DataFrame:
    """RESP-NET weekly rates per 100,000 for Maryland (all ages): columns COVID-NET, FluSurv-NET,
    RSV-NET, Combined; index week-ending date (Saturday). Latest values (recent weeks still fill in)."""
    d = _download(RESPNET_URL, {"$limit": 50000, "$where": "state='Maryland' AND age_category='Overall' AND "
                                "race='All' AND sex='All' AND data_type='Weekly Rate' AND "
                                "date_type='Week Ending Date'"}, CACHE / "respnet.csv")
    p = d.pivot_table(index="date", columns="surveillance_network", values="estimate")
    p.index = pd.to_datetime(p.index)
    return p.sort_index()


def nhsn_covid() -> pd.Series:
    """Maryland COVID admissions by week, NHSN (CDC's weekly preliminary file)."""
    d = _download(NHSN_URL, {"$limit": 5000, "$select": "weekendingdate,totalconfc19newadm",
                             "$where": "jurisdiction='MD' AND weekendingdate>='2022-07-01'"}, CACHE / "nhsn-covid.csv")
    s = d.dropna().set_index("weekendingdate").totalconfc19newadm.astype(float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def fill_covid_gaps(rn: pd.DataFrame, covid: pd.Series) -> pd.DataFrame:
    """`rn` with COVID-NET weeks of exactly 0 that are data gaps replaced by NHSN COVID admissions times
    the recent COVID-NET / NHSN ratio, and Combined raised by the same amount. A gap is a 0 after
    GAP_WEEKS weeks averaging at least GAP_LEVEL while hospitals still report COVID admissions (the
    weeks ending 2026-11-07 and 11-14 read 0.0 with 51 and 54 admissions)."""
    rn = rn.copy()
    c = rn["COVID-NET"].copy()
    for d in c.index[c == 0]:
        prev = c[c.index < d].dropna().iloc[-GAP_WEEKS:]
        adm = covid.reindex(prev.index)
        if len(prev) < GAP_WEEKS or prev.mean() < GAP_LEVEL or adm.isna().any() or not covid.get(d, 0) > 0:
            continue
        c[d] = float(covid[d] * prev.sum() / adm.sum())
        rn.loc[d, "Combined"] += c[d]
        log.warning("COVID-NET %s reads 0.0 with %d NHSN COVID admissions: using %.2f", d.date(), covid[d], c[d])
    rn["COVID-NET"] = c
    return rn


def nhsn() -> pd.Series:
    h = H.load_current("hosp")
    s = h[h.location == MD].set_index("date").value.astype(float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def population() -> float:
    loc = H.load_locations()
    return float(loc.loc[loc.location == MD, "population"].iloc[0])


def kappa_by_season(rn: pd.DataFrame, adm: pd.Series, pop: float) -> pd.Series:
    """FluSurv-NET rate / NHSN admissions per 100,000, summed over Nov-Mar weeks with activity."""
    j = pd.concat([adm.rename("nhsn") / pop * 1e5, rn["FluSurv-NET"].rename("fsn")], axis=1).dropna()
    j = j[((j.index.month >= 11) | (j.index.month <= 3)) & (j.nhsn > 0.5)]
    g = j.groupby(j.index.map(season_of))
    k = g.fsn.sum() / g.nhsn.sum()
    return k[g.size() >= 6]


def hospital_shares(rebuild: bool = False) -> pd.DataFrame:
    """Share of Maryland's confirmed COVID admissions by hospital, Oct 2022 - Apr 2024 (tracked file)."""
    if SHARES_PATH.exists() and not rebuild:
        return pd.read_csv(SHARES_PATH, dtype={"ccn": str, "county_fips": str})
    flu = "previous_day_admission_influenza_confirmed_7_day_sum"
    cov = ["previous_day_admission_adult_covid_confirmed_7_day_sum",
           "previous_day_admission_pediatric_covid_confirmed_7_day_sum"]
    d = _download(HHS_FACILITY_URL, {"$limit": 50000, "$select": ",".join(
        ["hospital_pk", "collection_week", "hospital_name", "city", "fips_code", "hospital_subtype", flu] + cov),
        "$where": "state='MD' AND collection_week >= '2022-10-01'"}, CACHE / "hhs_facility.csv", max_age_days=365)
    d["hospital_pk"] = d.hospital_pk.astype(str)
    for c in [flu] + cov:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d[c] = d[c].where(d[c] != -999999, 2.0)       # suppressed counts are 1-3
    d["covid"] = d[cov].sum(axis=1, min_count=1)
    t = (d.groupby("hospital_pk")
         .agg(name=("hospital_name", "last"), city=("city", "last"), county_fips=("fips_code", "last"),
              subtype=("hospital_subtype", "last"), covid_admissions=("covid", "sum"),
              flu_admissions_reported=(flu, "sum"), weeks=("collection_week", "nunique"))
         .reset_index().rename(columns={"hospital_pk": "ccn"}))
    t = t[t.covid_admissions > 0].copy()
    t["county_fips"] = t.county_fips.astype("Int64").astype(str)
    t["share"] = t.covid_admissions / t.covid_admissions.sum()
    t["name"] = (t.name.str.title().str.replace(r"\bU P M C\b", "UPMC", regex=True)
                 .str.replace(r"\b(Umd|Md|Ctr|Inc)\b", lambda m: {"Umd": "UMD", "Md": "MD", "Ctr": "Ctr",
                                                                   "Inc": "Inc"}[m.group(1)], regex=True))
    t = t.sort_values("share", ascending=False)
    SHARES_PATH.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(SHARES_PATH, index=False, float_format="%.5f")
    return t


# ------------------------------------------------------------------------------------- the paths
def _weeks_back(date, k: int) -> pd.Timestamp:
    return pd.Timestamp(date) - pd.Timedelta(weeks=k)


def _season_shift(season: int, analog: int) -> pd.Timedelta:
    """Whole weeks closest to the calendar-year gap between two seasons."""
    return pd.Timedelta(weeks=int(round(365.25 * (season - analog) / 7)))


def other_paths(rn: pd.DataFrame, cutoff, targets: pd.DatetimeIndex, n: int, rng,
                exclude: int | None = None, anchor_weeks=(1, 2)) -> np.ndarray:
    """COVID + RSV rate (per 100,000) in the target weeks, n paths: the level in the anchor weeks
    (weeks before the cutoff, whose reports are mostly complete) times the change over the same
    weeks in an earlier season."""
    season = season_of(cutoff)
    total = np.zeros((n, len(targets)))
    for net, seasons in (("COVID-NET", COVID_SEASONS), ("RSV-NET", RSV_SEASONS)):
        s = rn[net].dropna()
        anchor = [_weeks_back(cutoff, k) for k in anchor_weeks]
        level = float(s.reindex(anchor).mean())
        if np.isnan(level):
            raise ValueError(f"{net}: no data for {anchor}")
        curves = []
        for a in seasons:
            if a >= season or a == exclude:
                continue
            sh = _season_shift(season, a)
            base = s.reindex([t - sh for t in anchor]).mean()
            fut = s.reindex([t - sh for t in targets]).to_numpy()
            if np.isnan(base) or np.isnan(fut).any():
                continue
            curves.append((level + PSEUDO) * (fut + PSEUDO) / (base + PSEUDO) - PSEUDO)
        if not curves:
            raise ValueError(f"{net}: no analog seasons for {cutoff}")
        curves = np.clip(np.array(curves), 0, None)
        total += curves[rng.integers(len(curves), size=n)]
    return total


def growth_paths(x: np.ndarray, weeks: pd.DatetimeIndex, anchor, targets) -> np.ndarray:
    """Each path's value in the target weeks relative to its value in the anchor week."""
    w = list(pd.DatetimeIndex(weeks))
    a = x[:, w.index(pd.Timestamp(anchor))]
    return x[:, [w.index(pd.Timestamp(t)) for t in targets]] / np.maximum(a, 0.5)[:, None]


def live_flu(ref_date, adm: pd.Series):
    """Maryland NHSN flu admissions, (paths, weeks) from the last reported week (its report, in every
    path) through the season's end, from main FLUFFHEAD's state samples for this reference date."""
    z = np.load(C.REPO / "runs" / str(ref_date) / "state-samples.npz", allow_pickle=False)
    weeks = pd.DatetimeIndex(z["weeks"])
    cutoff = adm.index[adm.index <= pd.Timestamp(ref_date)].max()
    targets = weeks[weeks >= cutoff]
    g = growth_paths(z[MD].astype(float), weeks, cutoff, targets)
    return cutoff, targets, adm[cutoff] * g


def backtest_flu(path, adm: pd.Series):
    """The same, from a weekly backtest case of Washington-Baltimore (4 weeks ahead): the Maryland
    piece's growth applied to Maryland's reported admissions."""
    z = np.load(path, allow_pickle=False)
    weeks = pd.DatetimeIndex(z["weeks"])
    k = list(z["pieces"]).index(MD)
    cutoff = weeks[int(z["n_hist"]) - 1]
    targets = weeks[int(z["n_hist"]) - 1:]
    g = growth_paths(z["w"][:, :, k].astype(float), weeks, cutoff, targets)
    return cutoff, targets, adm[cutoff] * g


def trigger_paths(flu_adm: np.ndarray, cutoff, targets, rn, kappas: pd.Series, pop, n, rng,
                  exclude: int | None = None) -> dict:
    """Combined rate (settled) in the target weeks, and the first week over the threshold, per path."""
    flu_adm = flu_adm[rng.integers(flu_adm.shape[0], size=n)]
    ks = kappas[kappas.index != exclude].to_numpy()
    kappa = ks[rng.integers(len(ks), size=n)] * np.exp(KAPPA_SD * rng.standard_normal(n))
    flu = flu_adm / pop * 1e5 * kappa[:, None]
    other = other_paths(rn, cutoff, targets, n, rng, exclude=exclude)
    combined = flu + other
    over = combined > THRESHOLD
    hit = over.any(axis=1)
    tgt = np.asarray(targets, dtype="datetime64[D]")
    week = np.where(hit, tgt[over.argmax(axis=1)], np.datetime64("NaT"))
    return dict(combined=combined, flu=flu, other=other, hit=hit, week=pd.DatetimeIndex(week))


def p_by(tp: dict, date) -> float:
    """Share of paths over the threshold in some week ending on or before `date`."""
    return float((tp["week"] <= pd.Timestamp(date)).mean())


def recommend(tp: dict, made) -> tuple[bool, float]:
    p = p_by(tp, pd.Timestamp(made) + pd.Timedelta(days=LEAD_DAYS))
    return p >= P_RECOMMEND, p


def summarize(tp: dict, horizon_end) -> dict:
    """Crossing-week quantiles over all paths, a path that stays under 10 through the horizon counting
    as later than any week (so a quantile past the share that cross is None: 'after the horizon')."""
    out = dict(p_within=float(tp["hit"].mean()), horizon_end=str(pd.Timestamp(horizon_end).date()))
    srt = pd.Series(tp["week"]).sort_values(na_position="last").reset_index(drop=True)

    def pick(p):
        v = srt.iloc[int(round(p * (len(srt) - 1)))]
        return None if pd.isna(v) else str(v.date())
    out.update({f"q{int(p * 100)}": pick(p) for p in QUANTILES})
    out["median"] = out["q50"]
    return out


def actual_crossing(rn: pd.DataFrame, season: int):
    """Week ending in which the settled combined rate first exceeded the threshold that season."""
    comb = rn["Combined"].dropna()
    s = comb[(comb.index >= f"{season}-09-01") & (comb.index < f"{season + 1}-06-01")]
    w = s[s > THRESHOLD].index.min()
    return None if pd.isna(w) else w


# -------------------------------------------------------------------------------------- products
def hospital_forecast(flu_adm: np.ndarray, targets, shares: pd.DataFrame, rng, weeks_out: int = 10):
    """Per-hospital weekly admissions (quantiles), peak and total, from statewide paths."""
    n = np.round(np.clip(flu_adm, 0, None)).astype(np.int64)
    split = rng.multinomial(n, shares.share.to_numpy() / shares.share.sum())     # (paths, weeks, hospitals)
    rows, summ = [], []
    for j, h in enumerate(shares.itertuples()):
        x = split[:, :, j]
        q = np.quantile(x[:, :weeks_out], QUANTILES, axis=0)
        for i, t in enumerate(targets[:weeks_out]):
            rows.append(dict(ccn=h.ccn, hospital=h.name, city=h.city, target_end_date=t.date(),
                             **{f"q{p:g}": float(q[k, i]) for k, p in enumerate(QUANTILES)}))
        peak_wk = pd.Series(targets[x.argmax(axis=1)])
        summ.append(dict(ccn=h.ccn, hospital=h.name, city=h.city, share=round(float(h.share), 4),
                         peak_week_median=peak_wk.quantile(0.5, interpolation="nearest").date(),
                         peak_median=float(np.median(x.max(axis=1))),
                         peak_q10=float(np.quantile(x.max(axis=1), 0.1)),
                         peak_q90=float(np.quantile(x.max(axis=1), 0.9)),
                         season_rest_median=float(np.median(x.sum(axis=1)))))
    return pd.DataFrame(rows), pd.DataFrame(summ)


def history(rn, adm, kappas, pop, n=N_PATHS, seed=1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each backtest forecast of the past three seasons made before MDH announced (4 weeks ahead), and
    per season: when the rule would first have recommended masking vs when the settled rate crossed 10
    and when MDH announced. As of the forecast date: NHSN's last report is the hub's version of that
    week (not nowcast), and kappa and the analogs come from earlier seasons only. Not as of the date: the
    recent COVID and RSV rates (no archive) and the backtest's own fit, which are settled values."""
    rows = []
    vintages = H.load_vintages("hosp", f"{min(ANNOUNCED)}-10-01")
    for f in sorted((C.RUNS_DIR / "backtest" / BACKTEST_VARIANT).glob("*-csa548.npz")):
        season = int(f.name[:4])
        if season not in ANNOUNCED:
            continue
        rng = np.random.default_rng(seed)
        cutoff, targets, fl = backtest_flu(f, adm)
        announced = pd.Timestamp(ANNOUNCED[season][0])
        made = cutoff + pd.Timedelta(days=4)                    # Wednesday, with that week's NHSN data
        if made >= announced:
            continue
        v = H.vintage_for(vintages, (cutoff + pd.Timedelta(days=7)).date())
        v = v[(v.location == MD) & (v.date == cutoff)].value
        if v.empty:
            raise ValueError(f"no NHSN report as of {made.date()} for {cutoff.date()}")
        completeness = float(v.iloc[0]) / adm[cutoff]
        tp = trigger_paths(fl * completeness, cutoff, targets, rn, kappas[kappas.index < season], pop, n, rng)
        rec, p = recommend(tp, made)
        s = summarize(tp, targets[-1])
        rows.append(dict(season=f"{season}-{(season + 1) % 100:02d}", data_through=cutoff.date(),
                         forecast_made=made.date(), nhsn_completeness=round(completeness, 2),
                         p_over_10_within_2wk=round(p, 2),
                         p_over_10_within_4wk=round(s["p_within"], 2), recommend=rec,
                         crossing_week_median=s["median"], crossing_week_q10=s["q10"],
                         crossing_week_q90=s["q90"]))
    per = pd.DataFrame(rows)
    seasons = []
    for season, (date, src) in ANNOUNCED.items():
        lab = f"{season}-{(season + 1) % 100:02d}"
        h = per[per.season == lab] if len(per) else per
        first = pd.Timestamp(h[h.recommend].forecast_made.min()) if len(h) and h.recommend.any() else None
        x = actual_crossing(rn, season)
        ann = pd.Timestamp(date)
        seasons.append(dict(season=lab, model_recommends=first.date() if first is not None else None,
                            settled_rate_first_over_10_week_ending=x.date() if x is not None else None,
                            mdh_announced=ann.date(), mdh_source=src,
                            model_days_before_mdh=(ann - first).days if first is not None else None,
                            model_days_after_crossing_week=(first - x).days
                            if first is not None and x is not None else None,
                            mdh_days_after_crossing_week=(ann - x).days if x is not None else None,
                            forecasts=len(h)))
    return per, pd.DataFrame(seasons)


def run(ref_date, with_history: bool = False, seed: int = 1) -> dict:
    ref_date = pd.Timestamp(ref_date).date()
    rng = np.random.default_rng(seed)
    rn, adm, pop = fill_covid_gaps(respnet(), nhsn_covid()), nhsn(), population()
    kappas = kappa_by_season(rn, adm, pop)
    shares = hospital_shares()
    cutoff, targets, fl = live_flu(ref_date, adm)
    out = OUT_DIR / str(ref_date)
    out.mkdir(parents=True, exist_ok=True)

    season = season_of(cutoff)
    released = rn["Combined"][(rn.index >= f"{season}-09-01") & (rn.index <= cutoff)]
    made = pd.Timestamp(dt.date.today())
    tp = trigger_paths(fl, cutoff, targets, rn, kappas, pop, N_PATHS, rng)
    rec, p_soon = recommend(tp, made)
    weekly = pd.DataFrame({"week_ending": targets.date,
                           "combined_rate_median": np.median(tp["combined"], axis=0).round(2),
                           "combined_rate_q10": np.quantile(tp["combined"], 0.1, axis=0).round(2),
                           "combined_rate_q90": np.quantile(tp["combined"], 0.9, axis=0).round(2),
                           "flu_rate_median": np.median(tp["flu"], axis=0).round(2),
                           "covid_rsv_rate_median": np.median(tp["other"], axis=0).round(2),
                           "p_over_10": (tp["combined"] > THRESHOLD).mean(axis=0).round(3)})
    cum = pd.DataFrame({"by_week_ending": targets.date,
                        "p_crossed": [round(p_by(tp, t), 3) for t in targets]})
    summary = dict(reference_date=str(ref_date), data_through=str(cutoff.date()), made=str(made.date()),
                   last_combined_rate_reported=float(released.iloc[-1]) if len(released) else None,
                   kappa_by_season={int(k): round(float(v), 2) for k, v in kappas.items()},
                   crossing_week=summarize(tp, targets[-1]),
                   recommend_now=rec, p_over_10_within_2wk=round(p_soon, 3),
                   rule=f"recommend when P(settled combined rate > {THRESHOLD:g} in a week ending within "
                        f"{LEAD_DAYS} days of the forecast) >= {P_RECOMMEND:g}",
                   note="Combined COVID+flu+RSV RESP-NET rate, settled values (see maryland.py)")
    weekly.to_csv(out / "trigger-weekly.csv", index=False)
    cum.to_csv(out / "trigger-cumulative.csv", index=False)
    hosp, hsum = hospital_forecast(fl[rng.integers(fl.shape[0], size=N_PATHS), 1:], targets[1:], shares, rng)
    hosp.to_csv(out / "hospitals-weekly.csv", index=False)
    hsum.to_csv(out / "hospitals-summary.csv", index=False)
    if with_history:
        hist, seasons = history(rn, adm, kappas, pop, seed=seed)
        hist.to_csv(out / "trigger-history.csv", index=False)
        seasons.to_csv(out / "trigger-history-seasons.csv", index=False)
        summary["history"] = seasons.drop(columns="mdh_source").to_dict("records")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Maryland products: %s", out)
    return summary
