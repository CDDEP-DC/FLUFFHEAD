"""Maryland products: trigger timing logic and the hospital split (no downloads)."""
import numpy as np
import pandas as pd

from fluffhead import maryland as M


def _respnet(weeks):
    return pd.DataFrame({"COVID-NET": 1.0, "RSV-NET": 1.0, "FluSurv-NET": 1.0, "Combined": 3.0},
                        index=pd.DatetimeIndex(weeks))


def test_crossing_week_and_recommendation(monkeypatch):
    monkeypatch.setattr(M, "KAPPA_SD", 0.0)
    weeks = pd.date_range("2016-09-03", "2026-06-27", freq="W-SAT")
    rn = _respnet(weeks)
    targets = pd.date_range("2025-12-06", periods=4, freq="W-SAT")
    pop = 1e5                                       # admissions = rate per 100,000
    kappas = pd.Series({2022: 1.0})
    flu = np.array([[4.0, 6.0, 9.0, 12.0]])         # combined (flu + 2): 6, 8, 11, 14 -> crosses week 3
    tp = M.trigger_paths(flu, targets[0], targets, rn, kappas, pop, 10, np.random.default_rng(0))
    assert tp["hit"].all() and (tp["week"] == targets[2]).all()
    made = targets[0] - pd.Timedelta(days=3)        # 14 days on is before the week-3 end
    assert M.recommend(tp, made) == (False, 0.0)
    assert M.recommend(tp, made + pd.Timedelta(days=7)) == (True, 1.0)
    low = np.full((1, 4), 7.0)                      # 9 every week: never
    tp = M.trigger_paths(low, targets[0], targets, rn, kappas, pop, 10, np.random.default_rng(0))
    s = M.summarize(tp, targets[-1])
    assert s["p_within"] == 0 and s["median"] is None


def test_analog_keeps_flat_level():
    weeks = pd.date_range("2016-09-03", "2026-06-27", freq="W-SAT")
    targets = pd.date_range("2025-12-06", periods=3, freq="W-SAT")
    x = M.other_paths(_respnet(weeks), targets[0], targets, 5, np.random.default_rng(0))
    assert np.allclose(x, 2.0)                      # COVID 1 + RSV 1, flat in every analog season


def test_hospital_split_adds_up():
    shares = pd.DataFrame({"ccn": ["a", "b"], "name": ["A", "B"], "city": ["X", "Y"], "share": [0.75, 0.25]})
    targets = pd.date_range("2025-12-06", periods=3, freq="W-SAT")
    flu = np.full((400, 3), 100.0)
    weekly, summ = M.hospital_forecast(flu, targets, shares, np.random.default_rng(0))
    med = weekly.groupby("ccn")["q0.5"].mean()
    assert abs(med["a"] - 75) < 3 and abs(med["b"] - 25) < 3
    assert set(summ.ccn) == {"a", "b"}


def test_fill_covid_gaps():
    """A COVID-NET zero is filled only after a run of real rates with hospitals still reporting COVID."""
    idx = pd.date_range("2026-09-05", periods=8, freq="7D")
    rn = pd.DataFrame({"COVID-NET": [.1, 0.0, .8, .8, .8, .8, 0.0, 0.0], "Combined": 2.0}, index=idx)
    covid = pd.Series(40.0, index=idx)
    out = M.fill_covid_gaps(rn, covid)
    assert out["COVID-NET"].iloc[1] == 0.0                       # no run of real rates before it
    assert np.allclose(out["COVID-NET"].iloc[-2:], 0.8)
    assert np.allclose(out["Combined"].iloc[-2:], 2.8) and (out["Combined"].iloc[:-2] == 2.0).all()
    assert (M.fill_covid_gaps(rn, covid * 0)["COVID-NET"] == rn["COVID-NET"]).all()
