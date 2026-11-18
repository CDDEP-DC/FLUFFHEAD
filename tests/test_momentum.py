import numpy as np
import pandas as pd

from fluffhead import momentum as M


def _weeks(n=10, end="2025-12-13"):
    return pd.date_range(end=end, periods=n, freq="7D")


def test_fit_recovers_known_coefficients(tmp_path):
    rng = np.random.default_rng(0)
    n = 4000
    d = pd.DataFrame({k: rng.normal(size=n) for k in M.INPUTS})
    true = dict(zip(M.INPUTS, [0.1, 0.2, 0.4, -0.7, 0.0, -0.8]))
    d["err"] = 0.05 + sum(v * d[k] for k, v in true.items()) + rng.normal(scale=0.01, size=n)
    d["season"] = 2024
    d.to_csv(tmp_path / "t.csv", index=False)
    c = M.fit(tmp_path / "t.csv", alpha=1e-6)
    assert abs(c["intercept"] - 0.05) < 0.01
    for k, v in true.items():
        assert abs(c["coef"][k] - v) < 0.01


def test_inputs_need_both_series_to_reach_the_last_week():
    w = _weeks()
    adm = pd.DataFrame({"date": list(w) * 2, "location": ["24"] * 10 + ["51"] * 10,
                        "value": [10, 12, 15, 20, 26, 34, 44, 57, 74, 96] * 2})
    ed = pd.DataFrame({"date": list(w) + list(w[:-1]), "location": ["24"] * 10 + ["51"] * 9,
                       "pct": list(np.linspace(1, 3, 10)) + list(np.linspace(1, 3, 9))})
    x = M.inputs(adm, ed, w[-1])
    assert set(x) == {"24"}                    # 51's ED stops a week short
    l = np.log(np.array([44, 57, 74, 96]) + 1.0)
    assert np.isclose(x["24"]["adm"], (l[3] - l[1]) / 2)
    assert np.isclose(x["24"]["adm_acc"], (l[3] - l[1]) / 2 - (l[2] - l[0]) / 2)


def test_shifts_skip_small_states_and_apply_holds_the_last_shift():
    w = _weeks(12, end="2026-01-17")
    last = w[5]
    samples = {"24": np.full((50, 12), 100.0), "50": np.full((50, 12), 5.0)}
    coef = {"intercept": 0.2, "coef": {k: 0.0 for k in M.INPUTS}, "cap": 0.4}
    x = {s: {"adm": 0.1, "ed": 0.0, "adm_acc": 0.0, "ed_acc": 0.0} for s in samples}
    sh = M.shifts(samples, w, last, {"24": 100.0, "50": 5.0}, x, coef)
    assert set(sh) == {"24"} and np.allclose(sh["24"], 0.2)
    out = M.apply(samples, w, last, sh)
    assert np.allclose(out["24"][:, :6], 100)                       # reported weeks untouched
    assert np.allclose(out["24"][:, 6:], np.expm1(np.log1p(100) - 0.2))   # h 1-4 and after
    assert out["50"] is samples["50"]


def test_shift_is_capped():
    w = _weeks(8)
    samples = {"24": np.full((10, 8), 50.0)}
    coef = {"intercept": 5.0, "coef": {k: 0.0 for k in M.INPUTS}, "cap": 0.4}
    sh = M.shifts(samples, w, w[2], {"24": 50.0}, {"24": {"adm": 0, "ed": 0, "adm_acc": 0, "ed_acc": 0}}, coef)
    assert np.allclose(sh["24"], 0.4)


def test_state_ed_maps_state_rows():
    n = pd.DataFrame({"trend_source": ["State", "HSA"], "fips": ["24000", "24510"],
                      "percent_visits_influenza": [1.5, 2.0], "date": pd.to_datetime(["2025-12-13"] * 2)})
    e = M.state_ed(n)
    assert e.location.tolist() == ["24"] and e.pct.tolist() == [1.5]
