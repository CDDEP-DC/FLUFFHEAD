"""FLUFFHEADMetro: ED conversion and hub checks (no downloads)."""
import numpy as np
import pandas as pd

from fluffhead import local as L


def test_growth_whole_and_fractional_lead():
    weeks = pd.date_range("2026-10-03", periods=6, freq="W-SAT")
    w = np.array([[10, 20, 40, 80, 160, 320.0]])        # doubling weekly
    g = L.growth(w, weeks, weeks[2], [weeks[3], weeks[4]], lead=0.0)
    assert np.allclose(g, [[2, 4]])
    g = L.growth(w, weeks, weeks[2], [weeks[3]], lead=0.5)   # log-linear interpolation: still x2
    assert np.allclose(g, [[2]])
    g = L.growth(w, weeks, weeks[4], [weeks[5]], lead=1.0)   # past the end: last week's growth
    assert np.allclose(g, [[2]])


def test_sum_ranked_pairs_by_rank():
    a = np.array([[1.0, 1], [3, 3]] * 50)                   # 100 trajectories
    b = np.array([[10.0, 10], [30, 30]] * 50)
    s = L._sum_ranked([a, b], np.array([True, True]))
    assert s.shape == (L.N_OUT, 2)
    assert set(np.unique(s[:, 0])) == {11.0, 33.0}           # low with low, high with high


def test_check_flags_bad_rows(monkeypatch):
    monkeypatch.setattr(L, "hub_tasks", lambda: {L.ED_TARGET: {"locations": ["denver"], "reference_dates": ["2026-11-28"],
                                                               "target_end_dates": ["2026-11-28", "2026-12-05"]}})
    rows = [("2026-11-28", "denver", L.ED_TARGET, "2026-11-28", "sample", f"L000s{k:03d}", 1.0) for k in range(L.N_OUT)]
    df = pd.DataFrame(rows, columns=["reference_date", "location", "target", "target_end_date", "output_type",
                                     "output_type_id", "value"])
    assert L.check(df) == []
    bad = pd.concat([df, df.assign(location="boulder")])
    assert any("boulder" in p for p in L.check(bad))
    assert any("more than one location" in p for p in L.check(bad))
