"""2023 (GeoPops) network path: Connecticut translation and vintage fields."""
import json

import pandas as pd

from fluffhead import config as C
from fluffhead import geopops as G


def test_ct_counties_become_tracts(tmp_path, monkeypatch):
    regions = tmp_path / "regions"
    (regions / "g1").mkdir(parents=True)
    (regions / "g1" / "geo.json").write_text(json.dumps(
        {"geos": ["09003", "25013"], "commute_states": ["09", "25", "72"], "use_pums": ["09", "25"]}))
    monkeypatch.setattr(C, "REGIONS_DIR", regions)
    xw = pd.DataFrame({"county": ["09003", "09003", "09009"],
                       "geoid": ["09110500100", "09170500200", "09170140100"]})
    monkeypatch.setattr(G, "ct_tracts", lambda: xw)
    sp = G.spec("g1")
    assert sp["geos"] == ["25013", "09110500100", "09170500200"]     # only Hartford County's tracts
    assert sp["commute_states"] == ["09", "25"]                      # no LODES for Puerto Rico
    assert sp["use_pums"] == ["09", "25"]


def test_network_dir_is_separate():
    assert G.network_dir("csa548").name == f"network-{G.YEAR}"
    import os
    if "FLUFFHEAD_NETWORK_VINTAGE" not in os.environ:   # default: the model reads what the 2023 build writes
        assert C.NETWORK_SUBDIR == G.network_dir("csa548").name


def test_lodes_ct_codes_become_planning_regions(monkeypatch):
    import pandas as pd
    from fluffhead import geopops as G
    x = pd.DataFrame({"county": ["09003", "09001"], "geoid": ["09110400101", "09190010101"]})
    monkeypatch.setattr(G, "ct_tracts", lambda: x)
    s = pd.Series(["090034001011", "090010101011003", "360610001001", "090099999991"])
    assert list(G.recode_ct(s)) == ["091104001011", "091900101011003", "360610001001", "090099999991"]


def test_pums_state_code_is_padded(tmp_path, monkeypatch):
    from fluffhead import geopops as G
    monkeypatch.setattr(G, "STORE", tmp_path)
    src = tmp_path / "psam_h09.csv"
    src.write_text("RT,SERIALNO,PUMA,ST,NP\nH,2019HU1,20601,9,2\nH,2019HU2,903,9,\n")
    out = G.pums_st(src).read_text()
    assert out == "RT,SERIALNO,PUMA,ST,NP\nH,2019HU1,20601,09,2\nH,2019HU2,903,09,\n"
