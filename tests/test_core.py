"""Core checks on a small synthetic network (no downloaded data needed)."""
import datetime as dt

import numpy as np
import pytest

from fluffhead import calendar as K
from fluffhead import engine as E
from fluffhead import network as N
from fluffhead.disease import DiseaseParams
from fluffhead.submission import rate_change_pmf


def synthetic_network(n_hh=3000, seed=0) -> N.Network:
    """Households of 1-5, workplaces of ~20 adults, schools of ~300 children, 5 tracts."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 6, n_hh)
    n = int(sizes.sum())
    hh = np.repeat(np.arange(n_hh), sizes)
    age = rng.integers(0, 90, n)
    ag = np.searchsorted(N.AGE_BREAKS, age, side="right").astype(np.int8)
    adults = np.flatnonzero(age >= 18)
    kids = np.flatnonzero((age >= 5) & (age < 18))
    wp = rng.integers(0, max(1, len(adults) // 20), len(adults))
    sch = rng.integers(0, max(1, len(kids) // 300), len(kids))
    gp, gm, gb, pp, ps, go = [], [], [], [], [], []
    for grp, idx, blk in [(hh, np.arange(n), None), (wp, adults, rng.integers(0, 3, len(adults))),
                          (sch, kids, age[kids] - 5), (np.zeros(0, int), np.zeros(0, int), None)]:
        ng = int(grp.max()) + 1 if len(grp) else 0
        ptr, mem, order = N._csr(grp, idx, ng)
        b = (np.asarray(blk)[order] if blk is not None else np.zeros(len(mem))).astype(np.int16)
        gp.append(ptr); gm.append(mem); gb.append(b)
        p2, s2 = N._person_index(mem, n)
        pp.append(p2); ps.append(s2)
        go.append(np.repeat(np.arange(ng, dtype=np.int32), np.diff(ptr)))
    home = (hh % 5).astype(np.int32)
    work = np.full(n, -1, np.int32)
    work[adults] = (wp % 5).astype(np.int32)
    who = np.concatenate([np.arange(n), adults])
    where = np.concatenate([home, work[adults]])
    tptr, tmem, _ = N._csr(where, who, 5)
    return N.Network(n=n, age_group=ag, county=np.zeros(n, np.int16), counties=np.array(["46099"]),
                     resident=np.ones(n, bool), out_worker=np.zeros(n, bool), grp_ptr=gp, grp_mem=gm, grp_blk=gb,
                     per_ptr=pp, per_slot=ps, grp_of=go, home_tract=home, work_tract=work, tract_ptr=tptr,
                     tract_mem=tmem, params={"workplace_K": 8, "school_K": 12, "gq_K": 12, "netw_B": 0.25,
                                             "income_associativity_coefficient": 0.9,
                                             "school_associativity_coefficient": 0.9})


@pytest.fixture(scope="module")
def net():
    return synthetic_network()


def run(net, mode, seed=5, days=150, beta=2.5):
    p = DiseaseParams()
    p.beta_scale = beta
    p.prior_immune[:] = 0
    cal = np.ones((E.N_ROWS, days))
    imp = np.zeros(days)
    imp[:5] = 10 / net.n / 5
    imm = E.Immunity(np.zeros(net.n, bool), np.full(net.n, -1, np.int16))
    sim = E.Simulation(net, p, cal, dt.date(2026, 8, 1), imm, imp, mode=mode, net_seed=9, seed=seed,
                       lazy_budget_bytes=1 << 16)
    sim.run_to(days)
    return sim


def test_group_edges_deterministic_and_degree():
    blk = np.random.default_rng(1).integers(0, 4, 1500).astype(np.int16)
    a = N._group_edges(0, blk, 8, 0.9, 0.0, 77)
    b = N._group_edges(0, blk, 8, 0.9, 0.0, 77)
    assert list(a[0]) == list(b[0]) and list(a[1]) == list(b[1])
    ptr, nbr = N._local_csr(1500, *a)
    assert 7.0 < np.diff(ptr).mean() < 9.0
    # no self loops, no duplicates, symmetric
    for i in range(0, 1500, 97):
        row = nbr[ptr[i]:ptr[i + 1]]
        assert i not in row and len(set(row)) == len(row)
        for j in row:
            assert i in nbr[ptr[j]:ptr[j + 1]]


def test_small_groups_complete():
    us, vs = N._group_edges(0, np.zeros(6, np.int16), 8, 0.9, 0.0, 1)
    assert len(us) == 15


def test_full_and_lazy_identical(net):
    a, b = run(net, "full"), run(net, "lazy")
    assert a.rn > 50
    assert np.array_equal(a.rec[0][:a.rn], b.rec[0][:b.rn])
    assert np.allclose(a.rec[1][:a.rn], b.rec[1][:b.rn])


def test_infections_unique_and_causal(net):
    sim = run(net, "full")
    inf = sim.infections()
    assert inf.person.is_unique
    t = dict(zip(inf.person, inf.t))
    src = inf[inf.infector >= 0]
    # infector was infected before infecting, and was then past their latent period
    assert all(t[s] < ti for s, ti in zip(src.infector, src.t))


def test_save_load_continue(net, tmp_path):
    sim = run(net, "full", days=40)
    sim.save(tmp_path / "s.npz")
    sim2 = run(net, "full", days=1)
    sim2.load(tmp_path / "s.npz")
    assert sim2.day == 40 and sim2.rn == sim.rn
    sim2.run_to(80)
    assert sim2.rn >= sim.rn


def test_immunity_blocks_infection(net):
    p = DiseaseParams()
    p.beta_scale = 3
    cal = np.ones((E.N_ROWS, 60))
    imp = np.full(60, 1e-3)
    imm = E.Immunity(np.ones(net.n, bool), np.full(net.n, -1, np.int16))
    sim = E.Simulation(net, p, cal, dt.date(2026, 8, 1), imm, imp, seed=1)
    sim.run_to(60)
    assert sim.rn == 0


def test_school_calendar():
    c = K.build(dt.date(2026, 8, 1), 200, "24", 0.0)
    day = lambda d: (d - dt.date(2026, 8, 1)).days
    assert c[N.SCH, day(dt.date(2026, 10, 14))] == 1          # Wednesday in term
    assert c[N.SCH, day(dt.date(2026, 10, 17))] == 0          # Saturday
    assert c[N.SCH, day(dt.date(2026, 11, 26))] == 0          # Thanksgiving
    assert c[N.SCH, day(dt.date(2026, 12, 23))] == 0          # winter break
    assert c[N.SCH, day(dt.date(2026, 8, 25))] == 0           # before a post-Labor-Day start
    assert (c <= 1).all() and (c >= 0).all()


def test_rate_change_categories():
    p = rate_change_pmf(np.array([-100, -30, 0, 30, 100]), stable=20, large=50)
    assert np.allclose(p, 0.2)
    p = rate_change_pmf(np.array([5, -5]), stable=2, large=3)    # small counts are always stable
    assert p[2] == 1


def test_imports_follow_weights(net):
    p = DiseaseParams()
    p.beta_scale = 0.0
    days = 30
    w = np.zeros(net.n)
    w[:100] = 1.0
    w[100:110] = 5.0
    imm = E.Immunity(np.zeros(net.n, bool), np.full(net.n, -1, np.int16))
    sim = E.Simulation(net, p, np.ones((E.N_ROWS, days)), dt.date(2026, 8, 1), imm, np.full(days, 0.01), w, seed=3)
    sim.run_to(days)
    inf = sim.infections()
    assert len(inf) > 20 and (inf.person < 110).all() and (inf.layer == "import").all()


def test_community_multiplier(net):
    def community(mult):
        p = DiseaseParams()
        p.beta_scale = 2.0
        p.prior_immune[:] = 0
        days = 60
        imp = np.zeros(days)
        imp[:5] = 10 / net.n / 5
        imm = E.Immunity(np.zeros(net.n, bool), np.full(net.n, -1, np.int16))
        sim = E.Simulation(net, p, np.ones((E.N_ROWS, days)), dt.date(2026, 8, 1), imm, imp, seed=5,
                           community_mult=np.full(5, mult))
        sim.run_to(days)
        inf = sim.infections()
        return inf.layer.str.startswith("community").sum()
    # tract encounters stop; only the region-wide share (multiplier 1) remains
    assert community(0.0) < 0.2 * community(1.0)


def test_growth_term_penalises_wrong_slope():
    from fluffhead import calibrate as CB
    y = np.array([[100.], [125.], [156.], [195.], [244.]])          # x1.25 / week
    same = y * 1.1
    steep = np.array([[60.], [90.], [135.], [200.], [300.]])         # x1.5 / week
    assert CB.growth_term(same, y) > -0.01
    assert CB.growth_term(steep, y) < -3
    assert CB.growth_term(steep, y / 50) > CB.growth_term(steep, y)  # small counts: weaker



def two_strain_run(net, mask, cross, rel=(1.0, 1.0), days=120, seed=4):
    p = DiseaseParams()
    p.beta_scale = 2.5
    imp = np.zeros((2, days))
    imp[:, :5] = 10 / net.n / 5          # both strains seeded
    imm = E.Immunity(mask, np.full(net.n, -1, np.int16))
    sim = E.Simulation(net, p, np.ones((E.N_ROWS, days)), dt.date(2026, 8, 1), imm, imp, seed=seed,
                       strains=E.StrainParams(rel_beta=rel, cross=cross))
    sim.run_to(days)
    return sim.infections()


def test_variant_escapes_strain_one_immunity(net):
    inf = two_strain_run(net, np.full(net.n, 1, np.uint8), cross=0.0)     # everyone protected against strain 1 only
    assert (inf.strain == 0).sum() == 0 and (inf.strain == 1).sum() > 100


def test_cross_protection(net):
    none = np.zeros(net.n, np.uint8)
    both = lambda inf: inf.groupby("person").strain.nunique().gt(1).sum()
    full = two_strain_run(net, none, cross=1.0)
    free = two_strain_run(net, none, cross=0.0)
    assert both(full) == 0 and both(free) > 20
    assert not full.duplicated(["person", "strain"]).any() and not free.duplicated(["person", "strain"]).any()


def test_more_transmissible_strain_wins(net):
    inf = two_strain_run(net, np.zeros(net.n, np.uint8), cross=1.0, rel=(1.0, 1.4))
    assert (inf.strain == 1).sum() > 1.3 * (inf.strain == 0).sum()


def test_variant_share_term():
    from fluffhead import strains as ST
    sc = ST.Scenario(strains=2, origin="csa348", variant_share=0.03, variant_share_date=dt.date(2026, 10, 10))
    assert sc.seeded_in("csa348") and sc.follows_origin("csa548") and not sc.seeded_in("csa548")
    assert ST.Scenario.from_json(sc.to_json()) == sc


def test_share_regions_and_variant_led_r_now():
    """r_now is the variant's R where it leads: the dominant strain then gets less transmissibility,
    the more the variant escapes immunity, outspreads it and evades the vaccine."""
    import types
    from fluffhead import calibrate as CB, strains as ST
    sc = ST.Scenario(strains=2, origin="csa348", variant_share=0.95, variant_share_date=dt.date(2026, 10, 24),
                     share_groups=("csa348", "csa488"))
    assert sc.share_applies("csa488") and sc.variant_led("csa488") and sc.variant_led("csa348")
    assert not sc.share_applies("csa548") and not sc.variant_led("csa548")
    assert ST.Scenario.from_json(sc.to_json()) == sc
    rm = CB.RMap(np.array(CB.R_GRID_X), np.array((0.1, 0.3, 0.5, 1.0, 1.5)),
                 np.array([[r * x for x in CB.R_GRID_X] for r in (0.75, 0.65, 0.55, 0.4, 0.3)]), vaccinated=0.4)
    region = types.SimpleNamespace(group="csa348", scenario=sc, r_map=rm, base=DiseaseParams(), humidity=None,
                                   start=dt.date(2026, 8, 1), days=200, cutoff=dt.date(2026, 10, 31))
    th = lambda **k: types.SimpleNamespace(**{"r_now": 1.2, "immune_scale": 1.0, "humidity_floor": 0.5, "ve_infection": 0.4, "seed": 1,
                                              "variant_escape": 0.0, "variant_advantage": 0.0,
                                              "variant_ve_loss": 0.0, **k})
    one = CB.beta_scale(types.SimpleNamespace(**{**vars(region), "scenario": ST.Scenario()}), th())
    plain = types.SimpleNamespace(**{**vars(region), "scenario": ST.Scenario()})
    # a better vaccine takes more out of R, so r_now needs more transmissibility; at the RMap's own effect no change
    assert CB.beta_scale(plain, th(ve_infection=0.65)) > one > CB.beta_scale(plain, th(ve_infection=0.3))
    assert CB.beta_scale(region, th()) == pytest.approx(one)
    b = [CB.beta_scale(region, th(**k)) for k in
         ({"variant_escape": 0.5}, {"variant_advantage": 0.2}, {"variant_ve_loss": 0.5})]
    assert all(x < one for x in b)
    assert b[1] == pytest.approx(one / 1.2)


def test_variant_mechanisms_fix_or_fit():
    import types
    from fluffhead import calibrate as CB, strains as ST
    base = dict(group="csa348", start=dt.date(2026, 8, 1), cutoff=dt.date(2026, 10, 31))
    th = types.SimpleNamespace(variant_escape=0.3, variant_advantage=0.15, variant_ve_loss=0.4)
    for mech, fitted, expect in [("mixed", {"variant_escape", "variant_advantage", "variant_ve_loss"}, (0.3, 0.15, 0.4)),
                                 ("escape", {"variant_escape", "variant_ve_loss"}, (0.3, 0.0, 0.4)),
                                 ("fitness", {"variant_advantage"}, (0.1, 0.15, 0.1))]:
        r = types.SimpleNamespace(scenario=ST.Scenario(strains=2, origin="csa348", mechanism=mech), **base)
        pr = CB.region_prior(r)
        assert set(pr.extra) == fitted | {"variant_start", "variant_intro"}
        for k in fitted:
            assert getattr(pr, k) == ST.MECHANISMS[mech][k]
        assert CB.variant_of(r, th)[:3] == pytest.approx(expect)


def test_national_path(tmp_path):
    """Sum of regions' mean daily admissions, shifted earlier by the infection lag, relative to the cutoff week."""
    import pandas as pd
    from fluffhead import pipeline as P
    dates = pd.date_range("2026-08-01", periods=120)
    for g, scale in (("a", 1.0), ("b", 3.0)):
        y = np.exp(np.minimum(np.arange(120), 90) * 0.03)[None, :, None] * scale * np.ones((4, 1, 2))
        np.savez(tmp_path / f"{g}.npz", daily=y, dates=dates.values)
    cut = dt.date(2026, 10, 1)
    path = P.national_path(tmp_path, cut)
    c = (pd.Timestamp(cut) - dates[0]).days
    assert path[c - 3] == pytest.approx(1.0, rel=0.1)
    lag = P.INFECTION_LAG_DAYS
    assert path[c + 10] / path[c] == pytest.approx(np.exp(0.03 * 10), rel=1e-6)
    assert np.argmax(path) == 90 - lag                   # the admissions peak, moved to infections
    np.savez(tmp_path / "alt.npz", daily=2 * np.load(tmp_path / "b.npz")["daily"], dates=dates.values)
    assert P.national_path(tmp_path, cut, replace={"b": tmp_path / "alt.npz"}) == pytest.approx(path)


def test_pmap_matches_serial():
    from fluffhead.calibrate import pmap
    data = np.arange(10.0)
    f = lambda i: (i, float(data[i] ** 2))       # a closure: reaches the workers by fork
    assert pmap(f, 10, 1) == pmap(f, 10, 3) == [(i, float(i * i)) for i in range(10)]


def test_young_escape_by_age():
    """The "young" mechanism: escape under 25 fitted, from 25 fixed; immunity escaped by age group."""
    import types
    from fluffhead import calibrate as CB, strains as ST
    e = ST.escape_by_age(0.05, 0.5)
    assert e[0] == e[1] == 0.5 and e[3] == e[4] == 0.05
    assert e[2] == pytest.approx(ST.YOUNG_ADULT_SHARE * 0.5 + (1 - ST.YOUNG_ADULT_SHARE) * 0.05)
    r = types.SimpleNamespace(scenario=ST.Scenario(strains=2, origin="csa348", mechanism="young"), group="csa348",
                              start=dt.date(2026, 8, 1), cutoff=dt.date(2026, 10, 31))
    pr = CB.region_prior(r)
    assert set(pr.extra) == {"variant_escape_young", "variant_advantage", "variant_start", "variant_intro"}
    th = types.SimpleNamespace(variant_escape_young=0.6, variant_advantage=0.05)
    v = CB.variant_of(r, th)
    assert v[:3] == pytest.approx((0.05, 0.05, 0.1)) and v[4] == pytest.approx(0.6)
    # other mechanisms: the same escape at all ages
    r2 = types.SimpleNamespace(**{**vars(r), "scenario": ST.Scenario(strains=2, origin="csa348", mechanism="fitness")})
    v2 = CB.variant_of(r2, types.SimpleNamespace(variant_advantage=0.1))
    assert v2[4] == v2[0] == 0.1
    rm = CB.RMap(np.array(CB.R_GRID_X), np.array((0.5, 1.0)), np.ones((2, 5)), young=(np.array([0.3, 0.7]), np.array([0.8, 0.5])))
    assert rm.variant_immunity(0.1, 0.1) == pytest.approx(0.9)
    assert rm.variant_immunity(0.05, 0.5) == pytest.approx(0.65)


def test_escape_by_age_in_immunity(net):
    """Escape applies by age group: only the young lose protection against the variant."""
    from fluffhead import immunity as I, strains as ST
    p = np.full(5, 0.6)
    imm = I.build(net, p, dt.date(2026, 8, 1), 2026, "24", seed=3, escape=ST.escape_by_age(0.0, 1.0)).prior_immune
    s1, s2 = (imm & 1).astype(bool), (imm & 2).astype(bool)
    ag = np.where(net.age_group < 0, 2, net.age_group)
    assert not s2[(ag <= 1)].any()                              # all under-18s escape
    assert np.array_equal(s1[ag >= 3], s2[ag >= 3])             # from 50: none escape


def _slot_run(order, slots):
    """Regions a..f, largest first, with a test that ends each one by hand; returns (started, end)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from fluffhead import pipeline as P
    started, go, cond = [], {g: threading.Event() for g in "abcdef"}, threading.Condition()
    def job(g):
        with cond:
            started.append(g)
            cond.notify_all()
        assert go[g].wait(5)
    def wait_for(n):
        with cond:
            assert cond.wait_for(lambda: len(started) >= n, 5), started
    ex = ThreadPoolExecutor(slots)
    t = threading.Thread(target=P._in_slots, args=(ex, lambda e, g: e.submit(job, g), list("abcdef"), slots, order))
    t.start()
    return started, go, wait_for, t, ex


def test_barbell_order():
    """One slot works down from the largest, the others up from the smallest, each refilling its own way."""
    started, go, wait_for, t, ex = _slot_run("barbell", 3)
    wait_for(3)
    assert started == ["a", "f", "e"]
    go["f"].set()                # a small one ends: the next smallest
    wait_for(4)
    assert started[3] == "d"
    go["a"].set()                # the large one ends: the next largest
    wait_for(5)
    assert started[4] == "b"
    go["e"].set()
    wait_for(6)
    assert started[5] == "c"
    for e in go.values():
        e.set()
    t.join(5)
    ex.shutdown()
    assert sorted(started) == list("abcdef")


def test_largest_order_keeps_size_order():
    started, go, wait_for, t, ex = _slot_run("largest", 2)
    wait_for(2)
    for e in go.values():
        e.set()
    t.join(5)
    ex.shutdown()
    assert started[:2] == ["a", "b"] and sorted(started) == list("abcdef")


def test_start_correction():
    """The median trajectory starts at the data, spread and the past are kept, a missing week changes nothing."""
    from fluffhead import calibrate as CB
    days, cut = 40, 28
    t = np.arange(days)[None, :, None] * 1.0
    daily = np.concatenate([(t + 100) * k for k in (0.8, 1.0, 1.2)])          # [3 traj, days, 1 piece]
    bg = np.array([2.0])
    model = np.median(daily[:, cut - 7:cut].sum(1), axis=0)
    y = model / 2
    out, r = CB.start_correction(daily, y, cut, bg)
    assert r == pytest.approx((y + CB.START_PSEUDO) / (model + CB.START_PSEUDO))
    assert (out[:, :cut] == daily[:, :cut]).all()
    x = (daily[:, cut:] - bg) * r + bg
    assert out[:, cut:] == pytest.approx(x)
    assert (out[2, cut:] - bg) / (out[0, cut:] - bg) == pytest.approx((daily[2, cut:] - bg) / (daily[0, cut:] - bg))
    same, r = CB.start_correction(daily, np.array([np.nan]), cut, bg)
    assert r == pytest.approx(1.0) and same == pytest.approx(daily)
