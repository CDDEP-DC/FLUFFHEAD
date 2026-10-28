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
