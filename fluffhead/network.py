"""Population and contact network for one simulation group.

A region's network is stored as group memberships, not edges (see
greasypop_env/export_groups.jl). Contacts come from four sources:

    household    everyone in the household (complete graph; never stored)
    workplace    stochastic block model by income category   (netw.jl rules)
    school       stochastic block model by grade              (netw.jl rules)
    gq           small-world graph within each group quarters (netw.jl rules)
    community    random draws from the residents + workers of a census tract
                 (GREASYPOP's location matrices; never stored)

Workplace, school and gq edges are generated per group from a seed derived from
(network seed, layer, group), so a group's edges are identical whenever they are
built. `EdgePool` holds the generated groups: "full" mode (B) builds every group
up front, "lazy" mode (C) builds a group when one of its members first becomes
infectious and flushes the pool when it passes a memory budget.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

HH, WP, SCH, GQ, COMM = 0, 1, 2, 3, 4
LAYERS = ["household", "workplace", "school", "gq", "community"]
EDGE_LAYERS = (WP, SCH, GQ)     # layers whose edges are generated and pooled
AGE_BREAKS = np.array([5, 18, 50, 65])   # age groups 0-4, 5-17, 18-49, 50-64, 65+
N_AGE = 5


# ---------------------------------------------------------------- population
@dataclass
class Network:
    """Everything the engine needs about a region's people and groups (all numpy)."""
    n: int
    age_group: np.ndarray        # int8, -1 for dummies (age unknown: treated as 18-49)
    county: np.ndarray           # int16 index into `counties`
    counties: np.ndarray         # county FIPS strings
    resident: np.ndarray         # bool: lives in the region (dummies only work here)
    out_worker: np.ndarray       # bool: works outside the region
    # per layer (HH, WP, SCH, GQ): groups as CSR over members, plus person -> memberships
    grp_ptr: list                # int64 [n_groups+1]
    grp_mem: list                # int32 member person indices
    grp_blk: list                # int16 block (income category / grade) per member
    per_ptr: list                # int64 [n+1] person -> slots in grp_mem
    per_slot: list               # int64 slot indices
    grp_of: list                 # int32 [len(grp_mem)] group of each slot
    # community pools by tract: residents and workers
    home_tract: np.ndarray       # int32, -1 if none
    work_tract: np.ndarray       # int32, -1 if none
    tract_ptr: np.ndarray        # int64 [n_tracts+1] over tract_mem (residents then workers)
    tract_mem: np.ndarray        # int32
    params: dict                 # GREASYPOP network settings (K, associativity, beta)
    tracts: np.ndarray | None = None   # tract GEOIDs (2010) indexed by home/work tract

    @property
    def n_tracts(self) -> int:
        return len(self.tract_ptr) - 1


def _csr(group: np.ndarray, idx: np.ndarray, n_groups: int):
    order = np.lexsort((idx, group))
    g, i = group[order], idx[order]
    ptr = np.zeros(n_groups + 1, np.int64)
    np.add.at(ptr, g + 1, 1)
    return np.cumsum(ptr), i.astype(np.int32), order


def _person_index(mem: np.ndarray, n: int):
    """CSR from person -> positions in the member array."""
    order = np.argsort(mem, kind="stable")
    ptr = np.zeros(n + 1, np.int64)
    np.add.at(ptr, mem.astype(np.int64) + 1, 1)
    return np.cumsum(ptr), order.astype(np.int64)


def convert_export(export_dir: Path, out_dir: Path) -> Path:
    """GREASYPOP groups_export/*.csv -> one compressed network.npz (plus params.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ppl = pd.read_csv(export_dir / "people.csv", dtype={"sch_grade": str}, keep_default_na=False)
    cbg = pd.read_csv(export_dir / "cbg.csv", dtype={"geocode": str})
    n = len(ppl)
    geocode = ppl.cbg_id.map(dict(zip(cbg.cbg_id, cbg.geocode)))
    # dummies' cbg is their home outside the region; county is kept for reference only
    county = geocode.str[:5].fillna("00000")
    counties, county_idx = np.unique(county.to_numpy().astype("U5"), return_inverse=True)
    age = ppl.age.to_numpy()
    age_group = np.where(age < 0, -1, np.searchsorted(AGE_BREAKS, age, side="right")).astype(np.int8)
    resident = (ppl.hh_id >= 0).to_numpy()

    arrays = {"age_group": age_group, "county": county_idx.astype(np.int16), "counties": counties,
              "resident": resident}
    out = np.zeros(n, bool)
    ow = pd.read_csv(export_dir / "outside_workers.csv")
    out[ow.idx.to_numpy()] = True
    arrays["out_worker"] = out

    for L, name in [(HH, "hh"), (WP, "wp"), (SCH, "sch"), (GQ, "gq")]:
        m = pd.read_csv(export_dir / f"members_{name}.csv", dtype={"block": str}, keep_default_na=False)
        n_groups = int(m.group.max()) + 1 if len(m) else 0
        ptr, mem, order = _csr(m.group.to_numpy(), m.idx.to_numpy(), n_groups)
        blk = (pd.factorize(m.block)[0][order] if "block" in m else np.zeros(len(m), int)).astype(np.int16)
        arrays[f"grp_ptr_{L}"], arrays[f"grp_mem_{L}"], arrays[f"grp_blk_{L}"] = ptr, mem, blk

    # tracts: home tract from the household/gq cbg; work tract from the employer location
    home = np.where(resident, geocode.str[:11].fillna(""), "")
    wpm = pd.read_csv(export_dir / "members_wp.csv", usecols=["group", "idx"])
    wp = pd.read_csv(export_dir / "wp.csv", dtype={"geo": str})
    work = np.full(n, "", object)
    work[wpm.idx.to_numpy()] = wpm.group.map(dict(zip(wp.group, wp.geo.str[:11]))).to_numpy()
    tracts = np.unique(np.concatenate([home[home != ""], work[work != ""]]).astype("U11"))
    tix = {t: i for i, t in enumerate(tracts)}
    home_tract = np.array([tix.get(t, -1) for t in home], np.int32)
    work_tract = np.array([tix.get(t, -1) for t in work], np.int32)
    who = np.concatenate([np.flatnonzero(home_tract >= 0), np.flatnonzero(work_tract >= 0)])
    where = np.concatenate([home_tract[home_tract >= 0], work_tract[work_tract >= 0]])
    tptr, tmem, _ = _csr(where, who, len(tracts))
    arrays.update(home_tract=home_tract, work_tract=work_tract, tract_ptr=tptr, tract_mem=tmem,
                  tracts=tracts)

    path = out_dir / "network.npz"
    np.savez_compressed(path, **arrays)
    (out_dir / "params.json").write_text((export_dir / "params.json").read_text())
    log.info("network: %d people, %d tracts -> %s", n, len(tracts), path)
    return path


def load(group: str, regions_dir: Path = C.REGIONS_DIR) -> Network:
    d = regions_dir / group / "network"
    z = np.load(d / "network.npz", allow_pickle=False)
    n = len(z["age_group"])
    gp, gm, gb, pp, ps, go = [], [], [], [], [], []
    for L in (HH, WP, SCH, GQ):
        gp.append(z[f"grp_ptr_{L}"]); gm.append(z[f"grp_mem_{L}"]); gb.append(z[f"grp_blk_{L}"])
        ptr, slot = _person_index(gm[-1], n)
        pp.append(ptr); ps.append(slot)
        go.append(np.repeat(np.arange(len(gp[-1]) - 1, dtype=np.int32), np.diff(gp[-1])))
    return Network(n=n, age_group=z["age_group"], county=z["county"], counties=z["counties"],
                   resident=z["resident"], out_worker=z["out_worker"],
                   grp_ptr=gp, grp_mem=gm, grp_blk=gb, per_ptr=pp, per_slot=ps, grp_of=go,
                   home_tract=z["home_tract"], work_tract=z["work_tract"],
                   tract_ptr=z["tract_ptr"], tract_mem=z["tract_mem"],
                   params=json.loads((d / "params.json").read_text()),
                   tracts=z["tracts"] if "tracts" in z else None)


# ------------------------------------------------------ edge generation (numba)
@nb.njit(cache=True)
def _mix(a, b):
    """64-bit hash of two integers (splitmix64 finalizer), for per-group seeds."""
    x = (a * np.uint64(0x9E3779B97F4A7C15)) ^ b
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@nb.njit(cache=True)
def _rand(st):
    """Uniform [0, 1) from a private splitmix64 stream (st: uint64[1]), separate from the simulation's RNG."""
    st[0] += np.uint64(0x9E3779B97F4A7C15)
    z = st[0]
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@nb.njit(cache=True)
def _randint(st, n):
    return min(int(_rand(st) * n), n - 1)


@nb.njit(cache=True)
def _pairs_bernoulli(st, n_a, n_b, same, p, us, vs, off_a, off_b):
    """Append each possible (u, v) pair with probability p, by geometric skipping."""
    if p <= 0.0:
        return
    total = n_a * (n_a - 1) // 2 if same else n_a * n_b
    if p >= 1.0:
        step_log = 0.0
    else:
        step_log = np.log(1.0 - p)
    k = -1
    while True:
        if p >= 1.0:
            k += 1
        else:
            k += 1 + int(np.floor(np.log(1.0 - _rand(st)) / step_log))
        if k >= total:
            return
        if same:
            # unrank k -> (u, v) with u < v
            u = int((np.sqrt(8.0 * k + 1.0) + 1.0) / 2.0)
            while u * (u - 1) // 2 > k:
                u -= 1
            while (u + 1) * u // 2 <= k:
                u += 1
            v = k - u * (u - 1) // 2
            us.append(off_a + v); vs.append(off_a + u)
        else:
            us.append(off_a + k // n_b); vs.append(off_b + k % n_b)


@nb.njit(cache=True)
def _group_edges(kind, blk, K, assoc, beta, seed):
    """Local edges (u, v) for one group of len(blk) people, following netw.jl.

    kind 0: stochastic block model over `blk`; kind 1: Watts-Strogatz small world.
    Groups smaller than K + 2 are complete graphs.
    """
    st = np.array([np.uint64(seed)], np.uint64)
    n = len(blk)
    us = nb.typed.List.empty_list(nb.int64)
    vs = nb.typed.List.empty_list(nb.int64)
    if n < 2:
        return us, vs
    if n < K + 2:
        for i in range(n):
            for j in range(i + 1, n):
                us.append(i); vs.append(j)
        return us, vs
    if kind == 1:
        half = K // 2
        for i in range(n):
            for s in range(1, half + 1):
                j = (i + s) % n
                if _rand(st) < beta:
                    j = _randint(st, n)
                    while j == i:
                        j = _randint(st, n)
                us.append(i); vs.append(j)
        return us, vs

    # SBM: sort members by block so each block is a contiguous range
    order = np.argsort(blk, kind="mergesort")
    labels = np.unique(blk)
    nb_ = len(labels)
    sizes = np.zeros(nb_, np.int64)
    for i in range(n):
        for b in range(nb_):
            if blk[i] == labels[b]:
                sizes[b] += 1
    starts = np.zeros(nb_, np.int64)
    for b in range(1, nb_):
        starts[b] = starts[b - 1] + sizes[b - 1]
    # c[a, b] = assoc * K * (a == b) + (1 - assoc) * K * n_b / n, capped as in netw.jl
    lu = nb.typed.List.empty_list(nb.int64)
    lv = nb.typed.List.empty_list(nb.int64)
    for a in range(nb_):
        for b in range(a, nb_):
            c = (1.0 - assoc) * K * sizes[b] / n + (assoc * K if a == b else 0.0)
            if a == b:
                c = min(c, sizes[a] - 1)
                p = c / (sizes[a] - 1) if sizes[a] > 1 else 0.0
            else:
                c = min(c, sizes[b])
                p = c / sizes[b]
            _pairs_bernoulli(st, sizes[a], sizes[b], a == b, p, lu, lv, starts[a], starts[b])
    deg = np.zeros(n, np.int64)
    for e in range(len(lu)):
        u, v = order[lu[e]], order[lv[e]]
        us.append(u); vs.append(v)
        deg[u] += 1; deg[v] += 1
    # netw.jl gives isolated vertices one random edge
    for i in range(n):
        if deg[i] == 0:
            j = _randint(st, n)
            while j == i:
                j = _randint(st, n)
            us.append(i); vs.append(j)
            deg[i] += 1; deg[j] += 1
    return us, vs


@nb.njit(cache=True)
def _local_csr(n, us, vs):
    """Symmetric CSR (ptr[n+1], nbr) of local edges, without duplicates or self-loops."""
    cnt = np.zeros(n + 1, np.int64)
    for e in range(len(us)):
        if us[e] != vs[e]:
            cnt[us[e] + 1] += 1; cnt[vs[e] + 1] += 1
    ptr = np.cumsum(cnt)
    nbr = np.empty(ptr[n], np.int32)
    fill = ptr[:-1].copy()
    for e in range(len(us)):
        u, v = us[e], vs[e]
        if u != v:
            nbr[fill[u]] = v; fill[u] += 1
            nbr[fill[v]] = u; fill[v] += 1
    # sort each row and drop duplicate edges (SBM fix-ups and small-world rewiring can repeat)
    out_ptr = np.zeros(n + 1, np.int64)
    w = 0
    for i in range(n):
        row = np.sort(nbr[ptr[i]:ptr[i + 1]])
        last = -1
        for x in row:
            if x != last:
                nbr[w] = x; w += 1; last = x
        out_ptr[i + 1] = w
    return out_ptr, nbr[:w].copy()


# --------------------------------------------------------------- edge pool
@nb.njit(cache=True)
def pool_build(pool, used, start, layer, g, grp_ptr, grp_blk, kind, K, assoc, beta, net_seed):
    """Generate group g of `layer` into the pool. Returns (pool, used); pool may be reallocated.

    Pool block for a group of n members: [ptr (n+1 entries) | neighbors (local indices)].
    """
    a, b = grp_ptr[g], grp_ptr[g + 1]
    n = b - a
    seed = np.int64(_mix(np.uint64(net_seed) * np.uint64(8) + np.uint64(layer), np.uint64(g)) >> np.uint64(33))
    us, vs = _group_edges(kind, grp_blk[a:b], K, assoc, beta, seed)
    ptr, nbr = _local_csr(n, us, vs)
    need = n + 1 + len(nbr)
    if used + need > len(pool):
        new = np.empty(max(2 * len(pool), used + need), np.int32)
        new[:used] = pool[:used]
        pool = new
    pool[used:used + n + 1] = ptr
    pool[used + n + 1:used + need] = nbr
    start[g] = used
    return pool, used + need


@dataclass
class EdgePool:
    """Generated edges for the workplace, school and gq layers of one network."""
    mode: str                    # "full" (B) or "lazy" (C)
    pool: np.ndarray             # int32
    used: int
    start: list                  # per edge layer: int64 [n_groups], -1 if not generated
    budget: int                  # lazy mode: flush when the pool passes this many int32s
    net_seed: int

    @classmethod
    def create(cls, net: Network, mode: str, net_seed: int, budget_bytes: int = 2 << 30) -> "EdgePool":
        start = [np.full(len(net.grp_ptr[L]) - 1, -1, np.int64) for L in (HH, WP, SCH, GQ)]
        ep = cls(mode, np.empty(1 << 20, np.int32), 0, start, budget_bytes // 4, net_seed)
        if mode == "full":
            for L in EDGE_LAYERS:
                for g in range(len(start[L])):
                    ep.ensure(net, L, g)
        return ep

    def ensure(self, net: Network, L: int, g: int) -> None:
        if self.start[L][g] >= 0:
            return
        kind, K, assoc, beta = layer_rule(net, L)
        self.pool, self.used = pool_build(self.pool, self.used, self.start[L], L, g, net.grp_ptr[L],
                                          net.grp_blk[L], kind, K, assoc, beta, self.net_seed)

    def nbytes(self) -> int:
        return self.used * 4


def layer_rule(net: Network, L: int):
    """(kind, K, associativity, small-world beta) for a layer, from GREASYPOP's config."""
    p = net.params
    if L == WP:
        return 0, int(p["workplace_K"]), float(p["income_associativity_coefficient"]), 0.0
    if L == SCH:
        return 0, int(p["school_K"]), float(p["school_associativity_coefficient"]), 0.0
    return 1, int(p["gq_K"]), 0.0, float(p["netw_B"])


# ---------------------------------------------------------- memory planning
def estimate_full_bytes(net: Network) -> int:
    """Pool size if every group is generated (B): ptr + about K neighbors per member."""
    total = 0
    for L in EDGE_LAYERS:
        _, K, _, _ = layer_rule(net, L)
        sizes = np.diff(net.grp_ptr[L])
        deg = np.minimum(sizes - 1, K + 1)   # complete below K+2, about K above
        total += int((sizes + 1).sum() + (sizes * deg).sum())
    return total * 4


def available_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        return 4 << 30


def choose_mode(net: Network, n_parallel: int = 1, headroom: float = 0.6) -> str:
    """'full' if one shared pool plus per-run state fits in the memory available now."""
    need = estimate_full_bytes(net) + n_parallel * state_bytes(net.n)
    mode = "full" if need < headroom * available_bytes() else "lazy"
    log.info("network memory: full pool %.2f GB + %d runs x %.2f GB state; available %.2f GB -> %s",
             estimate_full_bytes(net) / 1e9, n_parallel, state_bytes(net.n) / 1e9,
             available_bytes() / 1e9, mode)
    return mode


def state_bytes(n: int) -> int:
    """Per-run engine state: infection time, susceptibility, flags, plus event heap headroom."""
    return n * (4 + 4 + 1) + n // 4 * 24
