"""Event-driven transmission on the contact network.

As in the Maryland model, nothing is updated between events and an agent's whole
disease course is drawn when they are infected. What changes is where onward
infections go: instead of a Poisson number of anonymous contacts, each of the
agent's actual contacts gets candidate transmission events at the layer's hazard
over each infectious segment of the timeline (presymptomatic, symptomatic,
asymptomatic). Community contacts are drawn from the tract's residents and workers,
at a rate scaled by the tract's density multiplier. Importations pick people in
proportion to their importation weight (exposure.py).

Time-varying effects (weekends, school calendars, seasonality) and the target's
susceptibility (prior immunity, vaccination) are applied when an event fires, by
thinning: the event infects with probability calendar[layer, day] * susceptibility.
Calendar values are <= 1, so this is exact for a time-varying hazard.

State is a set of numpy arrays (infection time per person, the event heap, and the
infection records), so a run can be saved at a date and continued later.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

from .disease import DiseaseParams
from .network import COMM, EDGE_LAYERS, GQ, HH, N_AGE, SCH, WP, EdgePool, Network, layer_rule, pool_build

INF = np.float32(np.inf)
COMM_WORK = 5      # calendar row for community encounters at the work tract
IMPORT = 6         # calendar row for importations (always 1)
N_ROWS = 7
LAYER_NAMES = ["household", "workplace", "school", "gq", "community", "community_work", "import"]

# scalar parameter vector layout
(S_LAT_M, S_LAT_K, S_PRE_M, S_PRE_K, S_SYM_K, S_ASYM_M, S_ASYM_REL, S_ADM_M, S_ADM_K,
 S_VE_INF, S_VE_HOSP, S_COMM_RATE, S_COMM_GLOBAL, S_COMM_WORK, S_VACC_DELAY) = range(15)
N_SCALARS = 15


def pack(p: DiseaseParams):
    """Scalars + per-age and per-layer arrays for the compiled loop."""
    sc = np.zeros(N_SCALARS)
    sc[[S_LAT_M, S_LAT_K, S_PRE_M, S_PRE_K, S_SYM_K, S_ASYM_M, S_ASYM_REL, S_ADM_M, S_ADM_K,
        S_VE_INF, S_VE_HOSP, S_COMM_RATE, S_COMM_GLOBAL, S_COMM_WORK, S_VACC_DELAY]] = [
        p.latent_mean, p.latent_shape, p.presym_mean, p.presym_shape, p.sym_infectious_shape,
        p.asym_infectious_mean, p.asym_rel_infectiousness, p.onset_to_admit_mean, p.onset_to_admit_shape,
        p.ve_infection, p.ve_hosp_extra, p.community_rate, p.community_global_share,
        p.community_work_share, p.vaccine_delay_days]
    beta = p.beta_layer * p.beta_scale
    sym_contact = np.append(p.sym_contact, p.sym_contact[COMM])   # community at work tract
    return (sc, p.sym_infectious_mean.astype(np.float64), p.asym_fraction.astype(np.float64),
            (p.hosp_per_symptomatic * p.hosp_multiplier).astype(np.float64), beta.astype(np.float64),
            sym_contact.astype(np.float64))


# ------------------------------------------------------------------ heap
@nb.njit(cache=True)
def _push(ht, hj, hs, hl, n, t, j, s, L):
    if n == len(ht):
        m = 2 * len(ht)
        ht2 = np.empty(m, ht.dtype); ht2[:n] = ht
        hj2 = np.empty(m, hj.dtype); hj2[:n] = hj
        hs2 = np.empty(m, hs.dtype); hs2[:n] = hs
        hl2 = np.empty(m, hl.dtype); hl2[:n] = hl
        ht, hj, hs, hl = ht2, hj2, hs2, hl2
    i = n
    while i > 0:
        p = (i - 1) // 2
        if ht[p] <= t:
            break
        ht[i] = ht[p]; hj[i] = hj[p]; hs[i] = hs[p]; hl[i] = hl[p]
        i = p
    ht[i] = t; hj[i] = j; hs[i] = s; hl[i] = L
    return ht, hj, hs, hl, n + 1


@nb.njit(cache=True)
def _pop(ht, hj, hs, hl, n):
    t, j, s, L = ht[0], hj[0], hs[0], hl[0]
    n -= 1
    lt, lj, ls, ll = ht[n], hj[n], hs[n], hl[n]
    i = 0
    while True:
        c = 2 * i + 1
        if c >= n:
            break
        if c + 1 < n and ht[c + 1] < ht[c]:
            c += 1
        if ht[c] >= lt:
            break
        ht[i] = ht[c]; hj[i] = hj[c]; hs[i] = hs[c]; hl[i] = hl[c]
        i = c
    if n > 0:
        ht[i] = lt; hj[i] = lj; hs[i] = ls; hl[i] = ll
    return t, j, s, L, n


@nb.njit(cache=True)
def _grow(a, n):
    if n < len(a):
        return a
    b = np.empty(2 * len(a), a.dtype)
    b[:len(a)] = a
    return b


# ------------------------------------------------------------- main loop
@nb.njit(cache=True)
def _seed(s):
    np.random.seed(s)


@nb.njit(cache=True)
def _run(day0, day1,
         age_group, home_tract, work_tract, tract_ptr, tract_mem, comm_mult, imp_cum,
         gp, gm, gb, pp, ps, go,
         starts, pool, used, budget, kinds, Ks, assocs, betas_sw, net_seed,
         prior_imm, vacc_day,
         sc, sym_mean, asym_frac, hosp_p, beta, sym_contact,
         cal, imp,
         t_inf, ht, hj, hs, hl, hn,
         r_j, r_t, r_src, r_L, r_sym, r_on, r_adm, r_vac, rn):
    n = len(t_inf)
    T = cal.shape[1]
    cmax = max(comm_mult.max(), 1.0) if len(comm_mult) > 0 else 1.0
    for d in range(day0, day1):
        # importations: Poisson count per day (hazard per unit weight x total weight), people picked
        # in proportion to their weight, uniform times within the day
        W = imp_cum[n - 1] if n > 0 else 0.0
        if W > 0:
            for _ in range(np.random.poisson(imp[d] * W)):
                k = min(np.searchsorted(imp_cum, np.random.random() * W, side="right"), n - 1)
                ht, hj, hs, hl, hn = _push(ht, hj, hs, hl, hn, d + np.random.random(), k, -1, IMPORT)

        while hn > 0 and ht[0] < d + 1:
            t, k, src, L, hn = _pop(ht, hj, hs, hl, hn)
            if t_inf[k] < INF:
                continue
            day = min(int(t), T - 1)
            vaccinated = vacc_day[k] >= 0 and t >= vacc_day[k] + sc[S_VACC_DELAY]
            p_acc = cal[L, day]
            if prior_imm[k]:
                p_acc = 0.0
            elif vaccinated:
                p_acc *= 1.0 - sc[S_VE_INF]
            if np.random.random() >= p_acc:
                continue

            # ---- infect k at time t: draw the timeline
            t_inf[k] = t
            a = age_group[k]
            if a < 0:
                a = 2
            ti = t + np.random.gamma(sc[S_LAT_K], sc[S_LAT_M] / sc[S_LAT_K])
            seg_a = np.zeros(2); seg_b = np.zeros(2); seg_rel = np.zeros(2); seg_sym = np.zeros(2, np.bool_)
            nseg = 0
            sym = np.random.random() >= asym_frac[a]
            on = -1.0
            adm = -1.0
            if sym:
                on = ti + np.random.gamma(sc[S_PRE_K], sc[S_PRE_M] / sc[S_PRE_K])
                end = on + np.random.gamma(sc[S_SYM_K], sym_mean[a] / sc[S_SYM_K])
                ph = hosp_p[a] * ((1.0 - sc[S_VE_HOSP]) if vaccinated else 1.0)
                if np.random.random() < ph:
                    adm = on + np.random.gamma(sc[S_ADM_K], sc[S_ADM_M] / sc[S_ADM_K])
                    end = min(end, adm)   # no transmission after admission
                seg_a[0] = ti; seg_b[0] = on; seg_rel[0] = 1.0; seg_sym[0] = False
                seg_a[1] = on; seg_b[1] = max(on, end); seg_rel[1] = 1.0; seg_sym[1] = True
                nseg = 2
            else:
                seg_a[0] = ti
                seg_b[0] = ti + np.random.gamma(4.0, sc[S_ASYM_M] / 4.0)
                seg_rel[0] = sc[S_ASYM_REL]; seg_sym[0] = False
                nseg = 1

            r_j = _grow(r_j, rn); r_t = _grow(r_t, rn); r_src = _grow(r_src, rn); r_L = _grow(r_L, rn)
            r_sym = _grow(r_sym, rn); r_on = _grow(r_on, rn); r_adm = _grow(r_adm, rn); r_vac = _grow(r_vac, rn)
            r_j[rn] = k; r_t[rn] = t; r_src[rn] = src; r_L[rn] = L
            r_sym[rn] = sym; r_on[rn] = on; r_adm[rn] = adm; r_vac[rn] = vaccinated
            rn += 1

            # ---- schedule candidate transmissions along each contact
            for s in range(nseg):
                D = seg_b[s] - seg_a[s]
                if D <= 0.0:
                    continue
                for Lc in range(4):
                    m = sym_contact[Lc] if seg_sym[s] else 1.0
                    lam = beta[Lc] * seg_rel[s] * m * D
                    if lam <= 0.0:
                        continue
                    for q in range(pp[Lc][k], pp[Lc][k + 1]):
                        slot = ps[Lc][q]
                        g = go[Lc][slot]
                        g0 = gp[Lc][g]
                        ng = gp[Lc][g + 1] - g0
                        if Lc == HH:
                            for x in range(g0, g0 + ng):
                                c = gm[Lc][x]
                                if c != k and t_inf[c] == INF:
                                    for _ in range(np.random.poisson(lam)):
                                        ht, hj, hs, hl, hn = _push(ht, hj, hs, hl, hn, seg_a[s] + np.random.random() * D, c, k, Lc)
                            continue
                        if starts[Lc][g] < 0:
                            if used > budget:
                                for L2 in range(1, 4):
                                    starts[L2][:] = -1
                                used = 0
                            pool, used = pool_build(pool, used, starts[Lc], Lc, g, gp[Lc], gb[Lc],
                                                    kinds[Lc], Ks[Lc], assocs[Lc], betas_sw[Lc], net_seed)
                        st = starts[Lc][g]
                        loc = slot - g0
                        for e in range(pool[st + loc], pool[st + loc + 1]):
                            c = gm[Lc][g0 + pool[st + ng + 1 + e]]
                            if t_inf[c] == INF:
                                for _ in range(np.random.poisson(lam)):
                                    ht, hj, hs, hl, hn = _push(ht, hj, hs, hl, hn, seg_a[s] + np.random.random() * D, c, k, Lc)

                # community encounters that transmit (thinned Poisson), at home or work tract; drawn
                # at the highest tract multiplier and kept with probability multiplier / highest
                m = sym_contact[COMM] if seg_sym[s] else 1.0
                for _ in range(np.random.poisson(sc[S_COMM_RATE] * cmax * D * m * beta[COMM] * seg_rel[s])):
                    tt = seg_a[s] + np.random.random() * D
                    row = COMM
                    if np.random.random() < sc[S_COMM_GLOBAL]:
                        if np.random.random() * cmax >= 1.0:
                            continue
                        c = np.random.randint(n)
                    else:
                        tr = home_tract[k]
                        if work_tract[k] >= 0 and (tr < 0 or np.random.random() < sc[S_COMM_WORK]):
                            tr = work_tract[k]
                            row = COMM_WORK
                        if tr < 0 or np.random.random() * cmax >= comm_mult[tr]:
                            continue
                        c = tract_mem[tract_ptr[tr] + np.random.randint(tract_ptr[tr + 1] - tract_ptr[tr])]
                    if c != k and t_inf[c] == INF:
                        ht, hj, hs, hl, hn = _push(ht, hj, hs, hl, hn, tt, c, k, row)
    return (pool, used, ht, hj, hs, hl, hn,
            r_j, r_t, r_src, r_L, r_sym, r_on, r_adm, r_vac, rn)


# --------------------------------------------------------------- wrapper
@dataclass
class Immunity:
    prior_immune: np.ndarray      # bool [n]
    vacc_day: np.ndarray          # int16 [n]: day index vaccinated, -1 never


class Simulation:
    """One stochastic run over one region, advanced a day range at a time."""

    def __init__(self, net: Network, params: DiseaseParams, calendar: np.ndarray, start: dt.date,
                 immunity: Immunity, imports: np.ndarray, import_weight: np.ndarray | None = None,
                 mode: str = "full", net_seed: int = 1, seed: int = 1, pool: EdgePool | None = None,
                 lazy_budget_bytes: int = 2 << 30, community_mult: np.ndarray | None = None):
        """`imports`: daily importation hazard per unit of `import_weight` (default weight 1 for
        residents, 0 otherwise); `community_mult`: per-tract community rate multiplier (default 1)."""
        params.check()
        self.net, self.params, self.start = net, params, start
        self.cal = np.ascontiguousarray(calendar, np.float64)
        assert self.cal.shape[0] == N_ROWS
        self.imp = np.asarray(imports, np.float64)
        w = net.resident.astype(np.float64) if import_weight is None else np.asarray(import_weight, np.float64)
        self.imp_cum = np.cumsum(w)
        self.comm_mult = (np.ones(net.n_tracts) if community_mult is None
                          else np.ascontiguousarray(community_mult, np.float64))
        self.imm = immunity
        self.pool = pool or EdgePool.create(net, mode, net_seed, lazy_budget_bytes)
        self.day = 0
        self.t_inf = np.full(net.n, INF, np.float32)
        self.heap = (np.empty(1 << 16), np.empty(1 << 16, np.int32), np.empty(1 << 16, np.int32),
                     np.empty(1 << 16, np.int8), 0)
        self.rec = [np.empty(1 << 14, np.int32), np.empty(1 << 14), np.empty(1 << 14, np.int32),
                    np.empty(1 << 14, np.int8), np.empty(1 << 14, np.bool_), np.empty(1 << 14),
                    np.empty(1 << 14), np.empty(1 << 14, np.bool_)]
        self.rn = 0
        rules = [layer_rule(net, L) if L in EDGE_LAYERS else (0, 0, 0.0, 0.0) for L in range(4)]
        self.kinds = np.array([r[0] for r in rules], np.int64)
        self.Ks = np.array([r[1] for r in rules], np.int64)
        self.assocs = np.array([r[2] for r in rules])
        self.betas_sw = np.array([r[3] for r in rules])
        _seed(seed)

    def reseed(self, seed: int) -> None:
        _seed(seed)

    def run_to(self, day: int) -> None:
        """Advance to the start of `day` (days since `start`)."""
        day = min(day, self.cal.shape[1])
        if day <= self.day:
            return
        net, ep, imm = self.net, self.pool, self.imm
        packed = pack(self.params)
        budget = ep.budget if ep.mode == "lazy" else np.iinfo(np.int64).max
        out = _run(self.day, day,
                   net.age_group, net.home_tract, net.work_tract, net.tract_ptr, net.tract_mem,
                   self.comm_mult, self.imp_cum,
                   tuple(net.grp_ptr), tuple(net.grp_mem), tuple(net.grp_blk), tuple(net.per_ptr), tuple(net.per_slot), tuple(net.grp_of),
                   tuple(ep.start), ep.pool, ep.used, budget, self.kinds, self.Ks, self.assocs, self.betas_sw,
                   ep.net_seed, imm.prior_immune, imm.vacc_day,
                   *packed, self.cal, self.imp,
                   self.t_inf, *self.heap, *self.rec, self.rn)
        ep.pool, ep.used = out[0], out[1]
        self.heap = out[2:7]
        self.rec = list(out[7:15])
        self.rn = out[15]
        self.day = day

    # ------------------------------------------------------------- outputs
    def infections(self) -> pd.DataFrame:
        """One row per infection so far: who, when, by whom, where, outcome."""
        n = self.rn
        j, t, src, L, sym, on, adm, vac = (a[:n] for a in self.rec)
        return pd.DataFrame({"person": j, "t": t, "infector": src, "layer": np.array(LAYER_NAMES)[L],
                             "symptomatic": sym, "onset": np.where(on >= 0, on, np.nan),
                             "admit": np.where(adm >= 0, adm, np.nan), "vaccinated": vac,
                             "age_group": self.net.age_group[j], "county": self.net.counties[self.net.county[j]],
                             "resident": self.net.resident[j]})

    def daily_admissions(self, by_county: bool = True) -> pd.DataFrame:
        """Admissions per day (residents only), including those already scheduled past today."""
        inf = self.infections()
        inf = inf[inf.resident & inf.admit.notna()]
        day = np.floor(inf.admit).astype(int)
        date = pd.to_datetime(self.start) + pd.to_timedelta(day, unit="D")
        keys = [date.rename("date")] + ([inf.county] if by_county else [])
        return inf.groupby(keys).size().rename("admissions").reset_index()

    def save(self, path: Path) -> None:
        np.savez_compressed(path, day=self.day, t_inf=self.t_inf, heap_n=self.heap[4],
                            ht=self.heap[0][:self.heap[4]], hj=self.heap[1][:self.heap[4]],
                            hs=self.heap[2][:self.heap[4]], hl=self.heap[3][:self.heap[4]],
                            **{f"rec{i}": a[:self.rn] for i, a in enumerate(self.rec)})

    def load(self, path: Path) -> None:
        z = np.load(path)
        self.day = int(z["day"]); self.t_inf = z["t_inf"]
        hn = int(z["heap_n"])
        size = max(1 << 16, 2 * hn)
        heap = [np.empty(size), np.empty(size, np.int32), np.empty(size, np.int32), np.empty(size, np.int8)]
        for a, k in zip(heap, ["ht", "hj", "hs", "hl"]):
            a[:hn] = z[k]
        self.heap = (*heap, hn)
        self.rn = len(z["rec0"])
        self.rec = []
        for i in range(8):
            a = z[f"rec{i}"]
            b = np.empty(max(1 << 14, 2 * len(a)), a.dtype); b[:len(a)] = a
            self.rec.append(b)
