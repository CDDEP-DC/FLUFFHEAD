"""Strain scenarios: which subtype dominates, and whether a variant of it co-circulates.

Options (fluffhead forecast --subtype --strains --variant-origin --variant-share):

  subtype        dominant subtype, H1N1 or H3N2. H3N2 seasons put more people in hospital
                 per case, mostly older adults, so H3N2 raises the prior admission risk
                 per symptomatic case by age (SEVERITY); the per-state multiplier is fitted
                 on top as before.
  strains        1: the dominant strain only. 2: plus a variant of it (strain 2).
  origin         where the variant starts: a simulation group (e.g. csa348, Los Angeles)
                 or "everywhere" (seeded in every region at the same fitted date).
  variant_share  optional: the reported share of characterized viruses that are the
                 variant, on variant_share_date. It enters the score as a logit-normal
                 term (sd SHARE_LOGIT_SD) on the simulated variant share of resident
                 infections over the SHARE_WINDOW_DAYS days before that date, in regions
                 where the variant is seeded (every region, or the origin only: a national
                 figure understates the origin's share, so give the origin's own if known),
                 or in share_groups if given (e.g. the West Coast's large regions).
  share_groups   optional (--variant-share-regions): the regions variant_share applies to.

r_now. Where the variant leads at the cutoff (the origin, and the regions the reported
share applies to), the fitted r_now is the variant's R: the growth the latest data
show is the variant's, so the dominant strain's transmissibility is derived from it
(calibrate.beta_scale). Elsewhere r_now is the dominant strain's R.

The variant. Relative to the dominant strain it can be more transmissible
(variant_advantage), escape part of prior immunity (variant_escape: that share of people
protected against strain 1 is not protected against it) and be less well covered by the
vaccine (variant_ve_loss: that share of the vaccine's effects on infection and on
admission given infection is lost). How it gains is an assumption, the mechanism
(--variant-mechanism, MECHANISMS): a range is fitted, a number is fixed.
  mixed    some of each (escape 10-40%, advantage 0-20%, vaccine-effect loss 10-50%);
  escape   antigenic drift only: no transmissibility advantage, escape 25-50% and
           vaccine-effect loss 25-50%;
  fitness  a fitness gain with little antigenic change: advantage 5-30%, escape and
           vaccine-effect loss fixed at 10%.
  young    escape concentrated in the young (variant_escape_young, under 25: 30-70%; 25 and
           older: escape fixed at 5%), a small advantage (0-10%) and vaccine-effect loss fixed at
           10%. The young drive transmission (school and household contacts) but are rarely
           admitted, so the variant can take over fast with fewer admissions per infection
           (the engine's admissions are by age). The networks keep age groups only (0-4, 5-17,
           18-49, ...), so under 25 is everyone under 18 plus YOUNG_ADULT_SHARE of 18-49
           (escape_by_age). Elsewhere variant_escape_young equals variant_escape.
Data before the cutoff barely identify escape (it changes R at the cutoff little), and
escape sets how many people the variant can infect, so these ranges are what most
separates the scenarios' season sizes.
An infection with either strain protects against the other with probability CROSS.

Introduction from abroad. From the fitted start date the variant arrives with travellers
from Japan and East Asia (airlinks.asia_arrivals): variant_intro (fitted, log-uniform
INTRO_PRIOR) is the share of arriving travellers infected, so each region's external
introductions scale with its own Asia air traffic (LA, SF, Seattle, New York...). Regions
away from the origin take the start date and variant_intro from the origin trajectory
they follow. A seeded region with no such traffic gets VARIANT_SEEDS infections over
VARIANT_SEED_DAYS days from the start date instead.

Spread within the US. Domestic imports split between the strains by the variant's share
among the people a region's travellers meet:
  - the origin: its own share for the part of its travel to the regions the reported
    share applies to (airlinks.air_share; those regions as one variant zone), the
    dominant strain for the rest of the nation;
  - seeded everywhere: the region's own share (the region standing in for the nation);
  - other regions: the origin's share (each trajectory of the origin's forecast) for the
    region's linkage to the origin (airlinks.py: air passengers to and from the origin's
    airports, plus a distance decay for driving), and the region's own share for the rest. Regions away from the
origin take the variant's escape and advantage from the origin trajectory they follow.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json

import numpy as np
import pandas as pd

SUBTYPES = ("H1N1", "H3N2")
EVERYWHERE = "everywhere"
# admission risk per symptomatic case relative to DiseaseParams (ages 0-4, 5-17, 18-49, 50-64, 65+)
SEVERITY = {"H1N1": np.ones(5), "H3N2": np.array([1.0, 1.0, 1.1, 1.3, 1.6])}
VE_LOSS_PRIOR = (0.1, 0.5)
CROSS = 0.5
VARIANT_SEEDS = 10
INTRO_PRIOR = (1e-5, 1e-2)       # share of travellers arriving from Japan / East Asia infected with the variant
VARIANT_SEED_DAYS = 7
SHARE_LOGIT_SD = 0.5
SHARE_WINDOW_DAYS = 14
ESCAPE_PRIOR = (0.1, 0.4)        # the "mixed" mechanism's ranges
ADVANTAGE_PRIOR = (0.0, 0.2)
MECHANISMS = {
    "mixed": {"variant_escape": ESCAPE_PRIOR, "variant_advantage": ADVANTAGE_PRIOR, "variant_ve_loss": VE_LOSS_PRIOR},
    "escape": {"variant_escape": (0.25, 0.5), "variant_advantage": 0.0, "variant_ve_loss": (0.25, 0.5)},
    "fitness": {"variant_escape": 0.1, "variant_advantage": (0.05, 0.3), "variant_ve_loss": 0.1},
    "young": {"variant_escape": 0.05, "variant_escape_young": (0.3, 0.7), "variant_advantage": (0.0, 0.1),
              "variant_ve_loss": 0.1},
}
YOUNG_ADULT_SHARE = 0.21     # share of 18-49 year olds aged 18-24 (US, 2019 ACS: about 30M of 140M)
START_MIN_LEAD = 14          # the variant's start date is at least this many days before the cutoff


@dataclasses.dataclass(frozen=True)
class Scenario:
    subtype: str = "H1N1"
    strains: int = 1
    origin: str = EVERYWHERE
    variant_share: float | None = None
    variant_share_date: dt.date | None = None
    share_groups: tuple = ()
    mechanism: str = "mixed"

    def __post_init__(self):
        assert self.subtype in SUBTYPES, self.subtype
        assert self.strains in (1, 2), self.strains
        assert (self.variant_share is None) == (self.variant_share_date is None), \
            "variant_share needs variant_share_date"
        object.__setattr__(self, "share_groups", tuple(self.share_groups))
        assert not self.share_groups or self.variant_share is not None, "share_groups needs variant_share"
        assert self.mechanism in MECHANISMS, self.mechanism

    def seeded_in(self, group: str) -> bool:
        """The variant is introduced (and its start, escape and advantage fitted) in this group."""
        return self.strains == 2 and self.origin in (EVERYWHERE, group)

    def follows_origin(self, group: str) -> bool:
        return self.strains == 2 and not self.seeded_in(group)

    def share_applies(self, group: str) -> bool:
        """The reported variant share enters this group's score."""
        if self.strains == 1 or self.variant_share is None:
            return False
        return group in self.share_groups if self.share_groups else self.seeded_in(group)

    def variant_led(self, group: str) -> bool:
        """r_now is the variant's R here (the origin, and where the reported share applies)."""
        return self.strains == 2 and (self.origin == group or self.share_applies(group))

    def severity(self) -> np.ndarray:
        return SEVERITY[self.subtype]

    def label(self) -> str:
        if self.strains == 1:
            return f"{self.subtype}, one strain"
        where = "everywhere at once" if self.origin == EVERYWHERE else f"from {self.origin}"
        share = (f"; variant {self.variant_share:.1%} of characterized viruses on {self.variant_share_date}"
                 if self.variant_share is not None else "")
        if self.share_groups:
            share += " in " + ", ".join(self.share_groups)
        return f"{self.subtype} with a variant ({self.mechanism}), introduced {where}{share}"

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["variant_share_date"] = str(self.variant_share_date) if self.variant_share_date else None
        d["share_groups"] = list(self.share_groups)
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "Scenario":
        d = json.loads(s)
        if d.get("variant_share_date"):
            d["variant_share_date"] = dt.date.fromisoformat(d["variant_share_date"])
        return cls(**d)


@dataclasses.dataclass
class OriginTrack:
    """The origin's forecast, for regions the variant reaches from it: per trajectory, the
    variant's daily share of resident infections and the draw's escape, advantage,
    vaccine-effect loss, start and introduction rate."""
    share: np.ndarray            # [traj, days]
    escape: np.ndarray           # [traj]
    advantage: np.ndarray        # [traj]
    ve_loss: np.ndarray          # [traj]
    start: np.ndarray            # [traj] day of the season the variant starts arriving
    intro: np.ndarray            # [traj] share of arriving travellers from East Asia infected
    escape_young: np.ndarray | None = None    # [traj] escape under 25 (None: as escape)

    def pick(self, seed: int) -> int:
        return int(seed) % len(self.escape)


def escape_by_age(escape: float, escape_young: float) -> np.ndarray:
    """[age group] escape (0-4, 5-17, 18-49, 50-64, 65+): escape_young under 25 (all of 0-17,
    YOUNG_ADULT_SHARE of 18-49), escape from 25."""
    mid = YOUNG_ADULT_SHARE * escape_young + (1 - YOUNG_ADULT_SHARE) * escape
    return np.array([escape_young, escape_young, mid, escape, escape])


def variant_share_daily(region, sim, smooth_days: int = 7) -> np.ndarray:
    """[days] the variant's share of resident infections, trailing smooth_days-day window."""
    n = sim.rn
    j, t, s = sim.rec[0][:n], sim.rec[1][:n], sim.rec[8][:n]
    res = region.net.resident[j]
    day = np.clip(np.floor(t[res]).astype(int), 0, region.days - 1)
    tot = np.bincount(day, minlength=region.days).astype(float)
    var = np.bincount(day, weights=(s[res] == 1).astype(float), minlength=region.days)
    k = np.ones(smooth_days)
    tot_s = np.convolve(tot, k)[:region.days]
    var_s = np.convolve(var, k)[:region.days]
    share = np.where(tot_s > 0, var_s / np.maximum(tot_s, 1), np.nan)
    first = np.argmax(tot_s > 0) if (tot_s > 0).any() else len(share)
    share[:first] = 0.0                        # before any infection: no variant
    return pd.Series(share).ffill().fillna(0.0).to_numpy()   # no infections later: last share
