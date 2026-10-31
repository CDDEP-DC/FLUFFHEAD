"""Influenza natural history and transmission parameters.

Values are priors for calibration, not fitted estimates. Durations are gamma
distributed (mean, shape), drawn per infection as in the Maryland model's
getDiseaseTimeline. Age groups: 0-4, 5-17, 18-49, 50-64, 65+.

Timeline of one infection, from infection time t:
    latent (not infectious)  ->  presymptomatic (infectious)  ->  symptomatic
    asymptomatic infections skip the symptom stage and are less infectious.
    A share of symptomatic cases is admitted to hospital after onset and
    transmits no further from admission.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from .network import COMM, GQ, HH, N_AGE, SCH, WP


def _age(*v) -> np.ndarray:
    return np.array(v, np.float64)


@dataclass
class DiseaseParams:
    # --- natural history (days)
    latent_mean: float = 1.2           # infection -> infectious; incubation ~1.4-1.9 d (Lessler 2009)
    latent_shape: float = 4.0
    presym_mean: float = 0.6           # infectious before onset
    presym_shape: float = 2.0
    sym_infectious_mean: np.ndarray = field(default_factory=lambda: _age(6.0, 6.0, 4.0, 4.0, 4.5))  # children shed longer (Carrat 2008)
    sym_infectious_shape: float = 4.0
    asym_infectious_mean: float = 3.5
    asym_fraction: np.ndarray = field(default_factory=lambda: _age(0.25, 0.30, 0.35, 0.35, 0.30))  # ~1/3 (Carrat 2008; Leung 2015)
    asym_rel_infectiousness: float = 0.5

    # --- severity
    # admissions per symptomatic infection, unvaccinated; rough CDC burden-model ratios, scaled by
    # `hosp_multiplier` in calibration (season severity)
    hosp_per_symptomatic: np.ndarray = field(default_factory=lambda: _age(0.009, 0.0025, 0.006, 0.016, 0.06))
    hosp_multiplier: float = 1.0
    onset_to_admit_mean: float = 2.5
    onset_to_admit_shape: float = 3.0

    # --- transmission: hazard per contact-day (per tract encounter for community)
    beta_scale: float = 1.0            # calibrated
    beta_layer: np.ndarray = field(default_factory=lambda: np.array([0.035, 0.010, 0.012, 0.010, 0.020]))
    community_rate: float = 4.0        # tract encounters per infectious day
    community_global_share: float = 0.05   # of encounters drawn from the whole region, not the tract
    community_work_share: float = 0.4      # of tract encounters at the work tract (workers on weekdays)
    community_density_gamma: float = 0.15  # tract encounter rate ~ density^gamma, mean 1 within a region (exposure.py)
    # after symptom onset, relative contact by layer (staying home); household unchanged
    sym_contact: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.5, 0.3, 1.0, 0.5]))

    # --- immunity
    prior_immune: np.ndarray = field(default_factory=lambda: _age(0.15, 0.25, 0.35, 0.40, 0.40))  # calibrated jointly with beta
    ve_infection: float = 0.40         # leaky vaccine effect on susceptibility
    ve_hosp_extra: float = 0.20        # further reduction of P(admit | infected) when vaccinated
    vaccine_delay_days: int = 14

    # --- seeding and importation
    seed_infections: int = 20
    import_per_100k_day: float = 0.5   # baseline; scaled by prevalence elsewhere when coupled

    def as_arrays(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = np.asarray(v, np.float64) if isinstance(v, np.ndarray) else v
        return out

    def check(self) -> None:
        for name in ["sym_infectious_mean", "asym_fraction", "hosp_per_symptomatic", "prior_immune"]:
            assert getattr(self, name).shape == (N_AGE,), name
        assert self.beta_layer.shape == (5,) and self.sym_contact.shape == (5,)


LAYER_IDS = {"household": HH, "workplace": WP, "school": SCH, "gq": GQ, "community": COMM}
