"""Settings for FLUFFHEAD."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
RAW_DIR = DATA_DIR / "raw"              # downloads (not tracked)
REGIONS_DIR = DATA_DIR / "regions"      # region definitions (tracked) and networks (not tracked)
WORK_DIR = REPO / "work"                # GREASYPOP working copies (not tracked)
GREASYPOP_SRC = Path(os.environ.get("GREASYPOP_SRC", REPO.parent / "GREASYPOP-CO"))

# ----------------------------------------------------------------- geography
# 2020 OMB delineations and 2019 county estimates, to match GREASYPOP's 2019 ACS
# with 2010 boundaries (the 2023 delineations use Connecticut planning regions).
DELINEATION_URL = ("https://www2.census.gov/programs-surveys/metro-micro/geographies/"
                   "reference-files/2020/delineation-files/list1_2020.xls")
COUNTY_POP_URL = ("https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/"
                  "counties/totals/co-est2019-alldata.csv")
COMMUTING_URL = ("https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2015/"
                 "commuting-flows-2015/table1.xlsx")

# ------------------------------------------------------------ region choice
N_TOP_REGIONS = 118          # largest CSAs / stand-alone CBSAs (about 80% of the US)
MIN_STATE_COVERAGE = 0.5     # states below this get their largest remaining region added
# small regions are merged into one simulation when strongly linked by commuting
MERGE_MAX_POP = 1_000_000    # only regions smaller than this are merged
MERGE_MIN_LINK = 0.02        # commuters between the two / resident workers of the smaller
MERGE_GROUP_MAX_POP = 2_500_000
COMMUTE_STATE_MIN_SHARE = 0.002  # a state is a GREASYPOP commute state above this share of workers
# states not simulated at all: forecast by scaling a simulated group's per-capita dynamics.
# Puerto Rico has no LODES commuting data, so GREASYPOP can't build its workplaces.
SCALED_STATES = {"72": "csa370"}   # Puerto Rico <- Miami-Port St. Lucie-Fort Lauderdale

FORECAST_DIR = REPO / "forecasts"   # weekly outputs (tracked)
CACHE_DIR = Path(os.environ.get("FLUFFHEAD_CACHE", REPO / "cache"))  # hub downloads (not tracked)

# ------------------------------------------------------------------- hub
TEAM_ABBR = "OHTJHU"
MODEL_ABBR = "FLUFFHEAD"
HUB_REPO = "cdcepi/FluSight-forecast-hub"
HUB_BRANCH = "main"
TARGET_PATHS = {"hosp": "target-data/target-hospital-admissions.csv"}
LOCATIONS_PATH = "auxiliary-data/locations.csv"


def model_id() -> str:
    return f"{TEAM_ABBR}-{MODEL_ABBR}"


def hub_raw(path: str, ref: str | None = None) -> str:
    return f"https://raw.githubusercontent.com/{HUB_REPO}/{ref or HUB_BRANCH}/{path}"


# ----------------------------------------------------- nowcast (from flusight-fm)
# Same settings flusight-fm chose from its revision replay (see its `flusight revisions`).
NOWCAST_WEEKS = 2
NOWCAST_MODEL = "size+growth"
NOWCAST_VARIANCE = "power"
REVISION_START = "2024-11-01"
VERSIONS_START = "2023-08-01"
REVISION_SETTLE_WEEKS = 4
REVISION_SHRINK = 5.0
REVISION_MIN_SD = 0.01
NOWCAST_FALLBACK_SD = 0.05
PRIOR_WEIGHT = 1.0
LOG_OFFSET = 1.0

# --------------------------------------------------------------- targets
# (same definitions as flusight-fm; see the hub's README)
HOSP_TARGET = "wk inc flu hosp"
RATE_TARGET = "wk flu hosp rate change"
PEAK_WEEK_TARGET = "peak week inc flu hosp"
PEAK_INC_TARGET = "peak inc flu hosp"
QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
MEDIAN_IDX = 11
HORIZONS = [-1, 0, 1, 2, 3]
RATE_CHANGE_HORIZONS = [0, 1, 2, 3]
RATE_CHANGE_COLS = {0: ("count_rate0p3", "count_rate1p7"), 1: ("count_rate0p5", "count_rate3"),
                    2: ("count_rate0p7", "count_rate4"), 3: ("count_rate1", "count_rate5")}
RATE_CHANGE_CATEGORIES = ["large_decrease", "decrease", "stable", "increase", "large_increase"]
MIN_STABLE_COUNT = 10
MAX_POP_FRACTION = 0.3
SEASON_START = dt.date(2026, 10, 10)   # first and last Saturdays eligible for the peak week
SEASON_END = dt.date(2027, 5, 29)

# ------------------------------------------------------------- ensemble
N_SAMPLES = 2000          # joint samples across regions
REGION_RHO = 0.7          # copula correlation between regions' forecast-window totals
