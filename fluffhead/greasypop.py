"""Running GREASYPOP for each simulation group.

GREASYPOP-CO itself is never modified. Its inputs live in one national store
(data/raw/greasypop), downloaded once per state. Each group gets a working
directory work/<group>/ holding a copy of the GREASYPOP code, the group's
geo.json, and symlinks to only the input files it needs, matching what
GREASYPOP's download_data.R would have fetched for that region:

    census/<ST>      ACS + decennial tables for the group's states
    pums/            PUMS for the group's states (use_pums)
    geo/             crosswalks (from the repo) + block group shapefiles
    work/            CBP, LODES od main + wac (group states), od aux (+ commute states)
    school/          NCES 2018-19 files

Steps: census.py -> CO.jl -> synthpop_groups.jl -> export_groups.jl -> convert.
synthpop_groups.jl is synthpop.jl without GREASYPOP's edge building (FLUFFHEAD
rebuilds edges from memberships). The export is converted to
data/regions/<group>/network/ (see network.py) and the bulky intermediates are
deleted unless keep=True.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from . import config as C
from .regions import census_key

log = logging.getLogger(__name__)

STORE = C.RAW_DIR / "greasypop"
NCES = C.RAW_DIR / "nces"
ENV_DIR = Path(__file__).resolve().parent / "greasypop_env"
PY = C.REPO / ".venv-greasypop" / "bin" / "python"
CODE_FILES = ["census.py", "ipfn.py", "CO.jl", "synthpop.jl", "households.jl", "schools.jl",
              "workplaces.jl", "netw.jl", "utils.jl", "fileutils.jl", "config.json"]
STEPS = ["census", "co", "synthpop", "export", "convert"]
INTERMEDIATES = ["jlse", "processed", "pop_export", "groups_export"]
NO_LODES = {"72"}   # LEHD does not cover Puerto Rico


_STATES = pd.DataFrame([
    ("01", "AL"), ("02", "AK"), ("04", "AZ"), ("05", "AR"), ("06", "CA"), ("08", "CO"), ("09", "CT"),
    ("10", "DE"), ("11", "DC"), ("12", "FL"), ("13", "GA"), ("15", "HI"), ("16", "ID"), ("17", "IL"),
    ("18", "IN"), ("19", "IA"), ("20", "KS"), ("21", "KY"), ("22", "LA"), ("23", "ME"), ("24", "MD"),
    ("25", "MA"), ("26", "MI"), ("27", "MN"), ("28", "MS"), ("29", "MO"), ("30", "MT"), ("31", "NE"),
    ("32", "NV"), ("33", "NH"), ("34", "NJ"), ("35", "NM"), ("36", "NY"), ("37", "NC"), ("38", "ND"),
    ("39", "OH"), ("40", "OK"), ("41", "OR"), ("42", "PA"), ("44", "RI"), ("45", "SC"), ("46", "SD"),
    ("47", "TN"), ("48", "TX"), ("49", "UT"), ("50", "VT"), ("51", "VA"), ("53", "WA"), ("54", "WV"),
    ("55", "WI"), ("56", "WY"), ("72", "PR")], columns=["fips", "abbr"])
ABBR = dict(zip(_STATES.fips, _STATES.abbr))


# ------------------------------------------------------------------- store
def init_store() -> None:
    """Create the shared store: crosswalks and CBP from the GREASYPOP repo, NCES files."""
    for sub in ["census", "pums", "geo", "work", "school"]:
        (STORE / sub).mkdir(parents=True, exist_ok=True)
    (STORE / ".here").touch()   # so R's here() resolves to the store
    for f in (C.GREASYPOP_SRC / "geo").iterdir():
        if f.is_file() and not (STORE / "geo" / f.name).exists():
            shutil.copy2(f, STORE / "geo" / f.name)
    cbp = C.GREASYPOP_SRC / "work" / "cbp16co.zip"
    if not (STORE / "work" / cbp.name).exists():
        shutil.copy2(cbp, STORE / "work" / cbp.name)
    # EDGE xlsx + Shapefile_SCH + CCD csv files. The CCD zips hold a csv and a sas file each,
    # which pandas can't read directly, so extract the csv.
    for z in NCES.glob("ccd_*.zip"):
        with zipfile.ZipFile(z) as zf:
            name = next(m for m in zf.namelist() if m.lower().endswith(".csv"))
            if not (NCES / name).exists() and not _school_csv(z).exists():
                zf.extract(name, NCES)
    # membership (052) is 2.5 GB by race/sex/grade; census.py only uses each school's total row
    raw = next(NCES.glob("ccd_[sS][cC][hH]_052_*_l_*.csv"), None)
    totals = NCES / "ccd_sch_052_1819_totals.csv"
    if raw is not None and not totals.exists():
        chunks = pd.read_csv(raw, dtype=str, encoding="latin-1", chunksize=2_000_000)
        pd.concat(c[c.TOTAL_INDICATOR == "Education Unit Total"] for c in chunks).to_csv(totals, index=False)
        raw.unlink()
    for f in [*NCES.glob("EDGE_*.xlsx"), NCES / "Shapefile_SCH", *NCES.glob("ccd_sch_029_*.csv"),
              *NCES.glob("ccd_sch_059_*.csv"), totals]:
        dest = STORE / "school" / f.name
        if not dest.exists():
            dest.symlink_to(f)


def _school_csv(z: Path) -> Path:
    """Where a CCD zip's csv ends up after extraction (052 is replaced by its totals file)."""
    return NCES / "ccd_sch_052_1819_totals.csv" if "_052_" in z.name else z.with_suffix(".csv")


def downloaded(st: str) -> bool:
    a = ABBR[st]
    ok = ((STORE / "census" / a).is_dir() and len(list((STORE / "census" / a).iterdir())) >= 18
          and (STORE / "pums" / f"psam_p{st}.csv").exists()
          and (STORE / "geo" / f"tl_2019_{st}_bg.zip").exists())
    return ok and (st in NO_LODES or bool(_lodes(st, "wac_S000")))


def _lodes(st: str, kind: str) -> list[Path]:
    """LODES files for a state (the year can be earlier than 2019 where LEHD lacks 2019)."""
    return sorted((STORE / "work").glob(f"{ABBR[st].lower()}_{kind}_JT01_*.csv.gz"))


def download(states: list[str], log_path: Path | None = None) -> None:
    """Fetch GREASYPOP inputs for these states (skips what is already there)."""
    init_store()
    todo = [s for s in states if not downloaded(s)]
    if not todo:
        return
    log.info("downloading GREASYPOP inputs for %s", " ".join(todo))
    for st in todo:
        if st not in NO_LODES:
            fetch_lodes(st)
    env = dict(os.environ, CENSUS_API_KEY=census_key())
    cmd = ["Rscript", str(ENV_DIR / "download.R"), str(STORE), str(C.GREASYPOP_SRC), *todo]
    with open(log_path or os.devnull, "a") as out:
        subprocess.run(cmd, cwd=STORE, env=env, check=True, stdout=out, stderr=subprocess.STDOUT)


# --------------------------------------------------------------------- LODES
LODES_URL = "https://lehd.ces.census.gov/data/lodes/LODES7/{st}/{kind}/{st}_{part}_JT01_{year}.csv.gz"
LODES_YEARS = range(2019, 2009, -1)   # latest available; LODES7 lacks some state-years (AK after 2016)


def fetch_lodes(st: str) -> None:
    """od main, od aux and wac S000 for a state, aggregated from blocks to block groups.

    Same output as GREASYPOP's pull_LODES (lehdr, agg_geo = "bg"): year, state,
    w_geocode[, h_geocode], job counts. Streams the block-level file in chunks.
    """
    import requests

    a = ABBR[st].lower()
    if _lodes(st, "wac_S000") and _lodes(st, "od_main") and _lodes(st, "od_aux"):
        return
    for year in LODES_YEARS:
        url = LODES_URL.format(st=a, kind="wac", part="wac_S000", year=year)
        if requests.head(url, timeout=60).status_code == 200:
            break
    else:
        raise RuntimeError(f"no LODES7 year available for {a}")
    if year != 2019:
        log.warning("LODES for %s uses %d (2019 unavailable)", a, year)
    for kind, part, keys in [("od", "od_main", ["w_geocode", "h_geocode"]), ("od", "od_aux", ["w_geocode", "h_geocode"]),
                             ("wac", "wac_S000", ["w_geocode"])]:
        out = STORE / "work" / f"{a}_{part}_JT01_{year}.csv.gz"
        if out.exists():
            continue
        url = LODES_URL.format(st=a, kind=kind, part=part, year=year)
        log.info("LODES %s", url)
        parts = []
        for chunk in pd.read_csv(url, dtype={k: str for k in keys}, chunksize=1_000_000):
            chunk = chunk.drop(columns=["createdate"], errors="ignore")
            for k in keys:
                chunk[k] = chunk[k].str[:12]
            parts.append(chunk.groupby(keys, sort=False).sum())
        agg = pd.concat(parts).groupby(level=list(range(len(keys)))).sum().reset_index()
        agg.insert(0, "state", a.upper())
        agg.insert(0, "year", year)
        tmp = out.with_suffix(".tmp")
        agg.to_csv(tmp, index=False, compression="gzip")
        tmp.rename(out)


# ------------------------------------------------------------- working dir
def _link(src: Path, dest: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(src)


def prepare(group: str) -> Path:
    """Build work/<group>/ with code, geo.json and links to the inputs this group needs."""
    geo = json.loads((C.REGIONS_DIR / group / "geo.json").read_text())
    main = sorted({g[:2] for g in geo["geos"]})
    commute = [s for s in geo["commute_states"] if s not in NO_LODES]
    wd = C.WORK_DIR / group
    for sub in ["census", "pums", "geo", "work", "school"]:
        if (wd / sub).exists():
            shutil.rmtree(wd / sub)
        (wd / sub).mkdir(parents=True)
    for f in CODE_FILES:
        shutil.copy2(C.GREASYPOP_SRC / f, wd / f)
    for f in ["synthpop_groups.jl", "export_groups.jl"]:
        shutil.copy2(ENV_DIR / f, wd / f)
    (wd / "geo.json").write_text(json.dumps(geo, indent=2) + "\n")

    for st in main:
        _link(STORE / "census" / ABBR[st], wd / "census" / ABBR[st])
        _link(STORE / "geo" / f"tl_2019_{st}_bg.zip", wd / "geo" / f"tl_2019_{st}_bg.zip")
    for st in geo["use_pums"]:
        for kind in "hp":
            _link(STORE / "pums" / f"psam_{kind}{st}.csv", wd / "pums" / f"psam_{kind}{st}.csv")
    for f in (STORE / "geo").iterdir():
        if f.is_file() and not f.name.startswith("tl_"):
            _link(f, wd / "geo" / f.name)
    _link(STORE / "work" / "cbp16co.zip", wd / "work" / "cbp16co.zip")
    for st in main:
        if st in NO_LODES:
            continue
        for f in _lodes(st, "od_main")[-1:] + _lodes(st, "wac_S000")[-1:]:
            _link(f, wd / "work" / f.name)
    for st in sorted(set(main) | set(commute)):
        if st in NO_LODES:
            continue
        for f in _lodes(st, "od_aux")[-1:]:
            _link(f, wd / "work" / f.name)
    for f in (STORE / "school").iterdir():
        _link(f.resolve(), wd / "school" / f.name)
    return wd


# --------------------------------------------------------------------- run
def run(group: str, steps: list[str] | None = None, procs: int = 4, keep: bool = False) -> Path:
    """Run GREASYPOP steps for a group; logs go to work/<group>/greasypop.log."""
    from . import network

    wd = C.WORK_DIR / group
    steps = steps or STEPS
    env = dict(os.environ, JULIA_PROJECT=str(ENV_DIR))
    cmds = {
        "census": [str(PY), "census.py"],
        "co": ["julia", f"-p{procs}", "CO.jl"],
        "synthpop": ["julia", "synthpop_groups.jl"],
        "export": ["julia", "export_groups.jl"],
    }
    out_dir = C.REGIONS_DIR / group / "network"
    with open(wd / "greasypop.log", "a") as out:
        for s in steps:
            log.info("%s: %s", group, s)
            out.write(f"\n===== {s}\n"); out.flush()
            if s == "convert":
                network.convert_export(wd / "groups_export", out_dir)
                if not keep:
                    for d in INTERMEDIATES:
                        shutil.rmtree(wd / d, ignore_errors=True)
                continue
            subprocess.run(cmds[s], cwd=wd, env=env, check=True, stdout=out, stderr=subprocess.STDOUT)
    return out_dir


def build(group: str, procs: int = 4, keep: bool = False) -> Path:
    """Download (if needed), prepare and run GREASYPOP for one group."""
    geo = json.loads((C.REGIONS_DIR / group / "geo.json").read_text())
    states = sorted({g[:2] for g in geo["geos"]} | set(geo["commute_states"]) | set(geo["use_pums"]))
    download(states, C.REPO / "logs" / "download.log")
    prepare(group)
    return run(group, procs=procs, keep=keep)
