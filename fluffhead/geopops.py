"""Networks from 2023 Census data, with GeoPops' data pipeline.

GeoPops (github.com/GeoPopsHub/geopops, installed at a pinned commit in .venv-geopops,
never modified) downloads and processes the 2019-2023 ACS, the 2020 Census, LODES8 2023,
CBP 2023 and NCES 2023-24 into the same processed inputs GREASYPOP's census.py makes.
GeoPops now generates populations in a single-process Python port, too slow for the
largest regions, so the households, schools, workplaces and memberships are built with
the Julia code from GeoPops' julia/ folder (GREASYPOP's code plus logging) and
FLUFFHEAD's synthpop_groups.jl / export_groups.jl, as for the 2019 networks.

LODES8 has no 2023 files for Alaska (latest 2016) or Michigan (2021); those states use their
latest year, fetched here and linked under the 2023 names GeoPops reads (lodes_year()).

The 2023 tables B09019-B09021 (household/group-quarters splits by age) aren't published
for block groups any more; GeoPops takes them from the 2022 ACS.

Layout:
    data/raw/geopops2023/    national store: census/<ST>, pums, geo, work, school
    work/geopops2023/<group> working directory (links into the store)
    data/regions/<group>/network-2023/   the converted network

Region definitions (geo.json) use 2010-era counties. From 2022 the Census Bureau
replaced Connecticut's counties with planning regions, so Connecticut counties are
translated to their 2020 tracts under the new codes (ct_tracts()).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .greasypop import ABBR, ENV_DIR, INTERMEDIATES, NO_LODES
from .regions import census_key

log = logging.getLogger(__name__)

YEAR = 2023
STORE = C.RAW_DIR / f"geopops{YEAR}"
TAG = os.environ.get("GEOPOPS_TAG", "")        # e.g. "-new": a side-by-side test build in its own folders
WORK = C.WORK_DIR / f"geopops{YEAR}{TAG}"
PY = C.REPO / ".venv-geopops" / "bin" / "python"
DRIVER = Path(__file__).resolve().parent / "geopops_env" / "driver.py"
PATCHED_JULIA = Path(__file__).resolve().parent / "geopops_env" / "julia"
JULIA_FILES = ["CO.jl", "synthpop.jl", "households.jl", "schools.jl", "workplaces.jl", "netw.jl",
               "utils.jl", "fileutils.jl"]
NATIONAL_STEPS = ["ct_puma_crosswalk", "geocorr", "cbp", "school"]
STATE_STEPS = ["census", "pums", "shapefiles", "lodes"]
STEPS = ["process", "co", "synthpop", "export", "convert"]
GAZ = {2023: "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_tracts_national.zip",
       2020: "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_tracts_national.zip"}


def network_dir(group: str) -> Path:
    return C.REGIONS_DIR / group / f"network-{YEAR}{TAG}"


def julia_dir() -> Path:
    # find_spec locates the package without importing it: a first import on a new machine
    # rebuilds matplotlib's font cache and prints a notice to stdout
    code = "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('geopops').origin))"
    out = subprocess.run([str(PY), "-c", code], check=True, capture_output=True, text=True).stdout
    return Path(out.strip().splitlines()[-1]) / "julia"


# ------------------------------------------------------------------- store
LODES_URL = "https://lehd.ces.census.gov/data/lodes/LODES8/{st}/{sub}/{st}_{part}_JT01_{year}.csv.gz"
LODES_PARTS = [("od", "od_main"), ("od", "od_aux"), ("wac", "wac_S000")]
LODES_LATE = {"02": 2016, "26": 2021}     # states whose LODES8 stops before YEAR (checked 2026-11-12)


def lodes_year(st: str) -> int:
    return LODES_LATE.get(st, YEAR)


def _lodes_file(st: str, part: str, year: int | None = None) -> Path:
    return STORE / "work" / f"{ABBR[st].lower()}_{part}_JT01_{year or lodes_year(st)}.csv.gz"


def fetch_late_lodes(st: str) -> None:
    from .greasypop import _fetch_file
    y = lodes_year(st)
    log.warning("LODES for %s uses %d (%d unavailable)", ABBR[st], y, YEAR)
    for sub, part in LODES_PARTS:
        dest = _lodes_file(st, part)
        if not dest.exists():
            _fetch_file(LODES_URL.format(st=ABBR[st].lower(), sub=sub, part=part, year=y), dest)


def downloaded(st: str) -> bool:
    a = ABBR[st]
    census = STORE / "census" / a
    ok = (census.is_dir() and len(list(census.glob("ACSDT5Y*.csv"))) >= 17
          and (STORE / "pums" / f"psam_p{st}.csv").exists()
          and (STORE / "geo" / f"tl_{YEAR}_{st}_bg.zip").exists())
    return ok and (st in NO_LODES or _lodes_file(st, "wac_S000").exists())


def _national_done() -> bool:
    return ((STORE / "work" / f"cbp{YEAR - 2000}co.zip").exists()
            and (STORE / "geo" / "2020_Census_Tract_to_2020_PUMA.txt").exists()
            and any((STORE / "school").glob("EDGE_GEOCODE_PUBLICSCH_*.xlsx")))


def _driver(stage: str, path: Path, spec: dict, log_path: Path, env: dict | None = None) -> None:
    spec_path = path / f"spec-{stage}.json"
    path.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(dict(spec, main_year=YEAR), indent=1))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as out:
        subprocess.run([str(PY), "-u", str(DRIVER), stage, str(path), str(spec_path)], env=env or dict(os.environ),
                       check=True, stdout=out, stderr=subprocess.STDOUT)


def download(states: list[str], log_path: Path | None = None, batch: int = 8) -> None:
    """Fetch GeoPops inputs for these states into the national store (skips what is there)."""
    log_path = log_path or C.REPO / "logs" / f"download-geopops{YEAR}.log"
    env = dict(os.environ, CENSUS_API_KEY=census_key())        # passed to the driver in memory only
    if not _national_done():
        _driver("download", STORE, {"geos": ["56"], "commute_states": [], "use_pums": [],
                                    "steps": NATIONAL_STEPS}, log_path, env)
    todo = [s for s in states if not downloaded(s)]
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        log.info("GeoPops %d inputs for %s", YEAR, " ".join(ABBR[s] for s in part))
        late = [s for s in part if s in LODES_LATE]
        on_time = [s for s in part if s not in LODES_LATE and s not in NO_LODES]
        if late:              # GeoPops fetches LODES for YEAR only
            _driver("download", STORE, {"geos": late, "commute_states": [], "use_pums": late,
                                        "steps": [x for x in STATE_STEPS if x != "lodes"]}, log_path, env)
            for s in late:
                fetch_late_lodes(s)
        if on_time:
            _driver("download", STORE, {"geos": on_time, "commute_states": on_time, "use_pums": on_time,
                                        "steps": STATE_STEPS}, log_path, env)
        rest = [s for s in part if s in NO_LODES and s not in late]
        if rest:
            _driver("download", STORE, {"geos": rest, "commute_states": [], "use_pums": rest,
                                        "steps": [x for x in STATE_STEPS if x != "lodes"]}, log_path, env)
    for s in states:
        if downloaded(s):
            patch_acs_gaps(s, env["CENSUS_API_KEY"])


# ------------------------------------------------------------ ACS gaps
ACS_FILL_YEAR = YEAR - 1       # the year GeoPops already takes B09019-B09021 from


def acs_gaps(st: str) -> set[str]:
    """Block groups in the 2022 tables (B09019) but missing from the 2023 ones. The 2019-2023 ACS
    leaves out 15 New York tracts (40 block groups, ~80k people: Suffolk and Ulster), which makes
    GeoPops fail; checked 2026-11-14, NY is the only state with a gap."""
    d = STORE / "census" / ABBR[st]
    ids = lambda f: set(pd.read_csv(f, usecols=[0], dtype=str).iloc[1:, 0])
    return ids(d / f"ACSDT5Y{ACS_FILL_YEAR}.B09019-Data.csv") - ids(d / f"ACSDT5Y{YEAR}.B11012-Data.csv")


def patch_acs_gaps(st: str, key: str) -> None:
    """store/acs-patched/<ST>/: the state's YEAR tables with the gap block groups filled from the
    ACS_FILL_YEAR 5-year ACS (same variables), so every block group has targets."""
    import time
    import requests
    gaps = acs_gaps(st)
    out = STORE / "acs-patched" / ABBR[st]
    if not gaps or out.exists():
        return
    log.warning("%s: %d block groups missing from the %d ACS; filled from %d", ABBR[st], len(gaps), YEAR, ACS_FILL_YEAR)
    counties = sorted({g[11:14] for g in gaps})
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in sorted((STORE / "census" / ABBR[st]).glob(f"ACSDT5Y{YEAR}.*-Data.csv")):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        cols = [c for c in df.columns if c not in ("GEO_ID", "NAME")]
        got = []
        for c in counties:
            parts = []
            for i in range(0, len(cols), 45):         # the API takes at most 50 variables a call
                r = requests.get(f"https://api.census.gov/data/{ACS_FILL_YEAR}/acs/acs5",
                                 params={"get": ",".join(["GEO_ID", "NAME"] + cols[i:i + 45]), "for": "block group:*",
                                         "in": f"state:{st} county:{c}", "key": key},
                                 headers={"User-Agent": "FLUFFHEAD influenza forecasting (research)"}, timeout=120)
                r.raise_for_status()
                rows = r.json()
                parts.append(pd.DataFrame(rows[1:], columns=rows[0]).set_index("GEO_ID")[["NAME"] + cols[i:i + 45]]
                             if i == 0 else pd.DataFrame(rows[1:], columns=rows[0]).set_index("GEO_ID")[cols[i:i + 45]])
                time.sleep(0.2)
            got.append(pd.concat(parts, axis=1))
        fill = pd.concat(got).loc[sorted(gaps)].reset_index()[df.columns]
        pd.concat([df, fill], ignore_index=True).to_csv(tmp / f.name, index=False)
    tmp.rename(out)


# -------------------------------------------------------------- connecticut
def _gazetteer(year: int) -> pd.DataFrame:
    path = C.RAW_DIR / "geo" / f"{year}_Gaz_tracts_national.txt"
    if not path.exists():
        import io
        import zipfile
        from .weather import _get
        zipfile.ZipFile(io.BytesIO(_get(GAZ[year], headers={"User-Agent": "FLUFFHEAD influenza forecasting (research)"}))
                        ).extractall(path.parent)
    g = pd.read_csv(path, sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    g.columns = g.columns.str.strip()
    return g


def ct_tracts() -> pd.DataFrame:
    """Connecticut's 2020 tracts: 2010-era county (2020 Census codes) -> 2022+ GEOID (planning regions).

    Tract codes were kept when the counties were recoded; a code that appears twice is
    matched to the nearest internal point."""
    path = STORE / "ct_tracts.csv"
    if path.exists():
        return pd.read_csv(path, dtype=str)
    new = _gazetteer(YEAR); new = new[new.GEOID.str[:2] == "09"]
    old = _gazetteer(2020); old = old[old.GEOID.str[:2] == "09"]
    m = new.assign(code=new.GEOID.str[5:]).merge(old.assign(code=old.GEOID.str[5:]), on="code", suffixes=("", "_old"))
    m["d"] = np.hypot(m.INTPTLAT - m.INTPTLAT_old, m.INTPTLONG - m.INTPTLONG_old)
    m = m.sort_values("d").drop_duplicates("GEOID")
    if len(m) != len(new):
        raise ValueError(f"{len(new) - len(m)} Connecticut tracts have no 2020 match")
    out = pd.DataFrame({"county": m.GEOID_old.str[:5], "geoid": m.GEOID}).sort_values("geoid")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def recode_ct(codes: pd.Series) -> pd.Series:
    """LODES8 keeps Connecticut's old county codes (0900x); swap each tract prefix for its
    planning-region GEOID (091xx), which the ACS, TIGER and CBP 2023 files use."""
    x = ct_tracts()
    m = dict(zip(x.county + x.geoid.str[5:], x.geoid))   # only water tract 990000 repeats
    ct = codes.str[:2] == "09"
    out = codes.copy()
    out[ct] = codes[ct].str[:11].map(m).fillna(codes[ct].str[:11]) + codes[ct].str[11:]
    return out


def _ct_copy(src: Path, sub: str, fix) -> Path:
    """Cached copy of src with fix(df) applied (store/ct-recoded/<sub>/); the original is kept."""
    dest = STORE / "ct-recoded" / sub / src.name
    if not dest.exists():
        df = fix(pd.read_csv(src, dtype=str, keep_default_na=False, encoding="utf-8-sig"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")   # parallel builds may race; rename is atomic
        df.to_csv(tmp, index=False, compression="gzip" if src.suffix == ".gz" else None)
        tmp.rename(dest)
    return dest


def lodes_ct(src: Path) -> Path:
    """LODES8 file with Connecticut geocodes recoded."""
    def fix(df):
        for col in ["w_geocode", "h_geocode"]:
            if col in df:
                df[col] = recode_ct(df[col])
        return df
    return _ct_copy(src, "work", fix)


def tract_puma_ct(src: Path) -> Path:
    """2020 tract -> PUMA file (its CT rows have the 2022 PUMAs but old county codes)."""
    def fix(df):
        df = df.rename(columns={df.columns[0]: "STATEFP"})   # its header has a garbled BOM, as GeoPops notes
        ct = df.STATEFP == "09"
        new = recode_ct("09" + df.COUNTYFP[ct] + df.TRACTCE[ct])
        df.loc[ct, "COUNTYFP"] = new.str[2:5]
        return df
    return _ct_copy(src, "geo", fix)


def decennial_ct(src: Path) -> Path:
    """2020 decennial table (row 0 holds labels; GEO_ID is 1500000US + block group)."""
    def fix(df):
        g = df.GEO_ID.iloc[1:]
        df.loc[g.index, "GEO_ID"] = g.str[:9] + recode_ct(g.str[9:])
        return df
    return _ct_copy(src, "census", fix)


def pums_st(src: Path) -> Path:
    """PUMS file with ST padded to 2 digits. The 2023 files have ST=9 for 09 (and 1..8); GeoPops
    pads PUMA but not ST, so st_puma ("9"+PUMA) never matches the crosswalks ("09"+PUMA) and the
    CO fit finds no samples. Streamed (CA's person file is large); other fields are untouched."""
    import csv
    dest = STORE / "pums-st" / src.name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
        with open(src, newline="", encoding="latin-1") as fi, open(tmp, "w", newline="", encoding="latin-1") as fo:
            r, w = csv.reader(fi), csv.writer(fo, lineterminator="\n")
            head = next(r)
            w.writerow(head)
            i = head.index("ST")
            for row in r:
                row[i] = row[i].zfill(2)
                w.writerow(row)
        tmp.rename(dest)
    return dest


def spec(group: str) -> dict:
    """The group's geo.json in 2023 geography (Connecticut counties -> 2020 tracts)."""
    geo = json.loads((C.REGIONS_DIR / group / "geo.json").read_text())
    geos = [g for g in geo["geos"] if not (len(g) >= 5 and g[:2] == "09")]
    ct = [g for g in geo["geos"] if len(g) >= 5 and g[:2] == "09"]
    if ct:
        x = ct_tracts()
        geos += sorted(x.geoid[x.county.isin(ct)])
    return {"geos": geos, "commute_states": [s for s in geo["commute_states"] if s not in NO_LODES],
            "use_pums": geo["use_pums"]}


# ------------------------------------------------------------------ prepare
def _link(src: Path, dest: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(src.resolve())


def prepare(group: str) -> Path:
    """work/geopops2023/<group>/ with the Julia code and links to the inputs this group needs."""
    sp = spec(group)
    main = sorted({g[:2] for g in sp["geos"]})
    wd = WORK / group
    for sub in ["census", "pums", "geo", "work", "school"]:
        if (wd / sub).exists():
            shutil.rmtree(wd / sub)
        (wd / sub).mkdir(parents=True)
    jd = julia_dir()
    for f in JULIA_FILES:
        shutil.copy2(jd / f, wd / f)
    if not os.environ.get("GEOPOPS_STOCK"):    # faster, same-output versions (geopops_env/julia/)
        for f in sorted(PATCHED_JULIA.glob("*.jl")):
            shutil.copy2(f, wd / f.name)
    for f in ["synthpop_groups.jl", "export_groups.jl"]:
        shutil.copy2(ENV_DIR / f, wd / f)
    (wd / "spec.json").write_text(json.dumps(sp, indent=1) + "\n")

    # Connecticut: ACS, TIGER, CBP and geocorr use planning regions; LODES8, the 2020 decennial
    # and the tract -> PUMA file keep the old counties, so those get recoded copies
    for st in main:                       # process_data reads every census/<ST> folder present
        src = STORE / "census" / ABBR[st]
        patched = STORE / "acs-patched" / ABBR[st]          # ACS gaps filled (patch_acs_gaps)
        if st == "09" or patched.is_dir():
            (wd / "census" / ABBR[st]).mkdir()
            for f in src.iterdir():
                g = decennial_ct(f) if st == "09" and f.name.startswith("DECENNIAL") else f
                g = patched / f.name if (patched / f.name).exists() else g
                _link(g, wd / "census" / ABBR[st] / f.name)
        else:
            _link(src, wd / "census" / ABBR[st])
        for f in (STORE / "geo").glob(f"tl_{YEAR}_{st}_bg.*"):
            _link(f, wd / "geo" / f.name)
    for st in sp["use_pums"]:
        for kind in "hp":
            src = STORE / "pums" / f"psam_{kind}{st}.csv"
            _link(pums_st(src) if st < "10" else src, wd / "pums" / src.name)
    for f in (STORE / "geo").iterdir():
        if f.is_file() and not f.name.startswith("tl_"):
            ct_puma = "09" in main and "Tract_to" in f.name
            _link(tract_puma_ct(f) if ct_puma else f, wd / "geo" / f.name)
    _link(STORE / "work" / f"cbp{YEAR - 2000}co.zip", wd / "work" / f"cbp{YEAR - 2000}co.zip")
    lodes = lodes_ct if "09" in main else (lambda p: p)   # CT geos: recode CT geocodes to match
    for st in main:
        if st in NO_LODES:
            continue
        a = ABBR[st].lower()
        for part in ["od_main", "wac_S000"]:      # a late state's file under the name GeoPops reads
            _link(lodes(_lodes_file(st, part)), wd / "work" / f"{a}_{part}_JT01_{YEAR}.csv.gz")
    for st in sorted(set(main) | set(sp["commute_states"])):
        if st in NO_LODES:
            continue
        a = ABBR[st].lower()
        _link(lodes(_lodes_file(st, "od_aux")), wd / "work" / f"{a}_od_aux_JT01_{YEAR}.csv.gz")
    for f in (STORE / "school").iterdir():
        _link(f, wd / "school" / f.name)
    return wd


# --------------------------------------------------------------------- run
def fix_targets(wd: Path) -> None:
    """Zero the industry/occupation targets GeoPops leaves NaN in block groups with no workers.

    In such a block group generate_gq can give a tiny negative GQ worker count, so the household
    share is inf, IPF returns NaN and CO.jl stops (seen in Lee County FL, 120710901001). Any other
    NaN target is still an error."""
    p = wd / "processed" / "acs_targets.csv"
    a = pd.read_csv(p, index_col=0, dtype={0: str})
    bad = a.index[a.isna().any(axis=1)]
    if not len(bad):
        return
    b = pd.read_csv(wd / "processed" / "df_bounds_test.csv", index_col=0, dtype={0: str})
    cols = [c for c in a.columns if c.startswith(("C24030:", "C24010:"))]
    no_workers = b.total_workers.reindex(bad).fillna(-1) == 0
    other = a.loc[bad, [c for c in a.columns if c not in cols]].isna().any(axis=1)
    if not no_workers.all() or other.any():
        raise ValueError(f"NaN targets in {list(bad[~no_workers.to_numpy() | other.to_numpy()])}")
    log.warning("%s: zeroed NaN industry/occupation targets in %d block group(s) with no workers: %s",
                wd.name, len(bad), " ".join(bad))
    a.loc[bad, cols] = a.loc[bad, cols].fillna(0)
    a.to_csv(p)


def run(group: str, steps: list[str] | None = None, procs: int = 4, keep: bool = False) -> Path:
    """Run the build steps for a group; logs go to work/geopops2023/<group>/build.log."""
    from . import network

    wd = WORK / group
    steps = steps or STEPS
    env = dict(os.environ, JULIA_PROJECT=str(ENV_DIR))
    env.pop("CENSUS_API_KEY", None)       # processing reads only local files
    cmds = {"co": ["julia", f"-p{procs}", "CO.jl"],
            "synthpop": ["julia", f"-t{procs}", "synthpop_groups.jl"],   # threads: commute matrices
            "export": ["julia", "export_groups.jl"]}
    out_dir = network_dir(group)
    blog = wd / "build.log"
    for s in steps:
        log.info("%s: %s", group, s)
        with open(blog, "a") as out:
            out.write(f"\n===== {s}\n")
        if s == "process":
            shutil.rmtree(wd / "processed", ignore_errors=True)
            _driver("process", wd, json.loads((wd / "spec.json").read_text()), blog, env)
            fix_targets(wd)
        elif s == "convert":
            network.convert_export(wd / "groups_export", out_dir, vintage=YEAR)
            if not keep:
                for d in INTERMEDIATES:
                    shutil.rmtree(wd / d, ignore_errors=True)
        else:
            with open(blog, "a") as out:
                subprocess.run(cmds[s], cwd=wd, env=env, check=True, stdout=out, stderr=subprocess.STDOUT)
    return out_dir


def build(group: str, procs: int = 4, keep: bool = False) -> Path:
    """Download (if needed), prepare and build one group's 2023 network."""
    sp = spec(group)
    states = sorted({g[:2] for g in sp["geos"]} | set(sp["commute_states"]) | set(sp["use_pums"]))
    download(states)
    prepare(group)
    return run(group, procs=procs, keep=keep)
