"""GeoPops download and processing for FLUFFHEAD (run with .venv-geopops/bin/python).

    driver.py download <store> <spec.json>    fetch inputs for the spec's states into the store
    driver.py process  <workdir> <spec.json>  write <workdir>/processed/ for one region

spec.json: {"main_year", "geos", "commute_states", "use_pums", "steps" (download only)}.
GeoPops is used as installed and never modified. The Census key is read from the
environment (CENSUS_API_KEY) into memory; every config written to disk has it set to
None, and this script checks that before it exits.
"""
import importlib
import json
import os
import pathlib
import sys

import geopops
from geopops.write_config import load_config, save_config

DD = importlib.import_module("geopops.download_data")


def _patched_process_data():
    """geopops.process_data with the faster prefix matching (geopops_env/process_data.py, a copy of
    GeoPops 095a5e0's with only those changes; proposed upstream). Loaded as the package's own
    module, so its relative imports resolve; GEOPOPS_STOCK=1 uses the installed one instead."""
    import importlib.util
    path = pathlib.Path(__file__).with_name("process_data.py")
    if os.environ.get("GEOPOPS_STOCK") or not path.exists():
        return importlib.import_module("geopops.process_data")
    spec = importlib.util.spec_from_file_location("geopops.process_data", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["geopops.process_data"] = mod
    spec.loader.exec_module(mod)
    return mod


PD = _patched_process_data()

# FLUFFHEAD's Julia build (utils.jl PersonData) expects these traits, in this order
TRAITS = ["sch_public", "sch_private", "female", "race_black_alone", "white_non_hispanic", "hispanic"]


def config(path, spec):
    # the package's shipped defaults; not load_config(), which reads the package's config.json,
    # a file the download stage rewrites (concurrent builds read it half-written)
    cfg = json.loads((pathlib.Path(geopops.__file__).parent / "config_defaults.json").read_text())
    year = int(spec["main_year"])
    cfg.update(path=str(path), main_year=year, decennial_year=2020 if year >= 2020 else 2010,
               geos=spec["geos"], commute_states=spec["commute_states"], use_pums=spec["use_pums"],
               additional_traits=TRAITS, census_api_key=None)
    return cfg


def main():
    stage, path, spec = sys.argv[1], pathlib.Path(sys.argv[2]), json.loads(pathlib.Path(sys.argv[3]).read_text())
    path.mkdir(parents=True, exist_ok=True)
    cfg = config(path, spec)
    # pull_census_data and geocorr_files read the package's own config.json (for main_year)
    # (download only: process_data takes cfg directly, and builds run many at once)
    written = [path / "config.json"]
    if stage == "download":
        written.insert(0, pathlib.Path(geopops.__file__).parent / "config.json")
    for p in written:
        save_config(cfg, str(p), sanitize=True)
    key = os.environ.get("CENSUS_API_KEY", "")
    if stage == "download":
        if not key:
            sys.exit("CENSUS_API_KEY is not set")
        DD.download_data(dict(cfg, census_api_key=key), steps=spec.get("steps"))
    elif stage == "process":
        if os.environ.get("GEOPOPS_SEED"):      # for comparing runs (employer sizes are random draws)
            import random
            import numpy as np
            random.seed(int(os.environ["GEOPOPS_SEED"]))
            np.random.seed(int(os.environ["GEOPOPS_SEED"]))
        PD.process_data(cfg)
    else:
        sys.exit(f"unknown stage {stage}")
    for p in written:
        if key and key in p.read_text():
            sys.exit(f"{p} holds the Census key")
    print("done", stage)


if __name__ == "__main__":
    main()
