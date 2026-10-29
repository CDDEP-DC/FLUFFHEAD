"""Command line: `fluffhead <command>` (or `python -m fluffhead <command>`).

Setup (once, or when region rules change):
    fluffhead regions        choose simulation groups -> data/regions/
    fluffhead fetch          download GREASYPOP inputs for every state the groups need
    fluffhead networks       build each group's network with GREASYPOP (resumable)

Weekly:
    fluffhead forecast       nowcast, fit and forecast every built group, write the submission
    fluffhead report         re-render a week's review report from its saved data
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time

import pandas as pd

from . import config as C

log = logging.getLogger("fluffhead")


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _states(group: str) -> list[str]:
    geo = json.loads((C.REGIONS_DIR / group / "geo.json").read_text())
    return sorted({g[:2] for g in geo["geos"]} | set(geo["commute_states"]) | set(geo["use_pums"]))


def cmd_regions(args) -> int:
    from . import regions
    g = regions.build()
    cv = regions.coverage()
    print(f"{len(g)} groups, {cv.simulated.sum() / cv['pop'].sum():.1%} of the population simulated")
    return 0


def cmd_fetch(args) -> int:
    from . import greasypop as G
    groups = pd.read_csv(C.REGIONS_DIR / "groups.csv").group
    states = sorted({s for g in groups for s in _states(g)})
    G.download(states, C.REPO / "logs" / "download.log")
    missing = [s for s in states if not G.downloaded(s)]
    print("missing:", " ".join(missing) or "none")
    return 1 if missing else 0


def cmd_networks(args) -> int:
    """Build anchor groups (every state's largest) first, then the rest, smallest first within each.

    With --wait, groups whose states aren't downloaded yet are retried.
    """
    from . import greasypop as G
    from .regions import anchor_groups
    groups = pd.read_csv(C.REGIONS_DIR / "groups.csv")
    anchors = set(anchor_groups())
    groups = groups.assign(anchor=~groups.group.isin(anchors)).sort_values(["anchor", "pop"])
    todo = [g for g in (args.groups or groups.group)
            if not (C.REGIONS_DIR / g / "network" / "network.npz").exists() or args.force]
    failed = []
    while todo:
        ready = [g for g in todo if all(G.downloaded(s) for s in _states(g))]
        if not ready:
            if not args.wait:
                print("not downloaded yet:", " ".join(todo))
                break
            time.sleep(120)
            continue
        g = ready[0]
        todo.remove(g)
        t = time.time()
        try:
            G.prepare(g)
            G.run(g, procs=args.procs, keep=args.keep)
            log.info("built %s in %.1f min", g, (time.time() - t) / 60)
        except Exception as e:  # keep going; the log in work/<group>/ has details
            log.error("%s failed after %.1f min: %s", g, (time.time() - t) / 60, e)
            failed.append(g)
    if failed:
        print("failed:", " ".join(failed))
    return 1 if failed else 0


def cmd_forecast(args) -> int:
    from . import pipeline as P
    ref = args.reference_date or P.reference_date_for(dt.date.today())
    wk, df, noisy, weeks, problems, path = P.run(ref, groups=args.groups, workers=args.workers,
                                                 n_draws=args.draws, n_keep=args.keep, n_cont=args.continuations,
                                                 nowcast=not args.no_nowcast)
    print(f"submission: {path}")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


def cmd_report(args) -> int:
    """Re-render a report from its saved data (e.g. after editing the template)."""
    from . import report
    from . import pipeline as P
    ref = args.reference_date or P.reference_date_for(dt.date.today())
    d = C.FORECAST_DIR / "reports"
    payload = json.loads((d / f"{ref}-data.json").read_text())
    print(report.render(payload, d / f"{ref}-report.html"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fluffhead", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("regions").set_defaults(fn=cmd_regions)
    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    p = sub.add_parser("networks")
    p.add_argument("groups", nargs="*")
    p.add_argument("--procs", type=int, default=4, help="Julia worker processes for CO.jl")
    p.add_argument("--wait", action="store_true", help="wait for downloads still in progress")
    p.add_argument("--keep", action="store_true", help="keep GREASYPOP intermediates")
    p.add_argument("--force", action="store_true", help="rebuild groups that already have a network")
    p.set_defaults(fn=cmd_networks)
    p = sub.add_parser("forecast")
    p.add_argument("--reference-date", type=_date)
    p.add_argument("--groups", nargs="*")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--draws", type=int, default=300)
    p.add_argument("--keep", type=int, default=20)
    p.add_argument("--continuations", type=int, default=5)
    p.add_argument("--no-nowcast", action="store_true")
    p.set_defaults(fn=cmd_forecast)
    p = sub.add_parser("report")
    p.add_argument("--reference-date", type=_date)
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
