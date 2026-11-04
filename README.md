# FLUFFHEAD

**FLU Forecasting Framework for Heterogeneous Edge-Activated Dynamics**: a network agent-based model that produces weekly influenza hospitalization forecasts for the [FluSight hub](https://github.com/cdcepi/FluSight-forecast-hub), as model `OHTJHU-FLUFFHEAD`.

It simulates individual people in 116 US regions, about 265 million residents and 81% of the population, on synthetic populations built with [GREASYPOP-CO](https://github.com/CDDEP-DC/GREASYPOP-CO).

> **Status:** in development. The forecasts in `forecasts/` are working versions and have not been submitted to the hub.

## How it works

1. **Data.** Hub target data, with the last two weeks nowcast for backfill using a revision model learned from past versions of that data, plus NSSP emergency department visits and state health department data.
2. **Fit.** Each region is fitted to its share of its states' weekly admissions by adaptive importance sampling. Fitted parameters are the reproduction number at the cutoff, overall prior immunity, the vaccine's effect on infection, the travel share of infection risk and the humidity floor.
3. **Forecast.** Kept draws run to the end of the season, with humidity from the NWS 7-day forecast, then climatology plus historical anomalies.
4. **National pass.** Regional forecasts are summed into a national curve, and every region is re-run with its imports following that curve.
5. **Combine.** Regions are joined into 2,000 joint samples, then scaled to states and the US, to produce the submission file and a review report.

An optional two-strain engine simulates a variant alongside the dominant strain (see `fluffhead/strains.py`).

## Outputs

| Target | Horizons | Format |
|---|---|---|
| `wk inc flu hosp` | -1 to 3 | quantiles |
| `wk flu hosp rate change` | 0 to 3 | category probabilities |
| `peak week inc flu hosp` | season | probabilities by week |
| `peak inc flu hosp` | season | quantiles |

## Setup

Requires Python 3.11 or later.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q tests
```

Networks and raw data aren't in this repo and must be built once, which takes many hours. That step also needs R, Julia, a local copy of GREASYPOP-CO, and a [Census API key](https://api.census.gov/data/key_signup.html) in the `CENSUS_API_KEY` environment variable. Don't commit the key.

```bash
python -m fluffhead regions     # choose simulation groups
python -m fluffhead fetch       # download GREASYPOP inputs
python -m fluffhead networks    # build each group's network (resumable)
```

## Weekly run

```bash
python -m fluffhead forecast --reference-date 2026-11-07 --workers 3 --draw-workers 3 --order barbell
```

Run this after the hub's target data updates on Wednesday. Output goes to `forecasts/model-output/` and `forecasts/reports/`; open the report and review it before submitting.

| Option | Use |
|---|---|
| `--workers`, `--draw-workers` | Regions at once, and processes per region |
| `--order barbell` | For small machines: the largest region runs alone, then one large region at a time beside small ones |
| `--draws` | Draws per region (default 300) |
| `--strains 2` | Add a variant; see `--variant-origin` and the other `--variant-*` options |
| `--one-pass` | Skip the national pass |

Finished regions are skipped, so an interrupted run resumes. Move `runs/<date>/` aside to start a date over.

## Layout

| Path | Contents |
|---|---|
| `fluffhead/` | The model and pipeline |
| `tests/` | Unit tests |
| `forecasts/` | Submission files, reports, archived versions, draft hub metadata |
| `data/state/` | Archived state data |

## License

Code is under the [MIT License](LICENSE). Forecasts are released under CC-BY-4.0.
