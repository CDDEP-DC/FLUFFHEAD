# FLUFFHEAD

**FLU Forecasting Framework for Heterogeneous Edge-Activated Dynamics**: a network agent-based model that produces weekly influenza hospitalization forecasts for the [FluSight hub](https://github.com/cdcepi/FluSight-forecast-hub), as model `OHTJHU-FLUFFHEAD`.

It simulates individual people in 116 US regions, about 265 million residents and 81% of the population, on synthetic populations built from 2023 Census data with [GeoPops](https://github.com/GeoPopsHub/geopops). Set `FLUFFHEAD_NETWORK_VINTAGE=2019` to use the earlier networks, built from 2019 data with [GREASYPOP-CO](https://github.com/CDDEP-DC/GREASYPOP-CO).

> **Status:** in development; submitting to the hub weekly since reference date 2026-11-07. See [Submissions](#submissions).

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

Networks and raw data aren't in this repo and must be built once, which takes many hours. That step also needs Julia, GeoPops in its own Python environment (`.venv-geopops/`), and a [Census API key](https://api.census.gov/data/key_signup.html) in the `CENSUS_API_KEY` environment variable. Don't commit the key.

```bash
python -m fluffhead regions     # choose simulation groups
python -m fluffhead networks --vintage 2023 --procs 8   # download GeoPops inputs and build each group's network (resumable)
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

To download the day's station humidity for every region ahead of a run, use `python -m fluffhead weather`; it goes to `cache/weather/`.

Finished regions are skipped, so an interrupted run resumes. Move `runs/<date>/` aside to start a date over.

**Momentum alternate (trial).** Each run also writes a second forecast from the same samples. It shifts each state's forecast by how far its flu ED visits lead its admissions (`fluffhead/momentum.py`). In backtests, this cut the error at peaks and declines by about a quarter. The file is `forecasts/alternates/momentum/<date>-OHTJHU-FLUFFHEAD.csv` and the review report is `forecasts/reports/<date>-momentum-report.html`, which also shows the main median for comparison. A state is corrected only when its ED data include the last reported week and it had at least 20 admissions. NSSP posts on Wednesday mornings, so a run before then corrects nothing. In that case, rerun just the alternate afterwards (it takes seconds):

```bash
python -m fluffhead alternate --reference-date 2026-11-21
```

To submit the alternate instead of the main forecast, first copy the main file to `forecasts/archive/<date>-main-OHTJHU-FLUFFHEAD.csv`, then copy the alternate over the one in `forecasts/model-output/`. The trial log is `forecasts/alternates/momentum/TRIAL.md`.

## Submissions

Files in `forecasts/model-output/OHTJHU-FLUFFHEAD/` for these dates are the ones submitted to the hub.

| Reference date | Submitted | Hub pull request |
|---|---|---|
| 2026-11-07 | main | [#4003](https://github.com/cdcepi/FluSight-forecast-hub/pull/4003) |
| 2026-11-14 | main | [#4101](https://github.com/cdcepi/FluSight-forecast-hub/pull/4101) |
| 2026-11-21 | momentum alternate (main version in `forecasts/archive/`) | [#4154](https://github.com/cdcepi/FluSight-forecast-hub/pull/4154) |

The 2026-10-31 file was not submitted.

## Layout

| Path | Contents |
|---|---|
| `fluffhead/` | The model and pipeline |
| `tests/` | Unit tests |
| `forecasts/` | Submission files, reports, archived versions, the momentum alternate, hub metadata |
| `data/regions/` | Simulation groups and their geography (networks are built separately) |
| `data/momentum/` | Coefficients and training data for the momentum alternate |
| `data/state/` | Archived state data |

## License

Code is under the [MIT License](LICENSE). Forecasts are released under CC-BY-4.0.
