# Age Is Not Habitat Persistence

Code and summary results for:

> Gabriela Morales Soto, Sandra Avila, Helio Pedrini. **Age is not Habitat Persistence: Forecasting Forest-Habitat Support to Prioritize Biodiversity Resurveys in the Atlantic Forest.** Tackling Climate Change with Machine Learning workshop at NeurIPS 2026 (poster).

The paper asks whether the site of a historical plant record still holds forest habitat, and which sites are likely to lose it next. It links GBIF occurrences to annual MapBiomas land-cover trajectories, propagates coordinate uncertainty by Monte Carlo sampling, and evaluates five-year habitat-loss forecasts (LightGBM and logistic regression) with spatially blocked, temporally frozen validation and a fixed field-budget decision metric.

## Repository layout

```
scripts/     analysis pipeline (run in the order below)
results/     summary tables behind every number reported in the paper
figures/     final figures and the data behind them
data/        inputs (only the 5 km grid is versioned; see "Data")
```

Script comments and diagnostic logs are written in Spanish; each script starts with a block explaining what it does and why.

## Data

| Input | Source | Location |
|---|---|---|
| Plant occurrences | GBIF.org (7 May 2026) GBIF Occurrence Download, <https://doi.org/10.15468/dl.evgrnx>. Tracheophyta, Brazil, 1970–2026, georeferenced, no geospatial issues (4,455,560 records) | `data/0014792-260430073515954.zip` (not versioned, 637 MB) |
| Land cover | MapBiomas Collection 10, Earth Engine asset `projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2`, accessed 2026-08-14. Forest classes 3, 4, 5, 49 | queried through Earth Engine |
| Atlantic Forest boundary | biome boundary polygon | `data/mata_atlantica.gpkg` (not versioned) |
| 5 km grid | derived from the biome boundary, EPSG:5880 | `data/grid_5km.gpkg` |

**Study extent.** The processed cohort is restricted to 19.5–25.7°S, 44.0–53.7°W (plus records labelled São Paulo state) and intersected with the Atlantic Forest, i.e. the south-central part of the biome. The box is defined in `scripts/build_tropical_mlm_sequences.py` (`LAT_MIN`, `LAT_MAX`, `LON_MIN`, `LON_MAX`). The resulting cohort has 491,228 records at 135,021 coordinates from 1985 onward.

## Pipeline

Earth Engine steps (marked GEE) need an authenticated `earthengine-api` and a Google Cloud project: pass it with `--gee-project YOUR-PROJECT-ID`.

| Step | Script | Produces | Paper |
|---|---|---|---|
| 0 | `build_tropical_mlm_sequences.py` | `data/processed/occurrences_with_cell_id.csv` (cohort) | Data and Methods |
| 1 | `01e_record_obsolescence_histogram.py` (GEE) | per-record loss histograms at 0/100/250/500 m | |
| 2 | `01f_buffer_sensitivity_common_cohort.py`, `01g_buffer_sensitivity_figure.py` | binary vs. continuous range factors | Fig. 1a |
| 3 | `20_trajectory_extract.py` (GEE) | annual forest trajectories 1985–2024, bit-packed per pixel | |
| 4 | `21_build_survival_dataset.py` | survival dataset, 20 positional Monte Carlo draws | |
| 5 | `22_survival_models.py`, `22b_…`, `22c_…` | Rayleigh σ̂, Kaplan–Meier, IBS | Fig. 1b, Appendix C |
| 6 | `24_landmark_forecast.py`, `24b_landmark_rolling_origin_extract.py` | landmark forecasts, frozen 2014 test, rolling-origin cuts | Table 1, Appendix B |
| 7 | `25_resurvey_decision.py`, `25b_…`, `25c_…`, `29_decision_lift_bootstrap.py` | budgeted ranking, Lift@B, paired bootstrap, prospective 2024 ranking | Table 1, Fig. 1c, Appendix C |
| 8 | `26b_prospective_map_appendix_figure.py` | prospective risk map | Fig. 2 |
| 9 | `27_permutation_importance.py` | spatial-block permutation importance | Appendix A |
| 10 | `28_forest_class_composition.py` (GEE) | forest-class composition and conversion | Results, Appendix C |

Example for the GEE steps:

```bash
python scripts/01e_record_obsolescence_histogram.py \
  --occurrences-csv data/processed/occurrences_with_cell_id.csv \
  --checkpoint-dir results/record_level_histogram/chunks \
  --output-dir results/record_level_histogram \
  --gee-project YOUR-PROJECT-ID

python scripts/20_trajectory_extract.py \
  --occurrences-csv data/processed/occurrences_with_cell_id.csv \
  --checkpoint-dir results/trajectory/chunks \
  --output-dir results/trajectory \
  --gee-project YOUR-PROJECT-ID
```

The remaining scripts read and write under `data/`, `results/` and `figures/` by default; run them from the repository root with `python scripts/<name>.py` (use `--help` for options).

Fixed settings: seed 42, M = 20 positional draws, 250 m main buffer (500 m sensitivity), k = 2 consecutive non-forest years, five-year horizon, training landmarks 1993/1998/2003/2008, frozen test landmark 2014, 117 spatial blocks of 100 km, LightGBM `num_leaves=7, learning_rate=0.05, n_estimators=150, min_child_samples=50`.

## Results included

`results/` contains the summary outputs behind the paper's numbers: buffer sensitivity (`record_level_histogram/`), survival calibration (`calibration/`), landmark metrics and paired bootstrap differences (`landmark/`), decision and Lift tables plus the prospective 2024→2029 ranking (`decision/`), and permutation importance and forest-class composition (`interpretation/`). Large intermediates (trajectories, survival datasets, row-level predictions) are not versioned and are regenerated by the pipeline.

The prospective ranking (`*_NO_OBSERVADO*`) is a forward projection with no observed outcome yet; it is not an independent validation.

## Requirements

Python 3.11. See `requirements.txt`.

## Citation

```bibtex
@inproceedings{moralessoto2026age,
  title     = {Age is not Habitat Persistence: Forecasting Forest-Habitat Support to Prioritize Biodiversity Resurveys in the Atlantic Forest},
  author    = {Morales Soto, Gabriela and Avila, Sandra and Pedrini, Helio},
  booktitle = {Tackling Climate Change with Machine Learning Workshop at NeurIPS},
  year      = {2026}
}
```

## License

Code: MIT (see `LICENSE`). Paper: CC BY 4.0. GBIF and MapBiomas data remain under their own licences.
