# Age Is Not Habitat Persistence

Code and summary results for:

> Gabriela Morales Soto, Sandra Avila, Helio Pedrini. **Age is not Habitat Persistence: Forecasting Forest-Habitat Support to Prioritize Biodiversity Resurveys in the Atlantic Forest.** Tackling Climate Change with Machine Learning Workshop at NeurIPS 2026.

The paper asks whether the site of a historical plant record still holds forest habitat, and which sites are likely to lose it next. It links GBIF occurrences to annual MapBiomas land-cover trajectories, propagates coordinate uncertainty by Monte Carlo sampling, forecasts five-year habitat loss with LightGBM and logistic regression, and evaluates the ranking as a fixed field-budget decision.

## Validation protocol

Three different procedures are used, and they guarantee different things:

| Procedure | Where | What it guarantees |
|---|---|---|
| **Temporal rolling-origin validation + frozen temporal test** | forecast (`24_landmark_forecast.py`) | hyperparameters are selected on cuts 1993→1998, →2003, →2008; the 2014→2019 test is never used for model development, so every evaluation predicts a later period than training |
| **Spatial-block bootstrap** (117 blocks of 100 km, 1,000 resamples, paired across methods) | all forecast and decision metrics | uncertainty intervals account for spatial dependence between nearby localities |
| **Spatial cross-fitting** (5-fold `GroupKFold` over 100 km blocks) | Rayleigh / Kaplan–Meier calibration (`22_survival_models.py`) | the persistence curve is evaluated on spatially held-out blocks |

The forecast is **not** evaluated on spatially held-out localities: the same localities can appear in earlier training landmarks and in the 2014 test. Generalisation to unsampled regions is therefore not tested.

## Repository layout

```
scripts/     analysis pipeline (run in the order below)
results/     summary tables behind the numbers reported in the paper
figures/     final figures and the data behind them
data/        inputs; see data/README.md (only the 5 km grid is versioned)
```

Script comments and diagnostic logs are written in Spanish; each script starts with a block explaining what it does and why.

## Data

See [`data/README.md`](data/README.md) for how to obtain each input.

- **Occurrences:** GBIF.org (7 May 2026) GBIF Occurrence Download, <https://doi.org/10.15468/dl.evgrnx> (Tracheophyta, Brazil, 1970–2026, georeferenced, no geospatial issues; 4,455,560 records).
- **Land cover:** MapBiomas Collection 10, Earth Engine asset `projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2`, accessed 2026-08-14; forest classes 3, 4, 5, 49.

**Study extent.** The cohort is restricted to 19.5–25.7°S, 44.0–53.7°W (plus records labelled São Paulo state) and intersected with the Atlantic Forest, i.e. the south-central part of the biome. The box is defined in `scripts/build_tropical_mlm_sequences.py` (`LAT_MIN`, `LAT_MAX`, `LON_MIN`, `LON_MAX`). The resulting cohort has 491,228 records at 135,021 coordinates from 1985 onward.

## Pipeline

Run every script from the repository root as `python scripts/<name>.py`; use `--help` for options. Default paths read and write under `data/`, `results/` and `figures/`. Earth Engine steps (GEE) need an authenticated `earthengine-api` and a Google Cloud project passed with `--gee-project`.

| Step | Script | Main output |
|---|---|---|
| 0 | `build_tropical_mlm_sequences.py` (uses `gbif_archive_utils.py`) | `data/processed/occurrences_with_cell_id.csv` |
| 1 | `01e_record_obsolescence_histogram.py` (GEE) | `results/record_level_histogram/record_loss_derived_all_buffers.csv` |
| 2 | `01f_buffer_sensitivity_common_cohort.py` | `results/record_level_histogram/buffer_sensitivity_*.csv` |
| 3 | `01g_buffer_sensitivity_figure.py` | `figures/fig_buffer_sensitivity.pdf` |
| 4 | `20_trajectory_extract.py` (GEE) | `results/trajectory/` (annual forest trajectories, 1985–2024) |
| 5 | `21_build_survival_dataset.py` | `results/survival/` (20 positional Monte Carlo draws) |
| 6 | `22_survival_models.py` | `results/calibration/` (σ̂, IBS, horizon errors) |
| 7 | `22b_survival_curves_figure.py`, then `22c_survival_curves_figure_render.py` | `figures/fig_survival_curves.pdf` |
| 8 | `24_landmark_forecast.py` | `results/landmark/` (pooled metrics, paired bootstrap, predictions) |
| 9 | `24b_landmark_rolling_origin_extract.py` | `results/landmark/per_landmark_validation.csv` |
| 10 | `24d_test_metrics_per_draw.py` | `results/landmark/test_landmark_2014_metrics_per_draw.csv` |
| 11 | `25_resurvey_decision.py` | `results/decision/` (Lift@B, top-B support, prospective 2024 ranking) |
| 12 | `25b_cumulative_gain_locations_figure.py`, then `25c_cumulative_gain_figure_render.py` | `figures/fig_cumulative_gain.pdf` |
| 13 | `29_decision_lift_bootstrap.py` | `results/decision/lift_bootstrap_diff_boosting_vs_logistic_2014.csv` |
| 14 | `26b_prospective_map_appendix_figure.py` | `figures/fig_prospective_map_appendix.pdf` |
| 15 | `27_permutation_importance.py` | `results/interpretation/permutation_importance.csv` |
| 16 | `28_forest_class_composition.py` (GEE) | `results/interpretation/forest_class_composition.csv` |

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

Fixed settings: seed 42, M = 20 positional draws, 250 m main buffer (500 m sensitivity), k = 2 consecutive non-forest years, five-year horizon, training landmarks 1993/1998/2003/2008, frozen test landmark 2014, LightGBM `num_leaves=7, learning_rate=0.05, n_estimators=150, min_child_samples=50`. The prospective 2024→2029 ranking refits LightGBM on landmarks 1993–2018 with the same hyperparameters.

## Where each paper result comes from

| Paper item | File |
|---|---|
| Fig. 1a, binary vs. converted range factors | `results/record_level_histogram/buffer_sensitivity_range_factor*.csv` |
| Fig. 1b, σ̂ and IBS | `results/calibration/sigma_hat_k2_buffer250m.csv`, `ibs_summary_k2_buffer250m.csv` |
| Table 1, AUROC / AUPRC | `results/landmark/test_landmark_2014_metrics_per_draw.csv` (mean over draws) |
| Table 1, A_i and Lift@B; Fig. 1c | `results/decision/diagnostics_decision_2014.txt`, `figures/fig_cumulative_gain_data.csv` |
| ΔAUROC / ΔAUPRC, paired | `results/landmark/test_landmark_2014_bootstrap_diffs.csv` |
| ΔLift, paired | `results/decision/lift_bootstrap_diff_boosting_vs_logistic_2014.csv` |
| Fig. 2, prospective ranking | `results/decision/prospective_localidades_2024_NO_OBSERVADO.csv`, `prospective_topB_candidatos_2024_NO_OBSERVADO.json` |
| Appendix A, permutation importance | `results/interpretation/permutation_importance.csv` |
| Appendix B, rolling-origin cuts | `results/landmark/per_landmark_validation.csv` |
| Forest-class composition | `results/interpretation/forest_class_composition.csv` |

**Metric aggregation.** Table 1 reports AUROC and AUPRC computed within each positional draw and averaged over the 20 draws (`test_landmark_2014_metrics_per_draw.csv`). `test_landmark_2014_metrics.csv` instead computes them once over the pooled rows of all draws. The two agree for AUROC but not for AUPRC (e.g. LightGBM 0.0390 averaged vs. 0.0371 pooled), because average precision is not linear in the rows. Paired bootstrap differences are computed on pooled rows.

Large intermediates (trajectories, survival datasets, row-level predictions) are not versioned and are regenerated by the pipeline. The prospective ranking (`*_NO_OBSERVADO*`) is a forward projection with no observed outcome yet; it is not an independent validation.

## Requirements

Python 3.11. `requirements.txt` lists the packages; `requirements-lock.txt` gives the exact versions of the analysis environment.

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
