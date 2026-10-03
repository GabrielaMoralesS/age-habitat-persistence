# CCAI OBSOLESCENCIA - PLAN_V4_VOZ_ECOLOGICA.md, Tier 1 punto 4: importancia por PERMUTACION de
# las covariables del landmark boosting sobre el TEST CONGELADO 2014->2019, respetando bloques
# espaciales al permutar y promediando sobre los 20 draws posicionales, con intervalos por
# repeticion de permutacion. Reentrena el modelo con EXACTAMENTE la misma configuracion que
# 24_landmark_forecast.py (mismos landmarks de entrenamiento, misma busqueda rolling-origin, mismo
# random_state=42) porque test_landmark_2014_predictions.csv solo guarda los SCORES ya calculados,
# no el objeto LightGBM -- la importancia por permutacion necesita reevaluar el modelo sobre
# covariables perturbadas, no solo leer predicciones ya hechas.
# Lee results/trajectory/.chunks/, results/trajectory/run_metadata.json,
# results/survival/dataset_k{K}_buffer{B}m.csv y data/processed/occurrences_with_cell_id.csv
# (las mismas cuatro fuentes que 24_landmark_forecast.py). NO CONSULTA EARTH ENGINE.
"""
POR QUE
-------
GANANCIA INTERNA DEL MODELO, NO USADA A PROPOSITO: LightGBM expone feature_importances_ (ganancia
o conteo de splits) gratis, pero esa importancia es IN-SAMPLE (se mide sobre los datos de
entrenamiento, no sobre el test congelado) y esta sesgada hacia variables de mayor cardinalidad o
mayor varianza, sin relacion necesaria con cuanto una variable ayuda a predecir en datos nuevos
-- exactamente el problema que motiva la importancia por permutacion (Breiman 2001), que mide la
caida de desempeño OUT-OF-SAMPLE al destruir la relacion covariable-outcome, la unica lectura
defendible para un parrafo de interpretacion ecologica ("el modelo predice bien PORQUE...").

POR QUE REENTRENAR EN VEZ DE LEER test_landmark_2014_predictions.csv: ese CSV tiene los scores ya
calculados por el modelo ORIGINAL, pero la importancia por permutacion exige volver a llamar
predict_proba() sobre covariables perturbadas -- hace falta el objeto LightGBM entrenado, que
24_landmark_forecast.py nunca persiste a disco. Se reconstruye aca llamando a las MISMAS funciones
de ese script (build_entries_by_location, load_sampled_codes, build_landmark_dataset,
rolling_origin_select_hyperparams, fit_predict_lgb), en el mismo orden, con los mismos argumentos
por defecto y el mismo random_state=42 -- deterministico, así que el modelo reconstruido es
BIT-A-BIT el mismo que genero score_landmark_boosting en ese CSV. El sanity check en diagnostics.txt
compara el AUROC pooled reconstruido contra el 0.794541 ya reportado en
results/landmark/test_landmark_2014_metrics.csv para confirmar la reconstruccion antes de reportar
ninguna importancia.

PERMUTACION RESTRINGIDA A CADA BLOQUE ESPACIAL DE 100 KM, NO GLOBAL: una permutacion global (mezclar
el valor de una covariable entre CUALQUIER par de localidades del bioma, sin importar distancia)
crea combinaciones covariable-ubicacion fisicamente inverosimiles bajo autocorrelacion espacial
(p.ej. asignarle a una localidad de Restinga en el litoral la dinamica de perdida reciente de un
fragmento interior de Parana) y puede inflar o deprimir la caida de desempeño medida por razones
que no son la relacion covariable-outcome que se quiere aislar. Restringir la permutacion a
intercambiar valores SOLO entre localidades del MISMO bloque espacial de 100 km (los mismos bloques,
mismo CRS EPSG:5880, que 22_survival_models.py y 24_landmark_forecast.py ya usan para el bootstrap
de todo el proyecto) mantiene el nulo de la permutacion dentro del rango de heterogeneidad espacial
real observada en el bioma. Bloques con una sola localidad no admiten permutacion (no hay con quien
intercambiar el valor): se cuentan y se excluyen de esa repeticion, reportados en diagnostics.txt;
con ~32.000 localidades en ~100-150 bloques de 100 km, la fraccion de bloques singleton es marginal.

COVARIABLES CONSTANTES POR LOCALIDAD A TRAVES DE LOS 20 DRAWS -- LA PERMUTACION OPERA POR
LOCALIDAD, NO POR FILA: se verifico empiricamente (test_landmark_2014_predictions.csv, nunique<=1
por localidad en las cuatro covariables) que age_at_t/A_i_t/loss_5y/n_transitions_before_t son
identicas para todos los draws de una misma localidad en un landmark fijo -- build_landmark_dataset
las calcula una vez por (location_id, entry_idx), nunca por draw (ver bloque POR QUE de
24_landmark_forecast.py sobre covariables agregadas). Permutar fila por fila (localidad, draw)
romperia esa estructura real (una misma localidad recibiria hasta 20 valores permutados distintos,
uno por draw, un artefacto que no existe en los datos observados) y ademas pseudo-replicaria la
permutacion. Por eso la permutacion se arma UNA vez por localidad (tabla deduplicada por
location_id) y el valor permutado se transmite por igual a todas las filas (localidad, draw) de esa
localidad antes de llamar a predict_proba sobre la tabla completa.

PROMEDIO SOBRE LOS 20 DRAWS DENTRO DE CADA REPETICION, INTERVALOS ENTRE REPETICIONES: para cada
repeticion de permutacion, la caida de AUROC/AUPRC se calcula POR DRAW (usando solo las filas de
ese draw, exactamente como astro el resto del proyecto trata la incertidumbre posicional -- "Point
AUROC/AUPRC values are computed within each of the 20 fixed positional draws and then averaged" en
el paper) y luego se promedia sobre los draws con ambas clases presentes -- un unico numero de
importancia por repeticion. El intervalo de 95% surge de la distribucion de ese numero A TRAVES DE
las repeticiones de permutacion (la fuente de aleatoriedad propia del metodo de Breiman), no de la
dispersion entre draws (que es incertidumbre posicional, una propiedad de los datos, no del
metodo) -- dos fuentes de incertidumbre distintas, igual que en el resto del proyecto no se
mezclan bootstrap espacial con draws posicionales dentro del mismo intervalo.

IMPORTANCIA CONJUNTA POR FAMILIA, NO SUMA DE IMPORTANCIAS INDIVIDUALES: para "dinamica reciente"
(loss_5y + n_transitions_before_t) las dos columnas se permutan JUNTAS, con el mismo indice de
permutacion dentro de cada bloque -- asi se preserva la correlacion entre ambas covariables tal
como aparece en los datos reales y se mide el aporte conjunto del grupo, en vez de sumar dos
importancias individuales calculadas con permutaciones independientes (que sobreestima el aporte
de un grupo cuando sus covariables estan correlacionadas, un problema conocido de la importancia
por permutacion univariada). "Edad" y "soporte remanente" tienen una sola covariable cada una
(age_at_t, A_i_t) asi que su importancia de familia es identica a su importancia individual -- no
se recalculan, se reusan los mismos resultados para evitar cómputo redundante.

"DESCRIPTORES DE HABITAT" NO TIENE COVARIABLE PROPIA EN EL MODELO ACTUAL -- DECLARADO, NO
INVENTADO: COVARIATE_COLS en 24_landmark_forecast.py es exactamente
["age_at_t","A_i_t","loss_5y","n_transitions_before_t"]; no existe una quinta covariable de
"descriptores de habitat" (p.ej. clase de vegetacion, ver PLAN_V4 Tier 1 punto 5 sobre las clases
3/4/5/49) en el landmark boosting que se entrena hoy. Esta familia se reporta en el CSV de salida
con covariables=[] y todas las metricas en NaN, en vez de omitirse en silencio o de mapearla por
error a alguna de las otras cuatro columnas -- el hueco es un hallazgo a favor del punto 5 del
plan (agregar la clase de vegetacion como covariable), no un bug de esta reconstruccion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
R3_PATH = SCRIPT_DIR / "24_landmark_forecast.py"
D1_PATH = SCRIPT_DIR / "20_trajectory_extract.py"
D2_PATH = SCRIPT_DIR / "21_build_survival_dataset.py"
M1_PATH = SCRIPT_DIR / "22_survival_models.py"

N_REPEATS_DEFAULT = 200
RANDOM_SEED_DEFAULT = 42

# location_id, draw_id -- llaves de fila; el resto son covariables o metadatos de outcome.
INDIVIDUAL_COVARIATES = ["age_at_t", "A_i_t", "loss_5y", "n_transitions_before_t"]
FAMILY_LABELS = {
    "age": "Age",
    "remaining_support": "Remaining support",
    "recent_dynamics": "Recent dynamics (loss + transitions)",
    "habitat_descriptors": "Habitat descriptors",
}
FAMILY_COVARIATES = {
    "age": ["age_at_t"],
    "remaining_support": ["A_i_t"],
    "recent_dynamics": ["loss_5y", "n_transitions_before_t"],
    "habitat_descriptors": [],  # ver bloque POR QUE: sin covariable propia en COVARIATE_COLS hoy.
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    obs_root = root
    parser = argparse.ArgumentParser(
        description="PLAN_V4 Tier 1.4 - importancia por permutacion (landmark boosting, test "
        "congelado 2014->2019), bloques espaciales al permutar, promedio sobre 20 draws, "
        "intervalos entre repeticiones."
    )
    parser.add_argument("--occurrences-csv", type=Path, default=root / "data" / "processed" / "occurrences_with_cell_id.csv")
    parser.add_argument("--trajectory-dir", type=Path, default=obs_root / "results" / "trajectory")
    parser.add_argument("--survival-dir", type=Path, default=obs_root / "results" / "survival")
    parser.add_argument("--landmark-dir", type=Path, default=obs_root / "results" / "landmark")
    parser.add_argument("--output-dir", type=Path, default=obs_root / "results" / "interpretation")
    # Mismos defaults que 24_landmark_forecast.py -- ver R3_DEFAULTS abajo, cargados desde el modulo.
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS_DEFAULT)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--max-draws", type=int, default=None, help="SOLO pruebas.")
    return parser.parse_args()


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError as exc:
        missing = getattr(exc, "name", "a required package")
        raise SystemExit(
            f"Falta una dependencia: {missing}\nInstalar con:\n"
            "  pip install pandas numpy geopandas lightgbm scikit-learn"
        ) from exc
    return {
        "np": np, "pd": pd, "gpd": gpd, "lgb": lgb,
        "roc_auc_score": roc_auc_score, "average_precision_score": average_precision_score,
    }


def load_module(path: Path, name: str):
    import importlib.util
    if not path.exists():
        raise SystemExit(f"No se encontro {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Reconstruccion EXACTA del final_model y el test_table de 24_landmark_forecast.py (hasta el
# refit final del landmark boosting inclusive; no se reconstruye la logistica, el bootstrap de
# metodos, ni el mapa prospectivo -- no hacen falta aca).
# ---------------------------------------------------------------------------


def rebuild_final_model_and_test_table(rt, r3, d1, d2, m1, args):
    np_mod, pd = rt["np"], rt["pd"]

    meta = r3.json_load_metadata(args.trajectory_dir)
    year_start, year_max = meta["year_start"], meta["year_max"]
    print(f"Ventana: {year_start}-{year_max} (run_metadata.json).")

    df = d1.load_occurrences(pd, args.occurrences_csv, year_start)
    df = d1.filter_year_upper_bound(df, year_max, r3.YEAR_MAX_TOLERANCE_DEFAULT)
    entries_by_location = r3.build_entries_by_location(pd, df)
    print(f"Localidades unicas: {len(entries_by_location):,}")

    sampled_codes_by_draw, draws_available = r3.load_sampled_codes(
        pd, args.survival_dir, r3.K_DEFAULT, r3.BUFFER_M_DEFAULT, r3.N_DRAWS_DEFAULT, args.max_draws
    )
    print(f"Draws Monte Carlo reusados de D2: {len(draws_available)}")

    combined = d1.load_buffer_trajectory(pd, args.trajectory_dir / ".chunks", r3.BUFFER_M_DEFAULT)
    if combined is None:
        raise SystemExit(f"Sin checkpoints de trayectoria para buffer {r3.BUFFER_M_DEFAULT}m.")
    loc_code_tables = d2.build_location_code_tables(pd, combined)

    all_landmarks = r3.TRAIN_LANDMARKS_DEFAULT + [r3.TEST_LANDMARK_DEFAULT]
    landmark_tables = {}
    for t in all_landmarks:
        print(f"--- Reconstruyendo dataset del landmark {t} ---")
        table, _ = r3.build_landmark_dataset(
            rt, d2, m1, t, r3.HORIZON_YEARS_DEFAULT, r3.K_DEFAULT, year_start, year_max,
            entries_by_location, loc_code_tables, sampled_codes_by_draw, r3.LOSS_WINDOW_YEARS,
        )
        landmark_tables[t] = table
        if len(table) == 0:
            raise SystemExit(f"Dataset vacio para el landmark {t} -- no se puede reconstruir el modelo.")

    print("\n=== Reconstruyendo seleccion de hiperparametros (rolling-origin, identica a R3) ===")
    best, _ = r3.rolling_origin_select_hyperparams(
        rt, lambda params, tr, va: r3.fit_predict_lgb(rt["lgb"], params, tr, va),
        rt["average_precision_score"], landmark_tables, r3.TRAIN_LANDMARKS_DEFAULT, r3.LGBM_GRID,
    )
    print(f"Mejor combinacion (debe coincidir con R3): {best['params']}")

    train_df_final = pd.concat(
        [landmark_tables[t] for t in r3.TRAIN_LANDMARKS_DEFAULT], ignore_index=True
    )
    final_model = rt["lgb"].LGBMClassifier(**best["params"], random_state=42, verbose=-1)
    final_model.fit(train_df_final[r3.COVARIATE_COLS], train_df_final["Z"])
    print(f"Refit final sobre {len(train_df_final):,} filas ({len(r3.TRAIN_LANDMARKS_DEFAULT)} landmarks).")

    test_table = landmark_tables[r3.TEST_LANDMARK_DEFAULT].reset_index(drop=True)
    print(f"Test congelado {r3.TEST_LANDMARK_DEFAULT}: {len(test_table):,} filas "
          f"(localidad x draw), {test_table['location_id'].nunique():,} localidades unicas.")
    return final_model, test_table, r3.COVARIATE_COLS


# ---------------------------------------------------------------------------
# Permutacion restringida a bloque espacial, por localidad (no por fila).
# ---------------------------------------------------------------------------


def build_locality_table(pd, test_table, covariate_cols):
    loc = test_table.drop_duplicates(subset=["location_id"])[["location_id"] + covariate_cols].reset_index(drop=True)
    return loc


def permute_within_blocks(rng, loc_table, block_ids, cols_to_permute, pd):
    """Permuta conjuntamente cols_to_permute entre localidades del MISMO bloque espacial de 100
    km. Bloques con una sola localidad quedan sin cambios (no hay con quien intercambiar el
    valor) -- se cuentan y se devuelven para diagnostics."""
    out = loc_table.copy()
    n_singleton_blocks = 0
    n_singleton_locs = 0
    for block_id in pd.unique(block_ids):
        idx = (block_ids == block_id).nonzero()[0]
        if len(idx) < 2:
            n_singleton_blocks += 1
            n_singleton_locs += len(idx)
            continue
        perm = idx.copy()
        # rng.permutation reordena una copia -- se aplica sobre las FILAS del bloque, no globalmente.
        perm = rng.permutation(perm)
        out.loc[idx, cols_to_permute] = loc_table.loc[perm, cols_to_permute].to_numpy()
    return out, n_singleton_blocks, n_singleton_locs


def draw_averaged_auc_drop(np_mod, roc_auc_score, average_precision_score, test_table, baseline_scores, permuted_scores):
    """Caida de AUROC/AUPRC (baseline - permutado) calculada DENTRO de cada uno de los draws
    posicionales y promediada sobre los draws con ambas clases presentes -- ver bloque POR QUE."""
    auroc_drops, auprc_drops = [], []
    draw_ids = test_table["draw_id"].to_numpy()
    y = test_table["Z"].to_numpy()
    for draw_id in np_mod.unique(draw_ids):
        mask = draw_ids == draw_id
        y_d = y[mask]
        if len(np_mod.unique(y_d)) < 2:
            continue
        b_auroc = roc_auc_score(y_d, baseline_scores[mask])
        p_auroc = roc_auc_score(y_d, permuted_scores[mask])
        b_auprc = average_precision_score(y_d, baseline_scores[mask])
        p_auprc = average_precision_score(y_d, permuted_scores[mask])
        auroc_drops.append(b_auroc - p_auroc)
        auprc_drops.append(b_auprc - p_auprc)
    n_used = len(auroc_drops)
    if n_used == 0:
        return float("nan"), float("nan"), 0
    return float(np_mod.mean(auroc_drops)), float(np_mod.mean(auprc_drops)), n_used


def permutation_importance_for_target(
    rt, rng, final_model, test_table, covariate_cols, loc_table, block_ids, cols_to_permute,
    baseline_scores, n_repeats: int,
):
    np_mod, pd, roc_auc_score, average_precision_score = rt["np"], rt["pd"], rt["roc_auc_score"], rt["average_precision_score"]
    auroc_drop_reps, auprc_drop_reps = [], []
    n_singleton_blocks_last, n_singleton_locs_last = 0, 0
    for _ in range(n_repeats):
        permuted_loc, n_singleton_blocks_last, n_singleton_locs_last = permute_within_blocks(
            rng, loc_table, block_ids, cols_to_permute, pd
        )
        # Transmite el valor permutado, por localidad, a TODAS sus filas (localidad, draw) -- ver
        # bloque POR QUE sobre por que la permutacion es por localidad y no por fila.
        value_by_loc = permuted_loc.set_index("location_id")[cols_to_permute]
        X_perm = test_table[covariate_cols].copy()
        replacement = test_table[["location_id"]].join(value_by_loc, on="location_id")[cols_to_permute]
        X_perm[cols_to_permute] = replacement.to_numpy()
        permuted_scores = final_model.predict_proba(X_perm[covariate_cols])[:, 1]
        auroc_drop, auprc_drop, n_draws_used = draw_averaged_auc_drop(
            np_mod, roc_auc_score, average_precision_score, test_table, baseline_scores, permuted_scores
        )
        auroc_drop_reps.append(auroc_drop)
        auprc_drop_reps.append(auprc_drop)
    auroc_arr = np_mod.array(auroc_drop_reps, dtype=float)
    auprc_arr = np_mod.array(auprc_drop_reps, dtype=float)
    return {
        "auroc_drop_mean": float(np_mod.nanmean(auroc_arr)),
        "auroc_drop_ci95_lo": float(np_mod.nanpercentile(auroc_arr, 2.5)),
        "auroc_drop_ci95_hi": float(np_mod.nanpercentile(auroc_arr, 97.5)),
        "auprc_drop_mean": float(np_mod.nanmean(auprc_arr)),
        "auprc_drop_ci95_lo": float(np_mod.nanpercentile(auprc_arr, 2.5)),
        "auprc_drop_ci95_hi": float(np_mod.nanpercentile(auprc_arr, 97.5)),
        "n_repeats": n_repeats,
        "n_singleton_blocks_last_repeat": n_singleton_blocks_last,
        "n_singleton_locs_last_repeat": n_singleton_locs_last,
    }


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    np, pd, gpd = rt["np"], rt["pd"], rt["gpd"]

    print("PLAN_V4 Tier 1.4 - Importancia por permutacion (landmark boosting, test 2014->2019)")
    print("=======================================================================================")

    r3 = load_module(R3_PATH, "r3_landmark_forecast")
    d1 = load_module(D1_PATH, "d1_trajectory_extract")
    d2 = load_module(D2_PATH, "d2_build_survival_dataset")
    m1 = load_module(M1_PATH, "m1_survival_models")

    final_model, test_table, covariate_cols = rebuild_final_model_and_test_table(rt, r3, d1, d2, m1, args)
    assert covariate_cols == INDIVIDUAL_COVARIATES, (
        f"COVARIATE_COLS de 24_landmark_forecast.py cambio respecto de lo esperado por este script: "
        f"{covariate_cols} vs {INDIVIDUAL_COVARIATES}. Actualizar FAMILY_COVARIATES antes de continuar."
    )

    baseline_scores = final_model.predict_proba(test_table[covariate_cols])[:, 1]
    y_true = test_table["Z"].to_numpy()
    baseline_auroc_pooled = float(rt["roc_auc_score"](y_true, baseline_scores))
    baseline_auprc_pooled = float(rt["average_precision_score"](y_true, baseline_scores))
    known_auroc = 0.794541
    print(f"\nSanity check -- AUROC pooled reconstruido: {baseline_auroc_pooled:.6f} "
          f"(reportado en test_landmark_2014_metrics.csv: {known_auroc:.6f}, "
          f"delta={baseline_auroc_pooled - known_auroc:+.6f}).")
    sanity_ok = abs(baseline_auroc_pooled - known_auroc) < 1e-4

    baseline_auroc_drawavg, baseline_auprc_drawavg, n_draws_used = draw_averaged_auc_drop(
        np, rt["roc_auc_score"], rt["average_precision_score"], test_table, baseline_scores, baseline_scores
    )
    # (baseline vs baseline: la "caida" es por construccion 0; se reusa la funcion solo para
    # obtener n_draws_used y confirmar cuantos de los 20 draws tienen ambas clases presentes.)
    baseline_auroc_by_draw = []
    baseline_auprc_by_draw = []
    for draw_id in np.unique(test_table["draw_id"].to_numpy()):
        mask = test_table["draw_id"].to_numpy() == draw_id
        y_d = y_true[mask]
        if len(np.unique(y_d)) < 2:
            continue
        baseline_auroc_by_draw.append(float(rt["roc_auc_score"](y_d, baseline_scores[mask])))
        baseline_auprc_by_draw.append(float(rt["average_precision_score"](y_d, baseline_scores[mask])))
    baseline_auroc_drawavg = float(np.mean(baseline_auroc_by_draw))
    baseline_auprc_drawavg = float(np.mean(baseline_auprc_by_draw))
    print(f"Baseline AUROC promediado sobre {n_draws_used}/{test_table['draw_id'].nunique()} draws "
          f"con ambas clases: {baseline_auroc_drawavg:.4f}. Baseline AUPRC promediado: "
          f"{baseline_auprc_drawavg:.4f}.")

    loc_table = build_locality_table(pd, test_table, covariate_cols)
    block_ids = m1.assign_spatial_blocks(gpd, np, pd, loc_table["location_id"])
    n_blocks = len(pd.unique(block_ids))
    print(f"\nBloques espaciales de 100 km (EPSG:5880) sobre {len(loc_table):,} localidades: {n_blocks:,} bloques.")

    rng = np.random.default_rng(args.random_seed)
    results = {}

    print(f"\nImportancia individual por covariable ({args.n_repeats} repeticiones, permutacion "
          "restringida a cada bloque espacial)...")
    for cov in INDIVIDUAL_COVARIATES:
        print(f"  Permutando {cov}...")
        res = permutation_importance_for_target(
            rt, rng, final_model, test_table, covariate_cols, loc_table, block_ids, [cov],
            baseline_scores, args.n_repeats,
        )
        results[("individual", cov)] = res
        print(f"    AUROC drop={res['auroc_drop_mean']:.4f} "
              f"[{res['auroc_drop_ci95_lo']:.4f}, {res['auroc_drop_ci95_hi']:.4f}]  "
              f"AUPRC drop={res['auprc_drop_mean']:.4f} "
              f"[{res['auprc_drop_ci95_lo']:.4f}, {res['auprc_drop_ci95_hi']:.4f}]")

    print("\nImportancia agrupada por familia de covariable...")
    for family, cols in FAMILY_COVARIATES.items():
        if len(cols) == 0:
            print(f"  {FAMILY_LABELS[family]}: sin covariable propia en COVARIATE_COLS -- NaN (ver bloque POR QUE).")
            results[("family", family)] = {
                "auroc_drop_mean": float("nan"), "auroc_drop_ci95_lo": float("nan"), "auroc_drop_ci95_hi": float("nan"),
                "auprc_drop_mean": float("nan"), "auprc_drop_ci95_lo": float("nan"), "auprc_drop_ci95_hi": float("nan"),
                "n_repeats": 0, "n_singleton_blocks_last_repeat": 0, "n_singleton_locs_last_repeat": 0,
            }
        elif len(cols) == 1 and ("individual", cols[0]) in results:
            print(f"  {FAMILY_LABELS[family]}: identica a la covariable individual {cols[0]} (reusada, sin recalcular).")
            results[("family", family)] = results[("individual", cols[0])]
        else:
            print(f"  {FAMILY_LABELS[family]}: permutando conjuntamente {cols}...")
            res = permutation_importance_for_target(
                rt, rng, final_model, test_table, covariate_cols, loc_table, block_ids, cols,
                baseline_scores, args.n_repeats,
            )
            results[("family", family)] = res
            print(f"    AUROC drop={res['auroc_drop_mean']:.4f} "
                  f"[{res['auroc_drop_ci95_lo']:.4f}, {res['auroc_drop_ci95_hi']:.4f}]  "
                  f"AUPRC drop={res['auprc_drop_mean']:.4f} "
                  f"[{res['auprc_drop_ci95_lo']:.4f}, {res['auprc_drop_ci95_hi']:.4f}]")

    # --- salidas ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cov in INDIVIDUAL_COVARIATES:
        r = results[("individual", cov)]
        rows.append({
            "level": "individual", "name": cov, "covariates": cov,
            "auroc_drop_mean": r["auroc_drop_mean"], "auroc_drop_ci95_lo": r["auroc_drop_ci95_lo"], "auroc_drop_ci95_hi": r["auroc_drop_ci95_hi"],
            "auprc_drop_mean": r["auprc_drop_mean"], "auprc_drop_ci95_lo": r["auprc_drop_ci95_lo"], "auprc_drop_ci95_hi": r["auprc_drop_ci95_hi"],
            "n_repeats": r["n_repeats"],
        })
    for family in FAMILY_COVARIATES:
        r = results[("family", family)]
        rows.append({
            "level": "family", "name": family, "covariates": "+".join(FAMILY_COVARIATES[family]) if FAMILY_COVARIATES[family] else "",
            "auroc_drop_mean": r["auroc_drop_mean"], "auroc_drop_ci95_lo": r["auroc_drop_ci95_lo"], "auroc_drop_ci95_hi": r["auroc_drop_ci95_hi"],
            "auprc_drop_mean": r["auprc_drop_mean"], "auprc_drop_ci95_lo": r["auprc_drop_ci95_lo"], "auprc_drop_ci95_hi": r["auprc_drop_ci95_hi"],
            "n_repeats": r["n_repeats"],
        })
    out_table = pd.DataFrame(rows).sort_values(
        by=["level", "auroc_drop_mean"], ascending=[True, False]
    )
    out_csv = args.output_dir / "permutation_importance.csv"
    out_table.to_csv(out_csv, index=False)
    print(f"\nGuardado: {out_csv}")

    diagnostics = [
        "PLAN_V4 Tier 1.4 diagnostics - importancia por permutacion (landmark boosting, test 2014->2019)",
        "=========================================================================================================",
        f"Modelo reentrenado con los mismos landmarks de entrenamiento ({r3.TRAIN_LANDMARKS_DEFAULT}) y "
        f"la misma busqueda rolling-origin que 24_landmark_forecast.py (random_state=42, mismo LGBM_GRID).",
        f"Sanity check -- AUROC pooled reconstruido: {baseline_auroc_pooled:.6f} vs. reportado "
        f"({known_auroc:.6f}) en test_landmark_2014_metrics.csv. Delta={baseline_auroc_pooled - known_auroc:+.6f}. "
        f"{'OK (< 1e-4)' if sanity_ok else 'AVISO: fuera de tolerancia -- revisar reconstruccion antes de usar estos numeros.'}",
        f"AUPRC pooled reconstruido: {baseline_auprc_pooled:.6f}.",
        f"Baseline promediado sobre draws (mismo criterio usado para las caidas por permutacion): "
        f"AUROC={baseline_auroc_drawavg:.4f}, AUPRC={baseline_auprc_drawavg:.4f}, sobre "
        f"{n_draws_used}/{test_table['draw_id'].nunique()} draws con ambas clases presentes.",
        f"Localidades en el test congelado: {len(loc_table):,}. Bloques espaciales de 100 km (EPSG:5880): {n_blocks:,}.",
        f"Repeticiones de permutacion por covariable/familia: {args.n_repeats}.",
        "",
        "METODO (resumen, ver bloque POR QUE en el script para el detalle completo):",
        "  - Importancia = caida de AUROC/AUPRC (baseline - permutado) del landmark boosting ya "
        "entrenado, NO ganancia interna del modelo (in-sample, sesgada hacia alta cardinalidad).",
        "  - La permutacion de cada covariable (o grupo de covariables, para la importancia por "
        "familia) se restringe a intercambiar valores SOLO entre localidades del MISMO bloque "
        "espacial de 100 km -- nunca globalmente entre todo el bioma.",
        "  - La permutacion opera por LOCALIDAD (las cuatro covariables son constantes a traves de "
        "los 20 draws de una misma localidad, verificado empiricamente) y el valor permutado se "
        "transmite por igual a todas las filas (localidad, draw) de esa localidad.",
        "  - Dentro de cada repeticion, la caida de AUROC/AUPRC se calcula POR DRAW y se promedia "
        "sobre los draws con ambas clases presentes (mismo criterio que el resto del proyecto para "
        "la incertidumbre posicional).",
        "  - El intervalo de 95% es el percentil [2.5, 97.5] de esa caida promediada-por-draws A "
        "TRAVES de las repeticiones de permutacion -- la fuente de incertidumbre es la aleatoriedad "
        "del metodo de permutacion, no la dispersion entre draws (que ya esta promediada dentro de "
        "cada repeticion).",
        "",
        "IMPORTANCIA INDIVIDUAL POR COVARIABLE:",
    ]
    for cov in INDIVIDUAL_COVARIATES:
        r = results[("individual", cov)]
        diagnostics.append(
            f"  {cov}: AUROC drop={r['auroc_drop_mean']:.4f} "
            f"[{r['auroc_drop_ci95_lo']:.4f}, {r['auroc_drop_ci95_hi']:.4f}]  "
            f"AUPRC drop={r['auprc_drop_mean']:.4f} "
            f"[{r['auprc_drop_ci95_lo']:.4f}, {r['auprc_drop_ci95_hi']:.4f}]  "
            f"(bloques singleton en la ultima repeticion: {r['n_singleton_blocks_last_repeat']} "
            f"bloques / {r['n_singleton_locs_last_repeat']} localidades, sin permutar)."
        )
    diagnostics += ["", "IMPORTANCIA AGRUPADA POR FAMILIA:"]
    for family in FAMILY_COVARIATES:
        r = results[("family", family)]
        cols = FAMILY_COVARIATES[family]
        if len(cols) == 0:
            diagnostics.append(
                f"  {FAMILY_LABELS[family]}: SIN covariable propia en COVARIATE_COLS actual "
                f"(age_at_t, A_i_t, loss_5y, n_transitions_before_t) -- NaN, no calculada. Ver bloque "
                "POR QUE: esto es un hueco real del modelo (falta clase de vegetacion / descriptor de "
                "habitat como covariable), no un error de mapeo de esta reconstruccion."
            )
        else:
            diagnostics.append(
                f"  {FAMILY_LABELS[family]} ({'+'.join(cols)}): AUROC drop={r['auroc_drop_mean']:.4f} "
                f"[{r['auroc_drop_ci95_lo']:.4f}, {r['auroc_drop_ci95_hi']:.4f}]  "
                f"AUPRC drop={r['auprc_drop_mean']:.4f} "
                f"[{r['auprc_drop_ci95_lo']:.4f}, {r['auprc_drop_ci95_hi']:.4f}]"
            )
    diag_path = args.output_dir / "permutation_importance_diagnostics.txt"
    diag_path.write_text("\n".join(str(l) for l in diagnostics) + "\n", encoding="utf-8")
    print(f"\nGuardado: {diag_path}")


if __name__ == "__main__":
    main()
