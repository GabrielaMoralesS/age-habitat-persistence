# CCAI OBSOLESCENCIA - M1/M2 (PLAN_V3_EJECUTABLE.md): tres modelos de CALIBRACION (R2) -- KM,
# Rayleigh sigma=30 fijo, Rayleigh sigma estimado por verosimilitud censurada -- evaluados SOLO
# dentro del soporte temporal observado, con validacion espacialmente cross-fitted (nunca
# in-sample). Lee results/survival/dataset_k{K}_buffer{B}m.csv (de 21_build_survival_dataset.py).
# NO CONSULTA EARTH ENGINE.
"""
POR QUE
-------
D2 ya produjo el dataset de supervivencia; este script responde la pregunta de R2: "¿esta el peso
temporal calibrado con la persistencia observada?" -- NO la pregunta de prediccion (esa es R3,
landmark forecasting, otro script). El plan es explicito en que estos son DOS modelos con DOS
propositos (bloque "SEPARACION OBLIGATORIA: calibracion != prediccion"): KM, Rayleigh-30 y
Rayleigh-sigma-estimado se usan EXCLUSIVAMENTE dentro del soporte temporal observado -- nunca se
les pide extrapolar mas alla del seguimiento maximo disponible, que es lo que la version anterior
de este proyecto (con gradient-boosted survival en M1) intentaba hacer y el plan explicitamente
corto.

SOLO TRES MODELOS, COX Y BOOSTING ELIMINADOS DE M1 A PROPOSITO: el plan lo dice sin ambiguedad
("Cox se elimina", "sustituido por el landmark model de R3"). Este script NO implementa ninguno de
los dos -- no es un descuido, es la especificacion.

NO C-INDEX NI CALIBRACION POR DECILES: KM, Rayleigh-30 y Rayleigh-sigma-hat son modelos
POBLACIONALES -- devuelven la MISMA curva de supervivencia para TODAS las localidades, no un
ranking individual. Un C-index sobre predicciones identicas para todos daria ~0,5 por
construccion (no hay orden que discriminar), y los deciles de calibracion no significan nada sin
variacion entre individuos. Esas metricas son para R3 (landmark model, con predicciones
individuales). Aca se usan las que el plan pide explicitamente: Integrated Brier Score (con
censura via IPCW), log-score, error absoluto y error integrado contra la curva KM de PRUEBA
(fuera de muestra), y sigma-hat con intervalo por bootstrap de bloque espacial.

scikit-survival PARA KM/BRIER, NO REIMPLEMENTADO A MANO: el Brier Score censurado exige ponderar
por la probabilidad inversa de censura (IPCW) -- un calculo con varias formas de introducir un
error sutil (la propia scikit-survival documenta casos borde de soporte temporal). Se instala
`scikit-survival` (antes ausente del entorno) y se usan `kaplan_meier_estimator`, `brier_score` e
`integrated_brier_score`, en vez de una implementacion propia sin la cobertura de tests de una
libreria establecida. *** ADVERTENCIA DE ENTORNO: instalar scikit-survival actualizo numpy
(1.26.3->2.4.6), pandas (2.1.4->3.0.5) y scikit-learn (1.3.2->1.9.0) en el .venv_ai COMPARTIDO por
TODO el proyecto, no solo por este script -- verificado con una prueba de regresion (re-correr el
smoke test de 21_build_survival_dataset.py con --max-locations 300 despues de instalar: mismos
numeros exactos que antes de instalar) y con un chequeo de que geopandas/lightgbm/rasterio/sklearn/
ee siguen importando sin conflicto. No deberia romper nada, pero es un cambio de entorno compartido
que vale la pena declarar aca, no solo en el chat. ***

MLE DE RAYLEIGH CENSURADO EN FORMA CERRADA, VERIFICADA NUMERICAMENTE: para S(a)=exp(-a^2/2*sigma^2)
con censura administrativa (Tipo I), el estimador de maxima verosimilitud tiene forma cerrada:
sigma_hat = sqrt( sum(duration_i^2) / (2 * n_eventos) ), sumando duration^2 sobre TODAS las
observaciones (eventos y censuradas) y dividiendo por el doble del numero de EVENTOS unicamente --
resultado analogo al MLE de la exponencial bajo censura (sigma^2 = suma/2*n_eventos en vez de
media). Usar la forma cerrada es mas rapido que optimizar numericamente miles de veces (20 draws x
5 folds x bootstrap), pero NO se da por sentada: verify_mle_matches_numerical_optimization() la
contrasta contra scipy.optimize.minimize_scalar sobre la log-verosimilitud negativa, en una
muestra, con assert de coincidencia -- mismo criterio de "verificar, no declarar" que el resto del
proyecto (D1: encode(decode(code))==code; D2: aritmetica de bits vs decode()).

VALIDACION ESPACIALMENTE CROSS-FITTED: GroupKFold sobre bloques espaciales de 100 km (misma
convencion y mismo tamaño de bloque que 02/06/09), agrupando por bloque de la LOCALIDAD (nunca por
registro, y nunca por año-entrada por separado de su localidad -- todas las entradas-año de una
misma localidad caen siempre en el mismo fold, porque comparten posicion fisica). Rayleigh-sigma-
hat y la curva KM usada como "referencia de entrenamiento" se ajustan SOLO con el fold de
entrenamiento; las metricas (IBS, log-score, error vs KM) se calculan sobre el fold de prueba,
nunca sobre los mismos datos con los que se ajusto nada. La curva "KM de prueba" (usada como
verdad empirica para el error absoluto/integrado) se ajusta POR SEPARADO, solo con el fold de
prueba -- comparar contra una curva que ya vio los datos de entrenamiento invalidaria la
comparacion.

20 DRAWS MONTE CARLO, MISMOS QUE D2, VARIABILIDAD REPORTADA APARTE: cada draw ya trae su propia
columna (duration, event) por localidad-año (la posicion muestreada de esa localidad en ESE draw,
ver bloque POR QUE de 21_build_survival_dataset.py). Este script corre el pipeline completo
(cross-fitting + metricas + sigma-hat + bootstrap) UNA vez POR CADA uno de los 20 draws -- no un
solo draw "representativo" -- y reporta la media entre draws junto con su desvio estandar, para
separar la incertidumbre POSICIONAL (entre draws) de la incertidumbre de MUESTREO/ESPACIAL (bootstrap
de bloque dentro de un draw), tal como el plan exige que se reporten por separado.

SIGMA=30 NO SE AJUSTA, EL RESCALADO A [0.05,1] NO ENTRA ACA: Rayleigh-30 usa sigma=30 fijo (el
"literature weight" del nucleo conceptual del plan) sin ningun ajuste, evaluado igual que los otros
dos SOLO para comparar calibracion -- no se reescala a [0.05,1] en ningun momento de este script,
porque esa version reescalada NO es una funcion de supervivencia valida (no converge a 0), es un
peso practico para otros scripts, no una hipotesis de supervivencia que se pueda calibrar aca.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HORIZONS_DEFAULT = [5.0, 10.0, 20.0]
N_RISK_THRESHOLD_DEFAULT = 200  # umbral fijado de antemano (M2), ver n_risk_k2_buffer250m ya calculado
SIGMA_FIXED_DEFAULT = 30.0  # literatura, Moreno et al. -- "literature weight" sin reescalar
N_OUTER_FOLDS_DEFAULT = 5  # misma convencion N_SPATIAL_FOLDS que 02/06/09
N_BOOTSTRAP_DEFAULT = 1000
SPATIAL_BLOCK_SIZE_M = 100_000  # identico a 02/06/09
GRID_CRS_FOR_BLOCKS = "EPSG:5880"  # SIRGAS2000/Brasil Polyconic, mismo CRS que data/grid_5km.gpkg
RANDOM_SEED_DEFAULT = 42
N_DRAWS_DEFAULT = 20
FINE_GRID_STEP_YEARS = 0.5  # para el error integrado vs KM de prueba


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    obs_root = root
    parser = argparse.ArgumentParser(
        description="M1/M2 - KM, Rayleigh-30 y Rayleigh-sigma-hat, calibracion espacialmente "
        "cross-fitted dentro del soporte temporal observado."
    )
    parser.add_argument("--survival-dir", type=Path, default=obs_root / "results" / "survival")
    parser.add_argument("--k", type=int, default=2, help="k principal=2, ver D2/R1.")
    parser.add_argument("--buffer-m", type=int, default=250, help="buffer principal=250, sensibilidad=500.")
    parser.add_argument("--output-dir", type=Path, default=obs_root / "results" / "calibration")
    parser.add_argument("--sigma-fixed", type=float, default=SIGMA_FIXED_DEFAULT)
    parser.add_argument("--n-outer-folds", type=int, default=N_OUTER_FOLDS_DEFAULT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--horizons", type=float, nargs="+", default=HORIZONS_DEFAULT)
    parser.add_argument("--n-risk-threshold", type=int, default=N_RISK_THRESHOLD_DEFAULT)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT, help="Debe coincidir con D2.")
    parser.add_argument("--max-draws", type=int, default=None, help="SOLO pruebas: limitar a los primeros N draws.")
    return parser.parse_args()


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        from scipy.optimize import minimize_scalar
        from sklearn.model_selection import GroupKFold
        from sksurv.nonparametric import kaplan_meier_estimator
        from sksurv.metrics import brier_score, integrated_brier_score
        from sksurv.util import Surv
    except ImportError as exc:
        missing = getattr(exc, "name", "a required package")
        raise SystemExit(
            f"Falta una dependencia: {missing}\nInstalar con:\n"
            "  pip install pandas numpy geopandas scipy scikit-learn scikit-survival"
        ) from exc
    return {
        "np": np, "pd": pd, "gpd": gpd, "minimize_scalar": minimize_scalar, "GroupKFold": GroupKFold,
        "kaplan_meier_estimator": kaplan_meier_estimator, "brier_score": brier_score,
        "integrated_brier_score": integrated_brier_score, "Surv": Surv,
    }


# ---------------------------------------------------------------------------
# Carga y bloques espaciales (parseados directo de location_id = "lat_lon", sin releer occurrences)
# ---------------------------------------------------------------------------


def load_evaluable_dataset(pd, path: Path):
    if not path.exists():
        raise SystemExit(f"No se encontro {path} -- correr 21_build_survival_dataset.py primero.")
    df = pd.read_csv(path, usecols=["location_id", "entry_year", "draw_id", "H_i", "event", "duration"])
    return df.dropna(subset=["event", "duration"]).copy()


def assign_spatial_blocks(gpd, np_mod, pd, location_ids: "pd.Series") -> "np.ndarray":
    """location_id = 'lat_lon' (formato de loc_key de 20_trajectory_extract.py) -- se parsea
    directo, sin releer occurrences_with_cell_id.csv ni el grid. Reproyectado a EPSG:5880 (mismo
    CRS que data/grid_5km.gpkg) para que el tamaño de bloque de 100 km sea comparable al resto del
    proyecto (02/06/09), no un bloque de 100 km "en grados" que seria de tamaño fisico distinto
    segun la latitud."""
    unique_locs = location_ids.drop_duplicates()
    lat = unique_locs.str.split("_").str[0].astype(float)
    lon = unique_locs.str.split("_").str[1].astype(float)
    points = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(GRID_CRS_FOR_BLOCKS)
    bx = np_mod.floor(points.x.to_numpy() / SPATIAL_BLOCK_SIZE_M).astype(int)
    by = np_mod.floor(points.y.to_numpy() / SPATIAL_BLOCK_SIZE_M).astype(int)
    block_by_loc = pd.Series([f"{a}_{b}" for a, b in zip(bx, by)], index=unique_locs.to_numpy())
    return location_ids.map(block_by_loc).to_numpy()


# ---------------------------------------------------------------------------
# Rayleigh: MLE en forma cerrada (verificada), supervivencia, densidad, log-verosimilitud.
# ---------------------------------------------------------------------------


def rayleigh_mle_sigma(np_mod, duration, event) -> float:
    n_events = int(event.sum())
    if n_events == 0:
        return float("nan")
    return float(np_mod.sqrt(np_mod.sum(duration.astype(float) ** 2) / (2.0 * n_events)))


def verify_mle_matches_numerical_optimization(np_mod, minimize_scalar_fn, duration, event, seed: int, n_checks: int = 5) -> dict:
    """Chequeo OBLIGATORIO -- ver bloque POR QUE. Contrasta la formula cerrada contra
    scipy.optimize.minimize_scalar sobre la log-verosimilitud negativa, en varias submuestras
    aleatorias (no solo el dataset completo, para que la verificacion cubra distintos tamaños de
    muestra)."""
    rng = np_mod.random.default_rng(seed)
    n = len(duration)
    fails = []
    for _ in range(n_checks):
        idx = rng.choice(n, size=max(50, n // 20), replace=False)
        d, e = duration[idx], event[idx]
        closed = rayleigh_mle_sigma(np_mod, d, e)

        def neg_log_lik(sigma):
            return -rayleigh_log_likelihood(np_mod, sigma, d, e).sum()

        res = minimize_scalar_fn(neg_log_lik, bounds=(0.1, 200.0), method="bounded")
        numeric = float(res.x)
        if abs(closed - numeric) > max(0.05, 0.01 * closed):
            fails.append((closed, numeric))
    return {"n_checks": n_checks, "n_fallas": len(fails), "ejemplos_falla": fails, "ok": len(fails) == 0}


def rayleigh_survival(np_mod, sigma: float, t):
    return np_mod.exp(-(np_mod.asarray(t, dtype=float) ** 2) / (2.0 * sigma**2))


def rayleigh_log_likelihood(np_mod, sigma: float, duration, event):
    d = duration.astype(float)
    log_s = -(d**2) / (2.0 * sigma**2)
    log_f = np_mod.log(np_mod.clip(d, 1e-6, None)) - 2 * np_mod.log(sigma) - (d**2) / (2.0 * sigma**2)
    return np_mod.where(event.astype(bool), log_f, log_s)


def spatial_block_bootstrap_sigma_ci(np_mod, duration, event, blocks, n_bootstrap: int, seed: int):
    """IC por bootstrap de BLOQUE espacial (nunca por registro) -- se remuestrean bloques UNICOS
    con reemplazo, y se reconstruye la muestra concatenando TODAS las filas de los bloques
    remuestreados (preserva la correlacion espacial dentro de cada bloque, a diferencia de un
    bootstrap ingenuo fila-por-fila)."""
    rng = np_mod.random.default_rng(seed)
    unique_blocks = np_mod.unique(blocks)
    block_to_idx = {b: np_mod.where(blocks == b)[0] for b in unique_blocks}
    sigmas = []
    for _ in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np_mod.concatenate([block_to_idx[b] for b in sampled_blocks])
        s = rayleigh_mle_sigma(np_mod, duration[idx], event[idx])
        if not np_mod.isnan(s):
            sigmas.append(s)
    sigmas = np_mod.array(sigmas)
    lo, hi = np_mod.percentile(sigmas, [2.5, 97.5])
    return float(lo), float(hi), sigmas


# ---------------------------------------------------------------------------
# Curvas KM evaluadas en un horizonte arbitrario (escalon, ver step_survival_at).
# ---------------------------------------------------------------------------


def step_survival_at(np_mod, km_times, km_surv, t):
    """Valor de una curva escalonada (KM) en el/los tiempo(s) t -- la supervivencia estimada en t
    es la del ULTIMO tiempo de evento/censura observado que sea <= t (convencion estandar de KM,
    funcion escalonada continua por la derecha en el sentido de sobrevivencia)."""
    t = np_mod.atleast_1d(np_mod.asarray(t, dtype=float))
    idx = np_mod.searchsorted(km_times, t, side="right") - 1
    out = np_mod.ones_like(t)
    valid = idx >= 0
    out[valid] = km_surv[idx[valid]]
    return out


def restrict_horizons_to_support(horizons, train_duration, test_duration):
    """'Restringido al soporte temporal observado' -- un horizonte solo es evaluable si cae
    ESTRICTAMENTE dentro del rango de seguimiento tanto de train como de test (sksurv exige esto
    para brier_score/integrated_brier_score; se aplica el mismo criterio a las demas metricas por
    consistencia)."""
    hi = min(train_duration.max(), test_duration.max())
    lo = max(train_duration.min(), test_duration.min())
    return [h for h in horizons if lo < h < hi]


# ---------------------------------------------------------------------------
# Un fold: ajustar en train, evaluar en test.
# ---------------------------------------------------------------------------


def evaluate_fold(rt, train, test, sigma_fixed, horizons, n_risk_threshold):
    np_mod, Surv, kaplan_meier_estimator, brier_score, integrated_brier_score = (
        rt["np"], rt["Surv"], rt["kaplan_meier_estimator"], rt["brier_score"], rt["integrated_brier_score"]
    )
    y_train = Surv.from_arrays(event=train["event"].to_numpy(dtype=bool), time=train["duration"].to_numpy(dtype=float))
    y_test = Surv.from_arrays(event=test["event"].to_numpy(dtype=bool), time=test["duration"].to_numpy(dtype=float))

    km_train_t, km_train_s = kaplan_meier_estimator(train["event"].to_numpy(dtype=bool), train["duration"].to_numpy(dtype=float))
    km_test_t, km_test_s = kaplan_meier_estimator(test["event"].to_numpy(dtype=bool), test["duration"].to_numpy(dtype=float))
    sigma_hat_train = rayleigh_mle_sigma(np_mod, train["duration"].to_numpy(), train["event"].to_numpy())

    valid_h = restrict_horizons_to_support(
        horizons, train["duration"].to_numpy(dtype=float), test["duration"].to_numpy(dtype=float)
    )
    times_eval = np_mod.array(sorted(valid_h)) if valid_h else np_mod.array([])

    models = {
        "KM_train": lambda t: step_survival_at(np_mod, km_train_t, km_train_s, t),
        "Rayleigh_30": lambda t: rayleigh_survival(np_mod, sigma_fixed, t),
        "Rayleigh_sigma_hat": lambda t: rayleigh_survival(np_mod, sigma_hat_train, t),
    }

    fold_metrics = {}
    n_test = len(test)
    for name, surv_fn in models.items():
        row = {"n_train": len(train), "n_test": n_test, "sigma_hat_train": sigma_hat_train}
        if len(times_eval) > 0:
            estimate = np_mod.tile(surv_fn(times_eval), (n_test, 1))
            try:
                _, bs_vals = brier_score(y_train, y_test, estimate, times_eval)
                ibs = float(integrated_brier_score(y_train, y_test, estimate, times_eval))
            except ValueError as exc:
                bs_vals, ibs = np_mod.full(len(times_eval), np_mod.nan), float("nan")
                row["ibs_error"] = str(exc)
            row["ibs"] = ibs
            for h, bs in zip(times_eval, bs_vals):
                row[f"brier_{h:g}y"] = float(bs)
        else:
            row["ibs"] = float("nan")
        if name.startswith("Rayleigh"):
            sigma_used = sigma_fixed if name == "Rayleigh_30" else sigma_hat_train
            row["log_score"] = float(np_mod.mean(rayleigh_log_likelihood(
                np_mod, sigma_used, test["duration"].to_numpy(), test["event"].to_numpy()
            )))
        fold_metrics[name] = row

    # --- error absoluto y n_risk (en TEST) por horizonte, contra la KM DE PRUEBA (fuera de muestra) ---
    horizon_rows = []
    for h in horizons:
        n_risk_test = int((test["duration"].to_numpy(dtype=float) >= h).sum())
        passes = n_risk_test >= n_risk_threshold and h in valid_h
        s_km_test = float(step_survival_at(np_mod, km_test_t, km_test_s, h)[0]) if h < test["duration"].max() else float("nan")
        s_r30 = float(rayleigh_survival(np_mod, sigma_fixed, h))
        s_rhat = float(rayleigh_survival(np_mod, sigma_hat_train, h))
        horizon_rows.append({
            "horizon": h, "n_risk_test": n_risk_test, "pasa_n_risk_200": passes,
            "S_KM_test": s_km_test, "S_Rayleigh_30": s_r30, "S_Rayleigh_sigma_hat": s_rhat,
            "abs_error_Rayleigh_30_vs_KM": abs(s_r30 - s_km_test) if passes else float("nan"),
            "abs_error_Rayleigh_sigma_hat_vs_KM": abs(s_rhat - s_km_test) if passes else float("nan"),
        })

    # --- error integrado vs KM de prueba, en una grilla fina dentro del soporte comun ---
    common_hi = min(train["duration"].max(), test["duration"].max())
    grid = np_mod.arange(0.5, common_hi, FINE_GRID_STEP_YEARS)
    integrated_error = {"KM_test_range_max": float(common_hi)}
    if len(grid) > 1:
        s_km_grid = step_survival_at(np_mod, km_test_t, km_test_s, grid)
        for name in ["Rayleigh_30", "Rayleigh_sigma_hat"]:
            sigma_used = sigma_fixed if name == "Rayleigh_30" else sigma_hat_train
            s_param_grid = rayleigh_survival(np_mod, sigma_used, grid)
            integrated_error[f"error_integrado_{name}_vs_KM_test"] = float(
                np_mod.trapezoid(np_mod.abs(s_param_grid - s_km_grid), grid) / (grid[-1] - grid[0])
            )

    return fold_metrics, horizon_rows, integrated_error


# ---------------------------------------------------------------------------
# Un draw completo: sigma-hat global + bootstrap espacial, y cross-fitting por bloques.
# ---------------------------------------------------------------------------


def run_draw(rt, draw_df, blocks, n_outer_folds, sigma_fixed, horizons, n_risk_threshold, n_bootstrap, seed):
    np_mod, GroupKFold = rt["np"], rt["GroupKFold"]

    duration = draw_df["duration"].to_numpy(dtype=float)
    event = draw_df["event"].to_numpy(dtype=bool)
    sigma_hat_global = rayleigh_mle_sigma(np_mod, duration, event)
    ci_lo, ci_hi, _ = spatial_block_bootstrap_sigma_ci(np_mod, duration, event, blocks, n_bootstrap, seed)

    gkf = GroupKFold(n_splits=min(n_outer_folds, len(np_mod.unique(blocks))))
    fold_metric_rows, horizon_rows_all, integrated_error_rows = [], [], []
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(draw_df, groups=blocks)):
        train, test = draw_df.iloc[train_idx], draw_df.iloc[test_idx]
        fold_metrics, horizon_rows, integrated_error = evaluate_fold(rt, train, test, sigma_fixed, horizons, n_risk_threshold)
        for model_name, m in fold_metrics.items():
            fold_metric_rows.append({"fold": fold_i, "model": model_name, **m})
        for hr in horizon_rows:
            horizon_rows_all.append({"fold": fold_i, **hr})
        integrated_error_rows.append({"fold": fold_i, **integrated_error})

    return {
        "sigma_hat_global": sigma_hat_global, "sigma_hat_ci_lo": ci_lo, "sigma_hat_ci_hi": ci_hi,
        "fold_metrics": fold_metric_rows, "horizon_metrics": horizon_rows_all,
        "integrated_error": integrated_error_rows,
    }


# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    np, pd, gpd = rt["np"], rt["pd"], rt["gpd"]

    print("M1/M2 - Modelos de calibracion: KM, Rayleigh-30, Rayleigh-sigma-hat (PLAN_V3_EJECUTABLE.md)")
    print("======================================================================================")

    data_path = args.survival_dir / f"dataset_k{args.k}_buffer{args.buffer_m}m.csv"
    df = load_evaluable_dataset(pd, data_path)
    print(f"Dataset: {data_path} -- {len(df):,} filas evaluables (soporte inicial + seguimiento observable).")

    draws = sorted(df["draw_id"].unique())
    if args.max_draws is not None:
        draws = draws[: args.max_draws]
        print(f"*** --max-draws={args.max_draws} activo (SOLO PRUEBAS): {len(draws)} draws. ***")
    print(f"Draws Monte Carlo a procesar: {len(draws)} (de {df['draw_id'].nunique()} disponibles en el dataset).")

    blocks_full = assign_spatial_blocks(gpd, np, pd, df["location_id"])
    print(f"Bloques espaciales de {SPATIAL_BLOCK_SIZE_M/1000:.0f} km: {len(set(blocks_full)):,} "
          f"(CRS {GRID_CRS_FOR_BLOCKS}, misma convencion que 02/06/09).")

    # --- chequeo OBLIGATORIO de la formula cerrada del MLE, antes de usarla miles de veces ---
    mle_check = verify_mle_matches_numerical_optimization(
        np, rt["minimize_scalar"], df["duration"].to_numpy(), df["event"].to_numpy(), args.random_seed
    )
    print(f"Verificacion MLE cerrado vs scipy.optimize: {mle_check}")
    if not mle_check["ok"]:
        raise SystemExit(f"El MLE en forma cerrada no coincide con la optimizacion numerica: {mle_check['ejemplos_falla']}")

    all_draw_results = {}
    for draw_id in draws:
        draw_mask = (df["draw_id"] == draw_id).to_numpy()
        draw_df = df.loc[draw_mask].reset_index(drop=True)
        draw_blocks = blocks_full[draw_mask]
        result = run_draw(
            rt, draw_df, draw_blocks, args.n_outer_folds, args.sigma_fixed, args.horizons,
            args.n_risk_threshold, args.n_bootstrap, args.random_seed + draw_id,
        )
        all_draw_results[draw_id] = result
        print(f"  draw {draw_id:>2}: sigma_hat={result['sigma_hat_global']:.3f} "
              f"IC95%=[{result['sigma_hat_ci_lo']:.3f}, {result['sigma_hat_ci_hi']:.3f}]")

    # --- consolidar ---
    sigma_rows = [
        {"draw_id": d, "sigma_hat": r["sigma_hat_global"], "ic95_lo": r["sigma_hat_ci_lo"], "ic95_hi": r["sigma_hat_ci_hi"]}
        for d, r in all_draw_results.items()
    ]
    sigma_table = pd.DataFrame(sigma_rows)

    fold_metrics_table = pd.concat(
        [pd.DataFrame(r["fold_metrics"]).assign(draw_id=d) for d, r in all_draw_results.items()], ignore_index=True
    )
    horizon_table = pd.concat(
        [pd.DataFrame(r["horizon_metrics"]).assign(draw_id=d) for d, r in all_draw_results.items()], ignore_index=True
    )
    integrated_error_table = pd.concat(
        [pd.DataFrame(r["integrated_error"]).assign(draw_id=d) for d, r in all_draw_results.items()], ignore_index=True
    )

    # --- resumen por modelo: media entre folds (por draw), luego media +/- DE entre draws ---
    per_draw_ibs = fold_metrics_table.groupby(["draw_id", "model"])["ibs"].mean().reset_index()
    ibs_summary = per_draw_ibs.groupby("model")["ibs"].agg(["mean", "std", "min", "max"]).reset_index()

    logscore_rows = fold_metrics_table.loc[fold_metrics_table["model"].str.startswith("Rayleigh")]
    per_draw_logscore = logscore_rows.groupby(["draw_id", "model"])["log_score"].mean().reset_index()
    logscore_summary = per_draw_logscore.groupby("model")["log_score"].agg(["mean", "std", "min", "max"]).reset_index()

    horizon_pass = horizon_table.groupby("horizon")["pasa_n_risk_200"].mean().reset_index()
    per_draw_abs_error = horizon_table.loc[horizon_table["pasa_n_risk_200"]].groupby(
        ["draw_id", "horizon"]
    )[["abs_error_Rayleigh_30_vs_KM", "abs_error_Rayleigh_sigma_hat_vs_KM"]].mean().reset_index()
    abs_error_summary = per_draw_abs_error.groupby("horizon")[
        ["abs_error_Rayleigh_30_vs_KM", "abs_error_Rayleigh_sigma_hat_vs_KM"]
    ].agg(["mean", "std"]).reset_index()

    per_draw_int_error = integrated_error_table.groupby("draw_id")[
        [c for c in integrated_error_table.columns if c.startswith("error_integrado")]
    ].mean().reset_index()
    int_error_summary = per_draw_int_error.drop(columns=["draw_id"]).agg(["mean", "std"]).T.reset_index()
    int_error_summary.columns = ["metrica", "mean", "std"]

    sigma_hat_mean = sigma_table["sigma_hat"].mean()
    sigma_hat_sd_between_draws = sigma_table["sigma_hat"].std()

    print("\n=== RESUMEN (media entre folds espaciales, luego media +/- DE entre 20 draws) ===")
    print("IBS por modelo:")
    print(ibs_summary.to_string(index=False))
    print("\nLog-score por modelo (solo Rayleigh):")
    print(logscore_summary.to_string(index=False))
    print(f"\nsigma_hat: media entre draws={sigma_hat_mean:.4f}, DE entre draws={sigma_hat_sd_between_draws:.4f}")
    print("Horizontes que pasan n_risk>=200 (fraccion de folds x draws):")
    print(horizon_pass.to_string(index=False))
    print("\nError absoluto vs KM de prueba, por horizonte (solo donde pasa n_risk>=200):")
    print(abs_error_summary.to_string(index=False))
    print("\nError integrado vs KM de prueba:")
    print(int_error_summary.to_string(index=False))

    # --- salidas ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sigma_table.to_csv(args.output_dir / f"sigma_hat_k{args.k}_buffer{args.buffer_m}m.csv", index=False)
    fold_metrics_table.to_csv(args.output_dir / f"fold_metrics_k{args.k}_buffer{args.buffer_m}m.csv", index=False)
    horizon_table.to_csv(args.output_dir / f"horizon_metrics_k{args.k}_buffer{args.buffer_m}m.csv", index=False)
    integrated_error_table.to_csv(args.output_dir / f"integrated_error_k{args.k}_buffer{args.buffer_m}m.csv", index=False)
    ibs_summary.to_csv(args.output_dir / f"ibs_summary_k{args.k}_buffer{args.buffer_m}m.csv", index=False)

    diagnostics = [
        "M1/M2 diagnostics - modelos de calibracion (PLAN_V3_EJECUTABLE.md)",
        "=====================================================================",
        f"Dataset: {data_path}. Filas evaluables: {len(df):,}. Draws: {len(draws)}. "
        f"Folds espaciales por draw: {args.n_outer_folds}. Bootstrap: {args.n_bootstrap}.",
        f"Bloques espaciales de {SPATIAL_BLOCK_SIZE_M/1000:.0f} km: {len(set(blocks_full)):,}.",
        f"Verificacion MLE cerrado vs numerico: {mle_check}",
        "",
        "*** SOLO TRES MODELOS (M1): KM, Rayleigh sigma=30 fijo (literature weight, sin "
        "reescalar), Rayleigh sigma estimado por verosimilitud censurada. Cox y boosting NO "
        "estan en este script -- eliminados de M1 por decision explicita del plan. ***",
        "*** NO se reporta C-index ni calibracion por deciles -- modelos poblacionales, esas "
        "metricas pertenecen a R3 (landmark model). ***",
        "",
        f"sigma_hat: media entre {len(draws)} draws = {sigma_hat_mean:.4f}, "
        f"DE entre draws = {sigma_hat_sd_between_draws:.4f} (variabilidad POSICIONAL, separada "
        "del IC espacial de bootstrap de bloque de cada draw individual).",
        "Tabla completa de sigma_hat por draw (con IC95% bootstrap de bloque espacial cada uno):",
        sigma_table.to_string(index=False),
        "",
        "IBS (Integrated Brier Score, restringido al soporte temporal observado, IPCW via "
        "scikit-survival) por modelo, media entre folds espaciales y entre draws:",
        ibs_summary.to_string(index=False),
        "",
        "Log-score (log-verosimilitud predictiva) -- solo modelos Rayleigh (KM no tiene densidad "
        "parametrica bien definida para esta metrica):",
        logscore_summary.to_string(index=False),
        "",
        f"Horizontes evaluados: {args.horizons} años. Umbral n_risk>=200 (fijado de antemano, M2). "
        "Fraccion de (fold x draw) en que cada horizonte pasa el umbral en el fold de TEST "
        "(no en el dataset completo -- criterio mas estricto):",
        horizon_pass.to_string(index=False),
        "",
        "Error absoluto |S_modelo(h) - S_KM_prueba(h)| en los horizontes que pasan n_risk>=200 "
        "(KM de prueba = ajustada SOLO con el fold de test, fuera de muestra respecto del ajuste "
        "de los modelos parametricos):",
        abs_error_summary.to_string(index=False),
        "",
        "Error integrado (area entre la curva parametrica y la KM de prueba, normalizada por el "
        "rango de la grilla, dentro del soporte comun train/test):",
        int_error_summary.to_string(index=False),
        "",
        "ADVERTENCIA DE ENTORNO: instalar scikit-survival para este script actualizo numpy/pandas/"
        "scikit-learn en el .venv_ai compartido -- ver bloque POR QUE. Verificado sin regresion en "
        "21_build_survival_dataset.py, pero declarado aca para quien re-corra otros scripts del "
        "proyecto despues de esta fecha.",
    ]
    (args.output_dir / f"diagnostics_k{args.k}_buffer{args.buffer_m}m.txt").write_text(
        "\n".join(str(l) for l in diagnostics) + "\n", encoding="utf-8"
    )
    print(f"\nGuardado: {args.output_dir / f'diagnostics_k{args.k}_buffer{args.buffer_m}m.txt'}")


if __name__ == "__main__":
    main()
