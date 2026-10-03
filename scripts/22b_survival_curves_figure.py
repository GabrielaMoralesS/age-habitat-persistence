# CCAI OBSOLESCENCIA - R2 auxiliar: figura de curvas de supervivencia completas (0-40 años),
# paper/figures/fig_survival_curves.pdf -- KM fuera-de-fold con banda de incertidumbre, Rayleigh
# sigma=30 fijo y Rayleigh sigma-hat=54.27. Lee results/survival/dataset_k{K}_buffer{B}m.csv (ya
# construido por 21_build_survival_dataset.py) y reusa funciones YA IMPLEMENTADAS y verificadas de
# 22_survival_models.py (assign_spatial_blocks, kaplan_meier_estimator, step_survival_at,
# rayleigh_survival). NO CONSULTA EARTH ENGINE, NO ENTRENA NINGUN MODELO NUEVO.
"""
POR QUE
-------
22_survival_models.py (M1/M2) ya calcula, dentro de evaluate_fold, la curva KM del fold de PRUEBA
(km_test_t, km_test_s) para cada combinacion (draw, fold espacial) -- pero solo la usa para evaluar
el error puntual en tres horizontes fijos (5, 10, 20 años) y el error integrado normalizado; nunca
persiste la curva completa. El plan pide la curva COMPLETA sobre una malla 0-40 años para la
figura del paper, no solo esos tres puntos. Este script reproduce EXACTAMENTE la misma particion
(GroupKFold por bloque espacial de 100 km, mismos bloques, mismo criterio "nunca por registro/año
por separado de su localidad") y el mismo kaplan_meier_estimator sobre el fold de prueba en cada
combinacion (draw x fold), pero evalua esa curva en una malla fina de 0 a 40 años en vez de en tres
puntos -- exactamente el mismo calculo no-parametrico que M1/M2 ya hace, solo con mas resolucion.
No es un modelo nuevo: KM es un estimador cerrado, no algo que se "entrena" con hiperparametros.

"FUERA DE FOLD" = curva KM ajustada SOLO con el fold de prueba de cada particion (nunca con el
fold de entrenamiento correspondiente), exactamente como ya hace evaluate_fold en 22_survival_models.py
para "KM de prueba". Con 20 draws x 5 folds espaciales = hasta 100 curvas de prueba independientes
(distintas localidades held-out cada vez), la banda de incertidumbre en la malla es la envolvente
2.5/97.5 percentil de esas ~100 curvas en cada punto de la malla -- mezcla incertidumbre POSICIONAL
(entre draws) y de particion espacial (entre folds), igual que el resto de la tesis nunca separa
esas dos fuentes salvo cuando se pide explicitamente (aca no se pidio).

SIGMA-HAT=54.27, NO RECALCULADO: es la media entre los 20 draws que 22_survival_models.py ya
calculo y guardo en results/calibration/sigma_hat_k{K}_buffer{B}m.csv (columna sigma_hat, ver
diagnostics_k2_buffer250m.txt: "sigma_hat: media entre 20 draws = 54.2700"). Este script LEE ese
numero en vez de reestimarlo, para no duplicar (con posible divergencia numerica) un calculo que
ya existe y esta verificado (MLE cerrado contra optimizacion numerica, ver
verify_mle_matches_numerical_optimization en 22_survival_models.py).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
M1_PATH = SCRIPT_DIR / "22_survival_models.py"

K_DEFAULT = 2
BUFFER_M_DEFAULT = 250
N_OUTER_FOLDS_DEFAULT = 5
N_DRAWS_DEFAULT = 20
SIGMA_FIXED_DEFAULT = 30.0
SIGMA_HAT_FALLBACK_DEFAULT = 54.27  # usado SOLO si no se encuentra sigma_hat_k{K}_buffer{B}m.csv
GRID_STEP_YEARS_DEFAULT = 0.25
T_MAX_YEARS_DEFAULT = 40.0


def parse_args() -> argparse.Namespace:
    obs_root = SCRIPT_DIR.parent
    parser = argparse.ArgumentParser(
        description="Figura R2 - curvas de supervivencia completas 0-40 años (KM fuera-de-fold + "
        "banda, Rayleigh-30, Rayleigh-sigma-hat)."
    )
    parser.add_argument("--survival-dir", type=Path, default=obs_root / "results" / "survival")
    parser.add_argument("--calibration-dir", type=Path, default=obs_root / "results" / "calibration")
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--buffer-m", type=int, default=BUFFER_M_DEFAULT)
    parser.add_argument("--n-outer-folds", type=int, default=N_OUTER_FOLDS_DEFAULT)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    parser.add_argument("--max-draws", type=int, default=None, help="SOLO pruebas.")
    parser.add_argument("--sigma-fixed", type=float, default=SIGMA_FIXED_DEFAULT)
    parser.add_argument("--sigma-hat-fallback", type=float, default=SIGMA_HAT_FALLBACK_DEFAULT)
    parser.add_argument("--grid-step-years", type=float, default=GRID_STEP_YEARS_DEFAULT)
    parser.add_argument("--t-max-years", type=float, default=T_MAX_YEARS_DEFAULT)
    parser.add_argument("--output-figure", type=Path, default=obs_root / "figures" / "fig_survival_curves.pdf")
    return parser.parse_args()


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        from sklearn.model_selection import GroupKFold
        from sksurv.nonparametric import kaplan_meier_estimator
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        missing = getattr(exc, "name", "a required package")
        raise SystemExit(
            f"Falta una dependencia: {missing}\nInstalar con:\n"
            "  pip install pandas numpy geopandas scikit-learn scikit-survival matplotlib"
        ) from exc
    return {
        "np": np, "pd": pd, "gpd": gpd, "GroupKFold": GroupKFold,
        "kaplan_meier_estimator": kaplan_meier_estimator, "plt": plt,
    }


def load_module(path: Path, name: str):
    if not path.exists():
        raise SystemExit(f"No se encontro {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sigma_hat_mean(pd, calibration_dir: Path, k: int, buffer_m: int, fallback: float) -> float:
    path = calibration_dir / f"sigma_hat_k{k}_buffer{buffer_m}m.csv"
    if not path.exists():
        print(f"AVISO: no se encontro {path}, usando --sigma-hat-fallback={fallback}.")
        return fallback
    df = pd.read_csv(path)
    return float(df["sigma_hat"].mean())


def out_of_fold_km_curves(rt, m1, df, blocks_full, n_outer_folds: int, grid):
    """Una curva KM por (draw, fold espacial de PRUEBA), evaluada en `grid`. Reusa exactamente
    assign_spatial_blocks/GroupKFold/kaplan_meier_estimator/step_survival_at de
    22_survival_models.py -- ver bloque POR QUE."""
    np_mod, pd, GroupKFold, kaplan_meier_estimator = rt["np"], rt["pd"], rt["GroupKFold"], rt["kaplan_meier_estimator"]
    curves = []
    n_folds_used = min(n_outer_folds, len(set(blocks_full)))
    gkf = GroupKFold(n_splits=n_folds_used)
    for _, test_idx in gkf.split(df, groups=blocks_full):
        test = df.iloc[test_idx]
        event = test["event"].to_numpy(dtype=bool)
        duration = test["duration"].to_numpy(dtype=float)
        if event.sum() == 0:
            continue
        km_t, km_s = kaplan_meier_estimator(event, duration)
        curves.append(m1.step_survival_at(np_mod, km_t, km_s, grid))
    return curves


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    np, pd, gpd, plt = rt["np"], rt["pd"], rt["gpd"], rt["plt"]
    m1 = load_module(M1_PATH, "m1_survival_models")

    print("Figura R2 - curvas de supervivencia completas 0-40 años")
    print("==========================================================")
    data_path = args.survival_dir / f"dataset_k{args.k}_buffer{args.buffer_m}m.csv"
    df_full = m1.load_evaluable_dataset(pd, data_path)
    print(f"Dataset: {data_path} -- {len(df_full):,} filas evaluables.")

    draws = sorted(df_full["draw_id"].unique())
    if args.max_draws is not None:
        draws = draws[: args.max_draws]
        print(f"*** --max-draws={args.max_draws} activo (SOLO PRUEBAS). ***")

    grid = np.arange(0.0, args.t_max_years + args.grid_step_years / 2, args.grid_step_years)
    grid = grid[grid <= args.t_max_years + 1e-9]
    print(f"Malla: {grid[0]:g}-{grid[-1]:g} años, paso {args.grid_step_years:g} ({len(grid)} puntos).")

    all_curves = []
    for draw_id in draws:
        draw_df = df_full.loc[df_full["draw_id"] == draw_id].reset_index(drop=True)
        blocks = m1.assign_spatial_blocks(gpd, np, pd, draw_df["location_id"])
        curves = out_of_fold_km_curves(rt, m1, draw_df, blocks, args.n_outer_folds, grid)
        all_curves.extend(curves)
    print(f"Curvas KM fuera-de-fold acumuladas: {len(all_curves)} ({len(draws)} draws x hasta "
          f"{args.n_outer_folds} folds espaciales cada uno).")

    curves_arr = np.vstack(all_curves)
    km_mean = curves_arr.mean(axis=0)
    km_lo = np.percentile(curves_arr, 2.5, axis=0)
    km_hi = np.percentile(curves_arr, 97.5, axis=0)

    sigma_hat_mean = load_sigma_hat_mean(pd, args.calibration_dir, args.k, args.buffer_m, args.sigma_hat_fallback)
    print(f"Rayleigh sigma=30 (fijo) y sigma_hat={sigma_hat_mean:.2f} (media entre draws, "
          f"results/calibration/sigma_hat_k{args.k}_buffer{args.buffer_m}m.csv).")
    rayleigh_30 = m1.rayleigh_survival(np, args.sigma_fixed, grid)
    rayleigh_sigma_hat = m1.rayleigh_survival(np, sigma_hat_mean, grid)

    curve_data = pd.DataFrame({
        "t_years": grid,
        "km_out_of_fold_mean": km_mean,
        "km_out_of_fold_ci95_lo": km_lo,
        "km_out_of_fold_ci95_hi": km_hi,
        "rayleigh_sigma30": rayleigh_30,
        "rayleigh_sigma_hat": rayleigh_sigma_hat,
    })
    data_out_path = args.output_figure.with_name(args.output_figure.stem + "_data.csv")
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    curve_data.to_csv(data_out_path, index=False, encoding="utf-8")
    print(f"\nGuardado (datos de la curva): {data_out_path}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.fill_between(grid, km_lo, km_hi, color="#4c72b0", alpha=0.20,
                     label="Out-of-fold KM, 95% CI (spatial block x draw)")
    ax.plot(grid, km_mean, color="#4c72b0", linewidth=2.0, label="Out-of-fold KM (mean)")
    ax.plot(grid, rayleigh_30, color="#c44e52", linestyle="--", linewidth=1.8,
             label=r"Rayleigh $\sigma=30$ (literature)")
    ax.plot(grid, rayleigh_sigma_hat, color="#55a868", linestyle="-.", linewidth=1.8,
             label=rf"Rayleigh $\hat\sigma={sigma_hat_mean:.2f}$ (estimated)")
    ax.set_xlim(0, args.t_max_years)
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Record age (years)", fontsize=10)
    ax.set_ylabel(r"Estimated forest support $S(t)$", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(args.output_figure)
    fig.savefig(args.output_figure.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Guardado (figura): {args.output_figure}")
    print(f"Guardado (figura, png): {args.output_figure.with_suffix('.png')}")

    diagnostics = [
        "Figura R2 diagnostics - curvas de supervivencia completas 0-40 años",
        "=======================================================================",
        f"Dataset: {data_path} ({len(df_full):,} filas evaluables).",
        f"Draws: {len(draws)}. Folds espaciales por draw: hasta {args.n_outer_folds}.",
        f"Curvas KM fuera-de-fold acumuladas: {len(all_curves)}.",
        f"Malla: 0-{args.t_max_years:g} años, paso {args.grid_step_years:g} ({len(grid)} puntos).",
        f"Rayleigh sigma fijo: {args.sigma_fixed:g}. Rayleigh sigma_hat: {sigma_hat_mean:.4f} "
        f"(media entre draws, leida de results/calibration/sigma_hat_k{args.k}_buffer{args.buffer_m}m.csv).",
        "",
        f"S(t) en horizontes de referencia:",
        curve_data.loc[curve_data["t_years"].isin([0, 5, 10, 20, 30, 40])].to_string(index=False),
    ]
    diag_path = args.output_figure.with_name(args.output_figure.stem + "_diagnostics.txt")
    diag_path.write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
    print(f"Guardado (diagnostics): {diag_path}")


if __name__ == "__main__":
    main()
