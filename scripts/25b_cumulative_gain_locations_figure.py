# CCAI OBSOLESCENCIA - B2 auxiliar: curva de ganancia acumulada a nivel de LOCALIDAD (no de fila
# localidad x draw), paper/figures/fig_cumulative_gain.pdf -- eje X en presupuesto ABSOLUTO de
# localidades (0-2000), no en porcentaje. Lee results/decision/r_bar_{method}_2014.csv (ya
# generado por 25_resurvey_decision.py, un score r_bar_i por localidad) y
# results/landmark/test_landmark_2014_predictions.csv (para el outcome Y-barra_i por localidad --
# ver bloque POR QUE). NO CONSULTA EARTH ENGINE, NO ENTRENA NINGUN MODELO NUEVO.
"""
POR QUE
-------
SUPERSEDE a 24c_cumulative_gain_figure.py (que rankeaba filas localidad x draw, 378.886 unidades,
eje X en % de presupuesto 0-100%): esa version pesaba cada localidad hasta 20 veces (una por draw)
y no correspondia a una decision real de reverificacion, que se toma por LOCALIDAD, no por
(localidad, draw). 25_resurvey_decision.py (B2) ya resuelve esto con el score r_bar_i =
(1/M)*sum_m I_im(t0)*valor_im, UN NUMERO POR LOCALIDAD (M=20 draws fijos en el denominador) --
exactamente la unidad de decision real ("si tuvieramos presupuesto para revisar B localidades").
Este script reusa esos r_bar_i YA CALCULADOS (results/decision/r_bar_{method}_2014.csv) en vez de
recalcularlos, y los ordena para una malla continua de presupuestos 0-2000 en vez de solo los tres
puntos {100,500,1000} que decision_localidades_2014.csv ya reporta -- la curva completa es lo que
pide la figura, esos tres puntos son casos particulares de la misma curva.

OUTCOME Y-BARRA_I, DERIVADO DE test_landmark_2014_predictions.csv, NO DE UN ARCHIVO NUEVO:
25_resurvey_decision.py calcula outcome_mean = outcome_matrix.mean(axis=1) internamente (lo usa en
spatial_bootstrap_topb y tiebreak_averaged_topb) pero nunca lo persiste en un CSV propio.
test_landmark_2014_predictions.csv (24_landmark_forecast.py) tiene las mismas filas
localidad x draw con el mismo Z (misma construccion, r3.build_landmark_dataset, mismos argumentos
t=2014/k/buffer/horizonte en ambos scripts -- verificado: mismas 32.073 localidades exactas en
ambas fuentes) -- Z.groupby(location_id).mean() reproduce Y-barra_i sin volver a correr el
pipeline completo de 25 (que reentrena un LightGBM y recorre ~135k localidades x 20 draws en
Python puro, costoso). No se recalcula ningun score aca, solo se lee.

"CAPTURED" EN UNIDADES ABSOLUTAS, NO PORCENTAJE: con el eje X en localidades absolutas, el eje Y
tambien queda en unidades absolutas de perdidas confirmadas capturadas (sum de Y-barra_i sobre el
top-B), la MISMA definicion de CapturedEvents@B que ya usa 25_resurvey_decision.py
(spatial_bootstrap_topb: `captured = float(y_sub[top_idx].sum())`) -- esta figura es la version
continua (B=1..2000) de esos mismos numeros, no una metrica nueva.

DIAGONAL DEL AZAR = B * media(Y-barra_i) SOBRE TODO EL UNIVERSO, no y=x: en unidades absolutas la
expectativa de seleccionar B localidades al azar (sin reemplazo) es B veces la tasa base del
universo completo, exactamente e_random de tiebreak_averaged_topb en 25_resurvey_decision.py.

BOOTSTRAP DE BLOQUE ESPACIAL: mismo patron que spatial_bootstrap_topb en 25_resurvey_decision.py
(reusa m1.assign_spatial_blocks, 100 km EPSG:5880) pero evaluado en la malla continua 1..2000 en
vez de solo B en {100,500,1000} -- en cada iteracion se ordena la remuestra UNA vez y se lee la
curva completa por indexado directo (sin interpolar: el universo bootstrapeado ronda las 32.000
localidades, muy por encima de 2000, asi que el rango pedido siempre cae dentro del soporte
observado en cada iteracion).

MAXIMO CUATRO CURVAS, PEDIDO EXPLICITO: landmark_boosting, perdida_reciente_ponderado (la variante
PONDERADA por I_im -- ver bloque POR QUE de 25_resurvey_decision.py sobre por que la variante sin
ponderar selecciona sitios ya casi sin bosque, un artefacto, no señal util), rayleigh_30, y la
diagonal del azar. Rayleigh_estimado, mas_antiguo (ambas variantes) y aleatorio quedan fuera de
esta figura por pedido explicito -- siguen disponibles en decision_localidades_2014.csv para quien
los necesite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
M1_PATH = SCRIPT_DIR / "22_survival_models.py"

DECISION_LANDMARK_DEFAULT = 2014
N_BOOTSTRAP_DEFAULT = 1000
RANDOM_SEED_DEFAULT = 42
MAX_BUDGET_DEFAULT = 2000
METHODS = ["landmark_boosting", "perdida_reciente_ponderado", "rayleigh_30"]
METHOD_LABELS = {
    "landmark_boosting": "Landmark boosting",
    "perdida_reciente_ponderado": "Weighted recent loss",
    "rayleigh_30": r"Rayleigh $\sigma=30$",
}
METHOD_COLORS = {
    "landmark_boosting": "#4c72b0",
    "perdida_reciente_ponderado": "#8172b2",
    "rayleigh_30": "#c44e52",
}


def parse_args() -> argparse.Namespace:
    obs_root = SCRIPT_DIR.parent
    parser = argparse.ArgumentParser(
        description="Figura B2 - curva de ganancia acumulada a nivel de localidad, presupuesto "
        "absoluto 0-2000, desde los rankings ya calculados por 25_resurvey_decision.py."
    )
    parser.add_argument("--decision-dir", type=Path, default=obs_root / "results" / "decision")
    parser.add_argument(
        "--predictions-csv", type=Path,
        default=obs_root / "results" / "landmark" / f"test_landmark_{DECISION_LANDMARK_DEFAULT}_predictions.csv",
    )
    parser.add_argument("--decision-landmark", type=int, default=DECISION_LANDMARK_DEFAULT)
    parser.add_argument("--max-budget", type=int, default=MAX_BUDGET_DEFAULT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--output-figure", type=Path, default=obs_root / "figures" / "fig_cumulative_gain.pdf")
    return parser.parse_args()


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        missing = getattr(exc, "name", "a required package")
        raise SystemExit(
            f"Falta una dependencia: {missing}\nInstalar con:\n"
            "  pip install pandas numpy geopandas matplotlib"
        ) from exc
    return {"np": np, "pd": pd, "gpd": gpd, "plt": plt}


def load_module(path: Path, name: str):
    import importlib.util
    if not path.exists():
        raise SystemExit(f"No se encontro {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_r_bar(pd, decision_dir: Path, method: str, landmark: int):
    path = decision_dir / f"r_bar_{method}_{landmark}.csv"
    if not path.exists():
        raise SystemExit(f"No se encontro {path} -- correr 25_resurvey_decision.py primero.")
    return pd.read_csv(path).set_index("location_id")["r_bar"]


def cumulative_capture(np_mod, r_bar_vals, y_vals, max_budget: int):
    """Ordena por r_bar descendente (desempate por orden estable) y devuelve la captura acumulada
    de Y en cada rango 1..min(max_budget, n)."""
    n = len(r_bar_vals)
    b_eff = min(max_budget, n)
    order = np_mod.argsort(-r_bar_vals, kind="stable")
    cum = np_mod.cumsum(y_vals[order].astype(float))
    out = np_mod.full(max_budget, np_mod.nan)
    out[:b_eff] = cum[:b_eff]
    return out


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    np, pd, gpd, plt = rt["np"], rt["pd"], rt["gpd"], rt["plt"]
    m1 = load_module(M1_PATH, "m1_survival_models")

    print("Figura B2 - curva de ganancia acumulada a nivel de localidad (presupuesto absoluto 0-2000)")
    print("================================================================================================")

    r_bar_by_method = {m: load_r_bar(pd, args.decision_dir, m, args.decision_landmark) for m in METHODS}
    print(f"Metodos: {METHODS}")
    for m, s in r_bar_by_method.items():
        print(f"  {m}: {len(s):,} localidades (results/decision/r_bar_{m}_{args.decision_landmark}.csv)")

    preds = pd.read_csv(args.predictions_csv, usecols=["location_id", "draw_id", "Z"])
    y_bar = preds.groupby("location_id")["Z"].mean()
    print(f"Outcome Y-barra_i (derivado de {args.predictions_csv}): {len(y_bar):,} localidades, "
          f"media={float(y_bar.mean()):.4%}.")

    universe = y_bar.index
    for m, s in r_bar_by_method.items():
        if set(s.index) != set(universe):
            raise SystemExit(f"Universo de localidades inconsistente entre r_bar_{m} y el outcome -- "
                              "revisar que ambos vengan del mismo landmark de decision.")
    print(f"Universo verificado: {len(universe):,} localidades identicas en las {len(METHODS)} fuentes de r_bar y el outcome.")

    budgets = np.arange(1, args.max_budget + 1)

    point_curves = {}
    for method, r_bar in r_bar_by_method.items():
        r_vals = r_bar.reindex(universe).to_numpy(dtype=float)
        y_vals = y_bar.reindex(universe).to_numpy(dtype=float)
        point_curves[method] = cumulative_capture(np, r_vals, y_vals, args.max_budget)

    universe_mean_y = float(y_bar.mean())
    chance_diagonal = budgets * universe_mean_y
    print(f"Diagonal del azar: pendiente = media(Y-barra_i) en todo el universo = {universe_mean_y:.6f} "
          f"perdidas/localidad.")

    print("\nBloques espaciales (100 km, EPSG:5880) para el bootstrap...")
    blocks = m1.assign_spatial_blocks(gpd, np, pd, pd.Series(universe))
    unique_blocks = np.unique(blocks)
    block_to_idx = {b: np.where(blocks == b)[0] for b in unique_blocks}
    print(f"Bloques unicos: {len(unique_blocks):,}.")

    r_arrays = {m: r_bar_by_method[m].reindex(universe).to_numpy(dtype=float) for m in METHODS}
    y_array = y_bar.reindex(universe).to_numpy(dtype=float)

    print(f"Bootstrap de bloque espacial ({args.n_bootstrap} iteraciones, semilla={args.random_seed})...")
    rng = np.random.default_rng(args.random_seed)
    boot_curves = {m: [] for m in METHODS}
    n_short = 0
    for _ in range(args.n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_to_idx[b] for b in sampled_blocks])
        if len(idx) < args.max_budget:
            n_short += 1
        y_sub = y_array[idx]
        for method in METHODS:
            r_sub = r_arrays[method][idx]
            boot_curves[method].append(cumulative_capture(np, r_sub, y_sub, args.max_budget))
    if n_short:
        print(f"AVISO: {n_short}/{args.n_bootstrap} iteraciones tuvieron menos de {args.max_budget:,} "
              "localidades remuestreadas (colas de la curva con NaN en esos tramos).")

    band_lo, band_hi = {}, {}
    for method in METHODS:
        arr = np.vstack(boot_curves[method])
        band_lo[method] = np.nanpercentile(arr, 2.5, axis=0)
        band_hi[method] = np.nanpercentile(arr, 97.5, axis=0)

    # --- datos ---
    data_out = pd.DataFrame({"budget_locations": budgets, "chance_diagonal": chance_diagonal})
    for method in METHODS:
        data_out[f"{method}_mean"] = point_curves[method]
        data_out[f"{method}_ci95_lo"] = band_lo[method]
        data_out[f"{method}_ci95_hi"] = band_hi[method]
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    data_out_path = args.output_figure.with_name(args.output_figure.stem + "_data.csv")
    data_out.to_csv(data_out_path, index=False, encoding="utf-8")
    print(f"\nGuardado (datos de la curva): {data_out_path}")

    # --- figura ---
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.plot(budgets, chance_diagonal, color="black", linestyle=":", linewidth=1.3, label="Chance baseline")
    for method in METHODS:
        color = METHOD_COLORS[method]
        ax.fill_between(budgets, band_lo[method], band_hi[method], color=color, alpha=0.18)
        ax.plot(budgets, point_curves[method], color=color, linewidth=1.9, label=METHOD_LABELS[method])
    ax.set_xlim(0, args.max_budget)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Resurvey budget (localities)", fontsize=10)
    ax.set_ylabel("Confirmed losses captured", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(args.output_figure)
    fig.savefig(args.output_figure.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Guardado (figura): {args.output_figure}")
    print(f"Guardado (figura, png): {args.output_figure.with_suffix('.png')}")

    diagnostics = [
        "Figura B2 diagnostics - curva de ganancia acumulada a nivel de localidad",
        "==============================================================================",
        f"Landmark de decision: {args.decision_landmark}. Universo: {len(universe):,} localidades.",
        f"Metodos: {METHODS} (maximo cuatro curvas + diagonal del azar, pedido explicito).",
        f"Presupuesto: 1-{args.max_budget:,} localidades (absoluto, no porcentaje).",
        f"Media(Y-barra_i) en el universo (pendiente de la diagonal del azar): {universe_mean_y:.6f}.",
        f"Bloques espaciales de 100 km: {len(unique_blocks):,}.",
        f"Bootstrap: {args.n_bootstrap} iteraciones, semilla={args.random_seed}.",
        "",
        "Captura a presupuesto de referencia (100, 500, 1000, 2000 localidades), media [IC95%]:",
    ]
    for b_ref in (100, 500, 1000, 2000):
        if b_ref > args.max_budget:
            continue
        i = b_ref - 1
        parts = [f"chance={chance_diagonal[i]:.2f}"]
        for method in METHODS:
            parts.append(
                f"{METHOD_LABELS[method]}={point_curves[method][i]:.2f} "
                f"[{band_lo[method][i]:.2f}, {band_hi[method][i]:.2f}]"
            )
        diagnostics.append(f"  B={b_ref}: " + "; ".join(parts))
    diag_path = args.output_figure.with_name(args.output_figure.stem + "_diagnostics.txt")
    diag_path.write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
    print(f"\nGuardado (diagnostics): {diag_path}")


if __name__ == "__main__":
    main()
