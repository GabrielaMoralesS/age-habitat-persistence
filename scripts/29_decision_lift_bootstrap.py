# CCAI OBSOLESCENCIA - B2b: IC95% PAREADO por bootstrap de bloque espacial para Delta-Lift@B
# (landmark_boosting - landmark_logistic), extendiendo el bootstrap ya usado para AUROC/AUPRC en
# 24_landmark_forecast.py (DIFF_PAIRS) a la capa de decision de 25_resurvey_decision.py, que hasta
# ahora solo tenia bootstrap espacial POR METODO (spatial_bootstrap_topb), nunca PAREADO entre
# metodos. NO reentrena boosting ni logistica: reusa los r_bar_i YA GUARDADOS por B2
# (results/decision/r_bar_landmark_{boosting,logistic}_{landmark}.csv) y solo reconstruye los
# labels Z del landmark de decision (build_landmark_dataset), que no depende de entrenar nada.
"""
POR QUE
-------
LA TABLA DE DECISION (tab:decision, Lift@100/500/1000) REPORTA UN NUMERO POR METODO, NUNCA UN IC DE
LA DIFERENCIA: 25_resurvey_decision.py ya bootstrapea CapturedEvents/Recall por bloque espacial
(spatial_bootstrap_topb), pero lo hace CON EL MISMO objeto rng REUSADO SECUENCIALMENTE entre
metodos -- el bucle `for method in DECISION_METHODS: ... spatial_bootstrap_topb(..., rng_boot, ...)`
avanza el estado del generador en cada llamada, asi que landmark_boosting y landmark_logistic
NUNCA ven la MISMA remuestra de bloques en la misma iteracion. Restar sus dos IC95% individuales
(lo que el texto actual hace informalmente al decir "boosting > logistic a 500/1000") es EXACTAMENTE
el error que 24_landmark_forecast.py ya identifico y evito para AUROC/AUPRC (ver su bloque POR QUE,
"DIFERENCIAS ENTRE METODOS: BOOTSTRAP PAREADO, NO DOS IC INDEPENDIENTES RESTADOS") -- dos intervalos
que se solapan no implican que la diferencia cruce cero, porque ignoran la correlacion entre ambos
metodos evaluados sobre la MISMA remuestra. Este script cierra esa brecha para Lift@B, reusando la
MISMA logica: una unica remuestra de bloques por iteracion, aplicada a AMBOS metodos, diferencia
calculada DENTRO de esa iteracion.

DENOMINADOR ALEATORIO POR ITERACION -- FORMA CERRADA, NO PERMUTATION NUEVA: dentro de cada remuestra
de bootstrap, el denominador de Lift@B (captura esperada bajo seleccion aleatoria) se calcula como
b_eff * media(Y-barra) SOBRE LA REMUESTRA, exactamente la misma formula cerrada que ya usa
tiebreak_averaged_topb en 25_resurvey_decision.py (E[bajo seleccion aleatoria] para muestreo simple
sin reemplazo) -- no se corren 1000 permutaciones nuevas por iteracion (séria 1000x1000, sin
beneficio marginal sobre la forma cerrada, que es exacta para este estimador). Como el denominador
depende SOLO del outcome remuestreado (Y-barra_i), no del metodo, es identico para boosting y
logistica DENTRO de la misma iteracion -- la unica fuente de diferencia entre lift_boosting y
lift_logistic en cada iteracion es el numerador (CapturedEvents@B bajo cada ranking), que es
exactamente lo que se quiere aislar.

OUTCOME: Y-BARRA_i (promedio entre los M draws), NO por-draw: misma eleccion que
spatial_bootstrap_topb y tiebreak_averaged_topb en 25_resurvey_decision.py -- cruzar bootstrap
espacial (1000 remuestras) con los 20 draws Monte Carlo por separado (1000x20) es computacionalmente
excesivo para el beneficio marginal, y mezclar las dos fuentes de incertidumbre en un unico eje
concentra el computo en la incertidumbre espacial, que es la que el bootstrap esta disenado para
capturar (ver bloque POR QUE de 25 sobre "DOS FUENTES DE INCERTIDUMBRE, REPORTADAS POR SEPARADO").

DESEMPATE: MISMA CLAVE SEEDED QUE LA TABLA PRINCIPAL (tiebreak_seed=4242 por defecto, identico al
default de 25_resurvey_decision.py) -- reusada tal cual, no regenerada, para que el ranking dentro
de cada remuestra sea consistente con el que ya produjo Lift@500/1000 en decision_localidades_2014.csv.

NO RECALCULA r_bar_i: landmark_boosting y landmark_logistic ya fueron entrenados y evaluados por
25_resurvey_decision.py el 2026-08-15 (rolling-origin + refit + prediccion sobre el landmark de
decision); sus r_bar_i estan en results/decision/r_bar_landmark_{boosting,logistic}_{landmark}.csv.
Reentrenar aca duplicaria compute sin cambiar el resultado (r_bar_i es deterministico dado el mismo
train_df y la misma seleccion de hiperparametros, ya fijada). Lo unico que este script SI reconstruye
es la tabla Z del landmark de decision (build_landmark_dataset), necesaria para el outcome
Y_i^(m) -- una funcion pura de (t, entries_by_location, loc_code_tables, sampled_codes_by_draw), sin
aleatoriedad ni dependencia de otros landmarks, asi que correrla aislada para un unico t reproduce
EXACTAMENTE las mismas filas Z que uso la corrida original de B2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
D1_PATH = SCRIPT_DIR / "20_trajectory_extract.py"
D2_PATH = SCRIPT_DIR / "21_build_survival_dataset.py"
M1_PATH = SCRIPT_DIR / "22_survival_models.py"
R3_PATH = SCRIPT_DIR / "24_landmark_forecast.py"
B2_PATH = SCRIPT_DIR / "25_resurvey_decision.py"

K_DEFAULT = 2
BUFFER_M_DEFAULT = 250
HORIZON_YEARS_DEFAULT = 5
DECISION_LANDMARK_DEFAULT = 2014
N_DRAWS_DEFAULT = 20
RANDOM_SEED_DEFAULT = 42
BUDGETS_DEFAULT = [500, 1000]
N_BOOTSTRAP_DEFAULT = 1000
TIEBREAK_SEED_DEFAULT = 4242
METHOD_A = "landmark_boosting"
METHOD_B = "landmark_logistic"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    obs_root = root
    parser = argparse.ArgumentParser(
        description="B2b - IC95% pareado por bootstrap de bloque espacial de Delta-Lift@B "
        "(landmark_boosting - landmark_logistic), reusando r_bar_i ya guardados por B2."
    )
    parser.add_argument("--occurrences-csv", type=Path, default=root / "data" / "processed" / "occurrences_with_cell_id.csv")
    parser.add_argument("--trajectory-dir", type=Path, default=obs_root / "results" / "trajectory")
    parser.add_argument("--survival-dir", type=Path, default=obs_root / "results" / "survival")
    parser.add_argument("--decision-dir", type=Path, default=obs_root / "results" / "decision",
                         help="Donde estan los r_bar_landmark_{boosting,logistic}_{landmark}.csv ya guardados por B2.")
    parser.add_argument("--output-dir", type=Path, default=obs_root / "results" / "decision")
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--buffer-m", type=int, default=BUFFER_M_DEFAULT)
    parser.add_argument("--horizon-years", type=int, default=HORIZON_YEARS_DEFAULT)
    parser.add_argument("--decision-landmark", type=int, default=DECISION_LANDMARK_DEFAULT)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    parser.add_argument("--max-draws", type=int, default=None, help="SOLO pruebas.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--budgets", type=int, nargs="+", default=BUDGETS_DEFAULT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--tiebreak-seed", type=int, default=TIEBREAK_SEED_DEFAULT)
    return parser.parse_args()


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
        import geopandas as gpd
    except ImportError as exc:
        missing = getattr(exc, "name", "a required package")
        raise SystemExit(
            f"Falta una dependencia: {missing}\nInstalar con:\n  pip install pandas numpy geopandas"
        ) from exc
    return {"np": np, "pd": pd, "gpd": gpd}


def load_module(path: Path, name: str):
    import importlib.util
    if not path.exists():
        raise SystemExit(f"No se encontro {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Bootstrap PAREADO: una unica remuestra de bloques por iteracion, aplicada a AMBOS metodos --
# ver bloque POR QUE.
# ---------------------------------------------------------------------------


def paired_spatial_bootstrap_lift_diff(np_mod, pd, gpd, r_a, r_b, outcome_mean_series, location_ids,
                                        budgets: list, n_bootstrap: int, rng, m1, tiebreak_key):
    blocks = m1.assign_spatial_blocks(gpd, np_mod, pd, pd.Series(location_ids))
    unique_blocks = np_mod.unique(blocks)
    block_to_idx = {b: np_mod.where(blocks == b)[0] for b in unique_blocks}
    ra_vals = r_a.reindex(location_ids).fillna(0.0).to_numpy()
    rb_vals = r_b.reindex(location_ids).fillna(0.0).to_numpy()
    y_vals = outcome_mean_series.reindex(location_ids).fillna(0.0).to_numpy()
    tb_vals = tiebreak_key.reindex(location_ids).to_numpy()
    results = {B: {"lift_a": [], "lift_b": [], "diff": []} for B in budgets}
    for _ in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np_mod.concatenate([block_to_idx[b] for b in sampled_blocks])
        ra_sub, rb_sub, y_sub, tb_sub = ra_vals[idx], rb_vals[idx], y_vals[idx], tb_vals[idx]
        order_a = np_mod.lexsort((tb_sub, -ra_sub))
        order_b = np_mod.lexsort((tb_sub, -rb_sub))
        y_mean_sub = float(y_sub.mean())
        for B in budgets:
            b_eff = min(B, len(idx))
            e_random = b_eff * y_mean_sub
            captured_a = float(y_sub[order_a[:b_eff]].sum())
            captured_b = float(y_sub[order_b[:b_eff]].sum())
            lift_a = captured_a / e_random if e_random > 0 else float("nan")
            lift_b = captured_b / e_random if e_random > 0 else float("nan")
            results[B]["lift_a"].append(lift_a)
            results[B]["lift_b"].append(lift_b)
            results[B]["diff"].append(lift_a - lift_b)
    return results


def summarize(np_mod, values: list) -> dict:
    arr = np_mod.array(values, dtype=float)
    arr = arr[~np_mod.isnan(arr)]
    if len(arr) == 0:
        return {"media": float("nan"), "sd": float("nan"), "p2.5": float("nan"), "p97.5": float("nan"), "n_validas": 0}
    return {
        "media": float(arr.mean()), "sd": float(arr.std()),
        "p2.5": float(np_mod.percentile(arr, 2.5)), "p97.5": float(np_mod.percentile(arr, 97.5)),
        "n_validas": int(len(arr)),
    }


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    np, pd, gpd = rt["np"], rt["pd"], rt["gpd"]

    print("B2b - IC95% pareado por bootstrap de bloque espacial de Delta-Lift@B (boosting - logistica)")
    print("===============================================================================")

    d1 = load_module(D1_PATH, "d1_trajectory_extract")
    d2 = load_module(D2_PATH, "d2_build_survival_dataset")
    m1 = load_module(M1_PATH, "m1_survival_models")
    r3 = load_module(R3_PATH, "r3_landmark_forecast")
    b2 = load_module(B2_PATH, "b2_resurvey_decision")

    meta = r3.json_load_metadata(args.trajectory_dir)
    year_start, year_max = meta["year_start"], meta["year_max"]
    print(f"Ventana: {year_start}-{year_max}. h={args.horizon_years}, k={args.k}. "
          f"Landmark de decision (t0) = {args.decision_landmark}.")

    df = d1.load_occurrences(pd, args.occurrences_csv, year_start)
    df = d1.filter_year_upper_bound(df, year_max, 3)
    print(f"Registros validos: {len(df):,}")
    entries_by_location = r3.build_entries_by_location(pd, df)
    print(f"Localidades unicas: {len(entries_by_location):,}")

    sampled_codes_by_draw, draws_available = r3.load_sampled_codes(
        pd, args.survival_dir, args.k, args.buffer_m, args.n_draws, args.max_draws
    )
    m_total = len(draws_available)
    print(f"Draws Monte Carlo reusados de D2/R3: {m_total}")

    combined = d1.load_buffer_trajectory(pd, args.trajectory_dir / ".chunks", args.buffer_m)
    if combined is None:
        raise SystemExit(f"Sin checkpoints de trayectoria para buffer {args.buffer_m}m.")
    loc_code_tables = d2.build_location_code_tables(pd, combined)

    print(f"\n--- Reconstruyendo SOLO los labels Z del landmark de decision {args.decision_landmark} "
          "(sin reentrenar boosting ni logistica) ---")
    decision_table, _ = r3.build_landmark_dataset(
        rt, d2, m1, args.decision_landmark, args.horizon_years, args.k, year_start, year_max,
        entries_by_location, loc_code_tables, sampled_codes_by_draw, r3.LOSS_WINDOW_YEARS,
    )
    if len(decision_table) == 0:
        raise SystemExit("Dataset del landmark de decision vacio.")
    print(f"Filas (localidad x draw en riesgo): {len(decision_table):,}. "
          f"Z=1: {int(decision_table['Z'].sum()):,} ({decision_table['Z'].mean():.2%}).")

    outcome_matrix = b2.build_outcome_matrix(pd, decision_table, draws_available)
    outcome_mean = outcome_matrix.mean(axis=1)
    location_ids = outcome_matrix.index.to_numpy()
    print(f"Universo de decision en t0={args.decision_landmark}: {len(outcome_matrix):,} localidades.")

    r_bar_path_a = args.decision_dir / f"r_bar_{METHOD_A}_{args.decision_landmark}.csv"
    r_bar_path_b = args.decision_dir / f"r_bar_{METHOD_B}_{args.decision_landmark}.csv"
    for p in (r_bar_path_a, r_bar_path_b):
        if not p.exists():
            raise SystemExit(f"No se encontro {p} -- correr 25_resurvey_decision.py primero.")
    r_a = pd.read_csv(r_bar_path_a).set_index("location_id")["r_bar"]
    r_b = pd.read_csv(r_bar_path_b).set_index("location_id")["r_bar"]
    print(f"r_bar_i cargados de {r_bar_path_a.name} y {r_bar_path_b.name} "
          f"(NO recalculados -- ver bloque POR QUE).")

    tiebreak_key = b2.generate_tiebreak_key(np, pd, outcome_matrix.index, args.tiebreak_seed)

    rng = np.random.default_rng(args.random_seed)
    print(f"\nBootstrap PAREADO de bloque espacial ({args.n_bootstrap} iteraciones, misma remuestra "
          f"de bloques aplicada a {METHOD_A} y {METHOD_B} en cada iteracion) para B en {args.budgets}...")
    boot = paired_spatial_bootstrap_lift_diff(
        np, pd, gpd, r_a, r_b, outcome_mean, location_ids, args.budgets, args.n_bootstrap, rng, m1, tiebreak_key
    )

    rows = []
    for B in args.budgets:
        lift_a_summary = summarize(np, boot[B]["lift_a"])
        lift_b_summary = summarize(np, boot[B]["lift_b"])
        diff_summary = summarize(np, boot[B]["diff"])
        crosses_zero = diff_summary["p2.5"] <= 0.0 <= diff_summary["p97.5"]
        rows.append({
            "B": B,
            "lift_landmark_boosting_bootstrap_media": lift_a_summary["media"],
            "lift_landmark_boosting_bootstrap_p2.5": lift_a_summary["p2.5"],
            "lift_landmark_boosting_bootstrap_p97.5": lift_a_summary["p97.5"],
            "lift_landmark_logistic_bootstrap_media": lift_b_summary["media"],
            "lift_landmark_logistic_bootstrap_p2.5": lift_b_summary["p2.5"],
            "lift_landmark_logistic_bootstrap_p97.5": lift_b_summary["p97.5"],
            "delta_lift_boosting_menos_logistic_media": diff_summary["media"],
            "delta_lift_ic95_lo": diff_summary["p2.5"],
            "delta_lift_ic95_hi": diff_summary["p97.5"],
            "ic95_cruza_cero": crosses_zero,
            "n_iteraciones_validas": diff_summary["n_validas"],
        })
    result_table = pd.DataFrame(rows)
    print("\n=== Delta-Lift@B pareado (landmark_boosting - landmark_logistic), IC95% bootstrap de bloque espacial ===")
    with pd.option_context("display.width", 160):
        print(result_table.to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f"lift_bootstrap_diff_boosting_vs_logistic_{args.decision_landmark}.csv"
    result_table.to_csv(out_csv, index=False)

    diagnostics = [
        "B2b diagnostics - IC95% pareado por bootstrap de bloque espacial de Delta-Lift@B",
        "===============================================================================",
        f"Landmark de decision (t0): {args.decision_landmark}. Comparacion: {METHOD_A} - {METHOD_B}.",
        f"Universo de decision: {len(outcome_matrix):,} localidades. Draws Monte Carlo (M): {m_total}.",
        f"r_bar_i reusados sin recalcular de: {r_bar_path_a.name}, {r_bar_path_b.name}.",
        f"Bootstrap: {args.n_bootstrap} iteraciones, bloques de 100 km (mismo CRS/tamano que "
        "22/24/25), remuestra UNICA por iteracion aplicada a ambos metodos (pareado -- ver bloque "
        "POR QUE), outcome Y-barra_i promediado entre los 20 draws Monte Carlo.",
        "Denominador de Lift@B por iteracion: b_eff * media(Y-barra) SOBRE LA REMUESTRA (forma "
        "cerrada, identica formula que tiebreak_averaged_topb en 25_resurvey_decision.py), NUNCA "
        "una permutation nueva -- identico entre los dos metodos dentro de la misma iteracion, asi "
        "que la unica fuente de la diferencia es el numerador (CapturedEvents@B bajo cada ranking).",
        f"Desempate: clave seeded fija (tiebreak_seed={args.tiebreak_seed}), identica a la usada en "
        "la tabla principal de 25_resurvey_decision.py.",
        "",
        result_table.to_string(index=False),
        "",
        "Lectura: un IC95% que NO cruza cero indica que la ventaja de boosting sobre logistica en "
        "Lift@B es consistente bajo la incertidumbre espacial (bloques de 100 km remuestreados con "
        "reemplazo), no solo un artefacto de dos intervalos individuales que casualmente no se "
        "solapan del todo.",
    ]
    diag_path = args.output_dir / f"diagnostics_lift_bootstrap_diff_{args.decision_landmark}.txt"
    diag_path.write_text("\n".join(str(l) for l in diagnostics) + "\n", encoding="utf-8")
    print(f"\nGuardado: {out_csv}")
    print(f"Guardado: {diag_path}")


if __name__ == "__main__":
    main()
