# CCAI OBSOLESCENCIA - R3 auxiliar: extrae del log de 24_landmark_forecast.py el desempeño POR
# LANDMARK de la busqueda rolling-origin (results/landmark/run_log.txt +
# rolling_origin_by_landmark.csv) y lo escribe como una tabla compacta,
# results/landmark/per_landmark_validation.csv. NO CONSULTA EARTH ENGINE, NO ENTRENA NINGUN MODELO
# NUEVO -- solo reorganiza numeros que 24_landmark_forecast.py ya calculo y guardo.
"""
POR QUE
-------
rolling_origin_by_landmark.csv (salida de rolling_origin_select_hyperparams en
24_landmark_forecast.py) tiene una fila por (combinacion de hiperparametros, landmark de
validacion) con SOLO el AUPRC -- ver la funcion, linea `split_scores.append({"landmark_val": ...,
"auprc": ...})`. n_filas y prevalencia de cada landmark de validacion no estan ahi: viven en las
lineas de texto libre que run_log.txt imprime al construir cada tabla de landmark ("Filas
(localidad x draw en riesgo): N. Z=1: n (p%)"). Este script junta ambas fuentes para la
combinacion de hiperparametros GANADORA (la que 24_landmark_forecast.py selecciono y uso para el
refit final) en una sola tabla por landmark de validacion.

AUROC NO ESTA DISPONIBLE Y NO SE RECALCULA: rolling_origin_select_hyperparams (linea 401 de
24_landmark_forecast.py) solo llama average_precision_score en cada particion -- roc_auc_score
nunca se calcula durante la busqueda rolling-origin, solo mas adelante sobre el test CONGELADO de
2014 (evaluate_method). Reconstruir un AUROC por landmark de validacion exigiria reentrenar el
LightGBM de cada particion (misma llamada a fit_predict_lgb, agregando roc_auc_score al resultado)
-- fuera de alcance de esta tarea, que pide explicitamente NO entrenar modelos nuevos. La columna
auroc_val queda como NaN con una nota explicita, en vez de inventar un numero o correr el
entrenamiento sin que la usuaria lo haya pedido.

AÑOS DE ENTRENAMIENTO Y HORIZONTE: se toman de la logica de particion documentada en el bloque POR
QUE de 24_landmark_forecast.py ("VALIDACION ROLLING-ORIGIN... {1993}->valida 1998,
{1993,1998}->valida 2003, {1993,1998,2003}->valida 2008") y de HORIZON_YEARS_DEFAULT=5 (constante
en el mismo script, confirmada en la primera linea de run_log.txt: "h=5, k=2"). No dependen de
ningun numero que haya que recalcular.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_LANDMARKS_DEFAULT = [1993, 1998, 2003, 2008]
HORIZON_YEARS_DEFAULT = 5

# Filas (localidad x draw en riesgo) y prevalencia por landmark, tal como
# 24_landmark_forecast.py las imprime al construir cada tabla (ver run_log.txt, seccion
# "Construyendo dataset del landmark ..."). Se listan aca literalmente en vez de re-parsear el
# log con una expresion regular fragil, porque el log ya esta congelado (correrlo de nuevo
# reentrenaria el modelo, fuera de alcance) -- ver bloque POR QUE.
LANDMARK_LOG_STATS = {
    1993: {"n_rows": 22037, "n_events": 647},
    1998: {"n_rows": 50383, "n_events": 1254},
    2003: {"n_rows": 95301, "n_events": 1647},
    2008: {"n_rows": 187934, "n_events": 2182},
    2014: {"n_rows": 378886, "n_events": 4023},  # test congelado, no es landmark de validacion
}


def parse_args() -> argparse.Namespace:
    obs_root = SCRIPT_DIR.parent
    landmark_dir = obs_root / "results" / "landmark"
    parser = argparse.ArgumentParser(
        description="Extrae el desempeño por landmark de validacion de la busqueda rolling-origin "
        "de 24_landmark_forecast.py en una tabla compacta."
    )
    parser.add_argument("--rolling-origin-csv", type=Path, default=landmark_dir / "rolling_origin_by_landmark.csv")
    parser.add_argument("--output-dir", type=Path, default=landmark_dir)
    parser.add_argument("--train-landmarks", type=int, nargs="+", default=TRAIN_LANDMARKS_DEFAULT)
    parser.add_argument("--horizon-years", type=int, default=HORIZON_YEARS_DEFAULT)
    return parser.parse_args()


def require_runtime():
    try:
        import pandas as pd
    except ImportError as exc:
        missing = getattr(exc, "name", "a required package")
        raise SystemExit(f"Falta una dependencia: {missing}\nInstalar con:\n  pip install pandas") from exc
    return {"pd": pd}


def best_params_string(pd, rolling_df: "pd.DataFrame") -> str:
    """La combinacion ganadora es la que 24_landmark_forecast.py selecciono: mayor AUPRC medio
    entre landmarks de validacion (misma logica que rolling_origin_select_hyperparams, linea 405:
    ordenar por mean_auprc_rolling descendente)."""
    means = rolling_df.groupby("params")["auprc"].mean().sort_values(ascending=False)
    return str(means.index[0])


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    pd = rt["pd"]

    print("R3 auxiliar - extraccion del desempeño por landmark de validacion (rolling-origin)")
    print("=====================================================================================")
    print(f"Entrada: {args.rolling_origin_csv}")

    rolling_df = pd.read_csv(args.rolling_origin_csv)
    best_params = best_params_string(pd, rolling_df)
    print(f"Combinacion de hiperparametros ganadora (mayor AUPRC medio rolling-origin): {best_params}")

    best_rows = rolling_df.loc[rolling_df["params"] == best_params].set_index("landmark_val")

    splits = [(args.train_landmarks[:i], args.train_landmarks[i]) for i in range(1, len(args.train_landmarks))]

    rows = []
    for train_lms, val_lm in splits:
        stats = LANDMARK_LOG_STATS[val_lm]
        n_rows = stats["n_rows"]
        n_events = stats["n_events"]
        prevalence_val = n_events / n_rows
        auprc_val = float(best_rows.loc[val_lm, "auprc"]) if val_lm in best_rows.index else float("nan")
        rows.append({
            "landmark_val": val_lm,
            "train_landmarks": ";".join(str(t) for t in train_lms),
            "horizon_years": args.horizon_years,
            "n_rows_val": n_rows,
            "n_events_val": n_events,
            "prevalence_val": prevalence_val,
            "auprc_val": auprc_val,
            # AUPRC de un ranking sin señal es aprox. igual a la prevalencia (ver bloque POR QUE de
            # 24_landmark_forecast.py) -- este cociente normaliza el AUPRC contra ese piso trivial,
            # asi que un valor de 4x se lee directamente como "4 veces mejor que un ranking al azar".
            "auprc_sobre_prevalencia": (auprc_val / prevalence_val) if prevalence_val > 0 else float("nan"),
            "auroc_val": float("nan"),
            "hyperparams": best_params,
        })

    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "per_landmark_validation.csv"
    table.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nGuardado: {out_path}")
    print(table.to_string(index=False))

    diagnostics = [
        "R3 auxiliar diagnostics - per_landmark_validation.csv",
        "=========================================================",
        f"Fuente: {args.rolling_origin_csv} (AUPRC) + run_log.txt / 24_landmark_forecast.py "
        "(n_filas, n_eventos por landmark, literal, ver LANDMARK_LOG_STATS en este script).",
        f"Hiperparametros ganadores (mayor AUPRC medio rolling-origin): {best_params}",
        "",
        table.to_string(index=False),
        "",
        "auprc_sobre_prevalencia = auprc_val / prevalence_val: normaliza el AUPRC contra el piso "
        "trivial de un ranking sin señal (que es aprox. igual a la prevalencia, no a 0.5 como el "
        "AUROC -- ver bloque POR QUE de 24_landmark_forecast.py). Un valor de 1.0 significa 'igual "
        "que al azar'; los tres landmarks de validacion quedan bien por encima de 1.0.",
        "",
        "auroc_val = NaN A PROPOSITO: rolling_origin_select_hyperparams en 24_landmark_forecast.py "
        "(linea ~401) solo calcula average_precision_score en cada particion de la busqueda "
        "rolling-origin, nunca roc_auc_score -- ese numero simplemente no existe en ningun log "
        "guardado. Calcularlo exigiria reentrenar el LightGBM de cada particion, fuera de alcance "
        "de esta tarea (que pide explicitamente no entrenar modelos nuevos). El AUROC SI esta "
        "disponible, pero solo para el test CONGELADO de 2014->2019 (no un landmark de "
        "validacion): ver results/landmark/test_landmark_2014_metrics.csv.",
    ]
    diag_path = args.output_dir / "per_landmark_validation_diagnostics.txt"
    diag_path.write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
    print(f"\nGuardado: {diag_path}")


if __name__ == "__main__":
    main()
