# CCAI OBSOLESCENCIA - O1f: tabla de sensibilidad al buffer, cohorte COMUN entre el criterio
# binario y la fraccion continua. Lee results/record_level_histogram/record_loss_derived_all_buffers.csv
# (O1e, muestra de 40.000 localidades x 4 buffers) -- NO toca results/record_level/ (O1d, 135.021
# localidades, subset "sin_sospechosas" de 267.576 registros). NO CONSULTA EARTH ENGINE, NO ENTRENA
# NINGUN MODELO.
"""
POR QUE
-------
El TODO dejado en paper/main.tex (seccion R1, parrafo "Cell-level estimates overstate...") marca el
problema explicitamente: el binario "26-fold, 1.8-47.7%" y el continuo "1.26-fold, 1.70-2.15%" que
el borrador citaba NO venian de la misma cohorte. El continuo si salia de O1e (40.000 localidades,
resultados/record_level_histogram/diagnostics.txt lo confirma: 0m=1.70%, 100m=1.95%, 250m=2.15%,
500m=2.15%). El binario, en cambio, salia del subset "sin_sospechosas" de O1d
(results/record_level/record_obsolescence_by_buffer.csv, 267.576 registros de 135.021 localidades
SIN filtrar coordenadas sospechosas de la misma forma que O1e): 0m=1.83%, 500m=47.71% -- coincide
exactamente con los numeros citados, y confirma la mezcla. Comparar un binario de una cohorte con
un continuo de OTRA cohorte de distinto tamaño invalida cualquier conclusion sobre cuanto varia cada
criterio con el radio del buffer -- una parte de la diferencia podria deberse a la cohorte, no al
criterio.

Este script recalcula AMBAS series (binario y continuo) desde la MISMA fuente -- O1e,
record_loss_derived_all_buffers.csv, exactamente las mismas 40.000 localidades en los cuatro
buffers (0, 100, 250, 500 m) -- para que la unica diferencia entre buffers sea el radio, no el
tamaño ni la composicion de la muestra.

BOOTSTRAP DE BLOQUE ESPACIAL, NO POR REGISTRO: muchos registros comparten localidad (varias
especies/años en el mismo punto), y localidades cercanas comparten historia de deforestacion --
remuestrear registros sueltos subestimaria el error estandar. Se reusa
assign_spatial_blocks de 22_survival_models.py (mismo CRS EPSG:5880, mismo tamaño de bloque de
100 km que 02/06/09/22/24) para que el bootstrap sea consistente con el resto de la tesis. Como las
CUATRO series de buffer comparten exactamente las mismas 40.000 localidades, el bloque de cada
localidad se calcula UNA sola vez y se reutiliza en los cuatro buffers -- no son cuatro particiones
espaciales distintas por casualidad.

FACTOR DE RANGO CON SU PROPIO IC95%: ademas del punto (max/min sobre los cuatro buffers observados),
cada iteracion del bootstrap recalcula las cuatro tasas y su propio max/min -- el IC95% resultante
del factor de rango hereda la correlacion entre buffers (todos remuestreados con el mismo sorteo de
bloques en cada iteracion), en vez de combinar por separado los IC95% de cada buffer (que series
error mas ancho e ignoraria la correlacion espacial compartida entre buffers).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
M1_PATH = SCRIPT_DIR / "22_survival_models.py"

BUFFERS_M = [0, 100, 250, 500]
N_BOOTSTRAP_DEFAULT = 1000
RANDOM_SEED_DEFAULT = 42


def parse_args() -> argparse.Namespace:
    obs_root = SCRIPT_DIR.parent
    parser = argparse.ArgumentParser(
        description="O1f - tabla de sensibilidad al buffer (binario y continuo) sobre la cohorte "
        "comun de O1e, con IC95% por bootstrap de bloque espacial."
    )
    parser.add_argument(
        "--input-csv", type=Path,
        default=obs_root / "results" / "record_level_histogram" / "record_loss_derived_all_buffers.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=obs_root / "results" / "record_level_histogram")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED_DEFAULT)
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
    if not path.exists():
        raise SystemExit(f"No se encontro {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_rates(pd, df, buffers_m: list) -> "pd.DataFrame":
    rows = []
    for buffer_m in buffers_m:
        sub = df.loc[df["buffer_m"] == buffer_m, "frac_loss_after_record"]
        rows.append({
            "buffer_m": buffer_m,
            "n_records": len(sub),
            "n_locations": df.loc[df["buffer_m"] == buffer_m, "loc_key"].nunique(),
            "binary_pct": float((sub > 0).mean()) * 100.0,
            "continuous_pct": float(sub.mean()) * 100.0,
        })
    return pd.DataFrame(rows)


def range_factor(series: "pd.Series") -> float:
    return float(series.max() / series.min())


def block_bootstrap(np_mod, pd, df, buffers_m: list, blocks_by_loc: dict, n_bootstrap: int, seed: int):
    """Remuestrea BLOQUES espaciales unicos con reemplazo (mismos bloques para los cuatro
    buffers, ver bloque POR QUE) y recalcula binary_pct/continuous_pct por buffer en cada
    iteracion, ademas del factor de rango de cada serie en esa misma iteracion."""
    rng = np_mod.random.default_rng(seed)
    unique_blocks = np_mod.array(sorted(set(blocks_by_loc.values())))

    # indices de fila por bloque, por buffer (una sola vez, reutilizados en cada iteracion)
    df = df.copy()
    df["block"] = df["loc_key"].map(blocks_by_loc)
    idx_by_block_per_buffer = {}
    frac_by_buffer = {}
    for buffer_m in buffers_m:
        sub = df.loc[df["buffer_m"] == buffer_m].reset_index(drop=True)
        frac_by_buffer[buffer_m] = sub["frac_loss_after_record"].to_numpy()
        idx_by_block_per_buffer[buffer_m] = {
            b: grp.index.to_numpy() for b, grp in sub.groupby("block")
        }

    draws = {buffer_m: {"binary": [], "continuous": []} for buffer_m in buffers_m}
    range_factor_draws = {"binary": [], "continuous": []}

    for _ in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        binary_vals, continuous_vals = {}, {}
        for buffer_m in buffers_m:
            idx_map = idx_by_block_per_buffer[buffer_m]
            idx = np_mod.concatenate([idx_map[b] for b in sampled_blocks if b in idx_map])
            frac = frac_by_buffer[buffer_m][idx]
            b_pct = float((frac > 0).mean()) * 100.0
            c_pct = float(frac.mean()) * 100.0
            draws[buffer_m]["binary"].append(b_pct)
            draws[buffer_m]["continuous"].append(c_pct)
            binary_vals[buffer_m] = b_pct
            continuous_vals[buffer_m] = c_pct
        range_factor_draws["binary"].append(max(binary_vals.values()) / min(binary_vals.values()))
        range_factor_draws["continuous"].append(max(continuous_vals.values()) / min(continuous_vals.values()))

    return draws, range_factor_draws


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    np, pd, gpd = rt["np"], rt["pd"], rt["gpd"]
    m1 = load_module(M1_PATH, "m1_survival_models")

    print("O1f - Sensibilidad al buffer, cohorte comun (O1e), binario + continuo, IC95% bootstrap de bloque")
    print("=========================================================================================")
    print(f"Entrada: {args.input_csv}")

    df = pd.read_csv(args.input_csv, usecols=["loc_key", "buffer_m", "frac_loss_after_record"])
    n_locs = df["loc_key"].nunique()
    print(f"Filas: {len(df):,}. Localidades unicas: {n_locs:,}. Buffers: {sorted(df['buffer_m'].unique())}")

    for buffer_m in BUFFERS_M:
        locs_here = df.loc[df["buffer_m"] == buffer_m, "loc_key"].nunique()
        if locs_here != n_locs:
            raise SystemExit(
                f"Cohorte NO comun: buffer {buffer_m}m tiene {locs_here:,} localidades, "
                f"se esperaban {n_locs:,} (las mismas en los cuatro buffers)."
            )
    print("Cohorte comun verificada: las mismas localidades en los cuatro buffers.")

    point = compute_rates(pd, df, BUFFERS_M)
    point["binary_range_factor"] = range_factor(point["binary_pct"])
    point["continuous_range_factor"] = range_factor(point["continuous_pct"])

    unique_locs = df["loc_key"].drop_duplicates()
    blocks_arr = m1.assign_spatial_blocks(gpd, np, pd, unique_locs)
    blocks_by_loc = dict(zip(unique_locs.to_numpy(), blocks_arr))
    n_blocks = len(set(blocks_by_loc.values()))
    print(f"Bloques espaciales de 100 km: {n_blocks:,} (compartidos por los cuatro buffers).")

    print(f"\nBootstrap de bloque espacial ({args.n_bootstrap} iteraciones, semilla={args.random_seed})...")
    draws, range_factor_draws = block_bootstrap(
        np, pd, df, BUFFERS_M, blocks_by_loc, args.n_bootstrap, args.random_seed
    )

    out_rows = []
    for _, row in point.iterrows():
        buffer_m = int(row["buffer_m"])
        b_arr = np.array(draws[buffer_m]["binary"])
        c_arr = np.array(draws[buffer_m]["continuous"])
        out_rows.append({
            "buffer_m": buffer_m,
            "n_records": int(row["n_records"]),
            "n_locations": int(row["n_locations"]),
            "binary_pct": row["binary_pct"],
            "binary_ci95_lo": float(np.percentile(b_arr, 2.5)),
            "binary_ci95_hi": float(np.percentile(b_arr, 97.5)),
            "continuous_pct": row["continuous_pct"],
            "continuous_ci95_lo": float(np.percentile(c_arr, 2.5)),
            "continuous_ci95_hi": float(np.percentile(c_arr, 97.5)),
        })
    out_table = pd.DataFrame(out_rows)
    out_path = args.output_dir / "buffer_sensitivity_common_cohort.csv"
    out_table.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nGuardado: {out_path}")
    print(out_table.to_string(index=False))

    rf_binary = np.array(range_factor_draws["binary"])
    rf_continuous = np.array(range_factor_draws["continuous"])
    range_factor_table = pd.DataFrame([
        {
            "serie": "binario (frac_loss_after_record > 0)",
            "buffer_min_pct": buffer_of_extreme(point, "binary_pct", "min"),
            "buffer_max_pct": buffer_of_extreme(point, "binary_pct", "max"),
            "range_factor": point["binary_range_factor"].iloc[0],
            "range_factor_ci95_lo": float(np.percentile(rf_binary, 2.5)),
            "range_factor_ci95_hi": float(np.percentile(rf_binary, 97.5)),
        },
        {
            "serie": "continuo (media frac_loss_after_record)",
            "buffer_min_pct": buffer_of_extreme(point, "continuous_pct", "min"),
            "buffer_max_pct": buffer_of_extreme(point, "continuous_pct", "max"),
            "range_factor": point["continuous_range_factor"].iloc[0],
            "range_factor_ci95_lo": float(np.percentile(rf_continuous, 2.5)),
            "range_factor_ci95_hi": float(np.percentile(rf_continuous, 97.5)),
        },
    ])
    range_factor_path = args.output_dir / "buffer_sensitivity_range_factor.csv"
    range_factor_table.to_csv(range_factor_path, index=False, encoding="utf-8")
    print(f"\nGuardado: {range_factor_path}")
    print(range_factor_table.to_string(index=False))

    # --- Bootstrap PAREADO de la diferencia entre los dos factores de rango, en escala log ---
    # rf_binary y rf_continuous vienen de la MISMA remuestra de bloques en cada iteracion (ver
    # block_bootstrap: ambas series se calculan dentro del mismo bucle, con el mismo sampled_blocks
    # de esa iteracion) -- la resta log(rf_binary) - log(rf_continuous) es por tanto una diferencia
    # PAREADA, no dos IC marginales restados. IC marginales que no se solapan ([23.1,47.7] vs
    # [1.05,1.75]) no son un test formal de la diferencia: ignoran la correlacion entre ambas series
    # (comparten la misma cohorte y los mismos bloques espaciales), asi que el IC marginal de la
    # resta tiende a ser mas ancho que el pareado. Escala log porque un factor de rango es una razon
    # (max/min): log(rf_binary/rf_continuous) = log(rf_binary) - log(rf_continuous) es la cantidad
    # simetrica y aditiva correcta para comparar dos razones, no su diferencia en escala lineal.
    log_diff = np.log(rf_binary) - np.log(rf_continuous)
    log_diff_media = float(np.mean(log_diff))
    log_diff_lo, log_diff_hi = float(np.percentile(log_diff, 2.5)), float(np.percentile(log_diff, 97.5))
    ratio_media = float(np.exp(log_diff_media))
    ratio_lo, ratio_hi = float(np.exp(log_diff_lo)), float(np.exp(log_diff_hi))
    paired_log_table = pd.DataFrame([{
        "comparacion": "factor_rango_binario / factor_rango_continuo",
        "log_diff_media": log_diff_media, "log_diff_ci95_lo": log_diff_lo, "log_diff_ci95_hi": log_diff_hi,
        "ratio_media": ratio_media, "ratio_ci95_lo": ratio_lo, "ratio_ci95_hi": ratio_hi,
        "excluye_1x_en_ratio": bool(ratio_lo > 1.0 or ratio_hi < 1.0),
    }])
    paired_log_path = args.output_dir / "buffer_sensitivity_range_factor_paired_log_bootstrap.csv"
    paired_log_table.to_csv(paired_log_path, index=False, encoding="utf-8")
    print(f"\nGuardado: {paired_log_path}")
    print(paired_log_table.to_string(index=False))

    diagnostics = [
        "O1f diagnostics - sensibilidad al buffer, cohorte comun (O1e)",
        "======================================================================",
        f"Entrada: {args.input_csv}",
        f"Localidades unicas (las mismas en los cuatro buffers): {n_locs:,}.",
        f"Bloques espaciales de 100 km (EPSG:5880, compartidos por los cuatro buffers): {n_blocks:,}.",
        f"Bootstrap: {args.n_bootstrap} iteraciones, semilla={args.random_seed}, remuestreo de BLOQUES "
        "unicos con reemplazo (nunca de registros sueltos).",
        "",
        "Tabla por buffer:",
        out_table.to_string(index=False),
        "",
        "Factor de rango (max/min entre los cuatro buffers), con IC95% que hereda la correlacion "
        "entre buffers (misma remuestra de bloques dentro de cada iteracion):",
        range_factor_table.to_string(index=False),
        "",
        "Bootstrap PAREADO (misma remuestra de bloques para ambas series en cada iteracion) de la "
        "diferencia entre los dos factores de rango, en escala log (log-razon, no diferencia lineal "
        "-- ver bloque POR QUE):",
        paired_log_table.to_string(index=False),
        f"  El binario es {ratio_media:.2f}x el continuo (IC95% [{ratio_lo:.2f}x, {ratio_hi:.2f}x])"
        + (" -- excluye 1x, diferencia formalmente confirmada." if paired_log_table['excluye_1x_en_ratio'].iloc[0]
           else " -- IC incluye 1x, la diferencia NO queda formalmente confirmada por este test pareado."),
        "",
        "NO mezclado con O1d (results/record_level/, 135.021 localidades, subset sin_sospechosas): "
        "los numeros previos del borrador del paper (binario 1.8-47.7%, 26-fold) salian de esa otra "
        "cohorte y quedan descartados por este script -- ver bloque POR QUE.",
    ]
    diag_path = args.output_dir / "buffer_sensitivity_common_cohort_diagnostics.txt"
    diag_path.write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
    print(f"\nGuardado: {diag_path}")


def buffer_of_extreme(point: "pd.DataFrame", col: str, which: str) -> int:
    idx = point[col].idxmin() if which == "min" else point[col].idxmax()
    return int(point.loc[idx, "buffer_m"])


if __name__ == "__main__":
    main()
