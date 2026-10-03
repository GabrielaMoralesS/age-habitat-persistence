# CCAI OBSOLESCENCIA - D2 (PLAN_V3_EJECUTABLE.md): dataset de supervivencia localidad-año de
# entrada, derivado EN PANDAS de los checkpoints de trayectoria de 20_trajectory_extract.py --
# SIN VOLVER A EARTH ENGINE, como el propio D1 promete ("De la trayectoria se deriva en pandas,
# sin volver a GEE"). Lee results/trajectory/.chunks/, results/trajectory/run_metadata.json y
# data/processed/occurrences_with_cell_id.csv (solo lectura). NO CONSULTA EARTH ENGINE.
"""
POR QUE
-------
D1 ya extrajo la trayectoria forestal COMPLETA (1985-Y_max, un bit por año) de las 135.021
ubicaciones en los buffers 250 y 500 m -- este script es donde esa trayectoria se convierte en un
dataset de supervivencia utilizable por 22_survival_models.py. Nada de esto vuelve a tocar Earth
Engine: todo lo que hace falta (estado forestal en cualquier año, primer evento persistente
condicionado a cualquier año de entrada, para cualquier k) ya esta en los codigos empaquetados de
D1, es aritmetica de bits sobre un entero.

UNIDAD: LOCALIDAD-AÑO DE ENTRADA, NUNCA ESPECIE-LOCALIDAD-AÑO. Varias especies colectadas en la
misma ubicacion el mismo año comparten EXACTAMENTE el mismo evento, la misma censura y la misma
incertidumbre posicional -- son la misma observacion de supervivencia repetida tantas veces como
especies se hayan visto ese dia. Incluirlas por separado es pseudorreplicacion: infla el tamaño de
muestra artificialmente y hace que las localidades muy muestreadas (mismo sitio, mismo año, 40
especies) dominen la verosimilitud sobre localidades vistas una sola vez. Por eso el primer paso
deduplica a (loc_key, year) UNICOS -- verificado con assert, no solo declarado -- antes de que la
especie vuelva a aparecer en ningun lado de este script.

ARITMETICA DE BITS DIRECTA SOBRE EL CODIGO, NO decode()+busqueda EN LISTA, POR VELOCIDAD --
VERIFICADO CONTRA decode(), NO SOLO ASUMIDO EQUIVALENTE: con ~135.021 localidades, hasta varios
años de entrada por localidad, 20 draws Monte Carlo y 3 valores de k, la ruta de calculo del
evento/censura se ejecuta varios millones de veces. decode() (ya verificado en el piloto de D1)
devuelve una lista de 40 bits y buscar en ella con un bucle Python por cada (localidad, año,
draw, k) es demasiado lento a esa escala. En su lugar, find_event_bit() opera directo sobre el
ENTERO del codigo con shifts y AND -- "los k bits desde la posicion t son todos cero" es
`(code >> t) & ((1<<k)-1) == 0`, una operacion de enteros nativa de Python, sin decodificar nada.
Esto NO se asume equivalente a decode(): verify_bit_arithmetic_matches_decode() corre ambos
caminos sobre una muestra aleatoria de (codigo, año de entrada, k) real de los datos y exige
coincidencia exacta antes de que el script continue -- el mismo criterio de "verificar con assert,
no declarar" que ya uso el piloto de D1 (encode(decode(code))==code) y 09_learned_record_
weighting.py (assert_no_block_overlap).

H_i, PERDIDA EN LOS 5 AÑOS PREVIOS Y TRANSICIONES SON AGREGADOS DETERMINISTAS, LOS DRAWS SON PARA
EVENTO/CENSURA -- distincion que el plan no hace explicita pero que importa: H_i(y_i) describe el
SITIO completo (que fraccion del buffer era bosque en el año de entrada), no una posicion
particular -- se calcula UNA vez por (localidad, año de entrada) ponderando TODOS los codigos
distintos de esa localidad por su n_px, sin muestrear nada. Lo mismo para "perdida en los 5 años
previos" y "frecuencia de transiciones": son covariables que describen el paisaje alrededor de la
localidad, no dependen de que pixel exacto haya sido el colectado. Los DRAWS Monte Carlo, en
cambio, existen exactamente para lo que D2 dice que resuelven: los estimadores de supervivencia
esperan UN tiempo y UN indicador de evento por observacion, no una distribucion -- así que para
CADA draw se muestrea un pixel (via su codigo, con probabilidad proporcional a n_px -- ver mas
abajo) y el evento/censura de ESE pixel especifico es lo que varia entre draws. Si las covariables
tambien variaran aleatoriamente por draw, un modelo entrenado sobre los 20 draws confundiria
incertidumbre posicional con variacion real de covariables -- exactamente el problema que D2
advierte para el ranking operacional, aplicado aca a los INSUMOS del modelo, no solo a su salida.

MUESTREAR CODIGO CON PROBABILIDAD PROPORCIONAL A n_px ES EXACTAMENTE q_i(p) UNIFORME, SIN
ENUMERAR PIXELES: D2 pide "muestrear una posicion verdadera segun q_i(p) uniforme". q_i(p) uniforme
sobre TODOS los pixeles del buffer es matematicamente identico a muestrear un CODIGO de trayectoria
con probabilidad proporcional a su conteo de pixeles n_px -- porque todos los pixeles que
comparten un mismo codigo son, por definicion, indistinguibles entre si para este proposito (misma
trayectoria forestal completa). Enumerar pixeles individuales seria mas caro y no cambiaria el
resultado: por eso este script muestrea directamente sobre la tabla (codigo, n_px) que D1 ya
produjo, sin reconstruir geometria de pixeles en ningun momento.

EL DRAW ES POR LOCALIDAD, NO POR LOCALIDAD-AÑO: la posicion fisica verdadera del sitio de colecta
NO cambia entre dos visitas a la misma localidad en años distintos -- es el mismo lugar. Por eso
un draw m fija UNA posicion (un codigo muestreado) POR LOCALIDAD, y esa misma posicion se reusa
para TODAS las entradas-año de esa localidad dentro del mismo draw -- no se vuelve a muestrear una
posicion nueva por cada año de entrada. Muestrear independientemente por año exageraria la
incertidumbre combinada de una localidad con varias entradas (cada año "vería" un sitio
potencialmente distinto dentro del mismo buffer, que no es lo que el modelo fisico dice).

CENSURA ADMINISTRATIVA L_k = Y_max-k+1, LEIDA DE run_metadata.json, NUNCA HARDCODEADA: Y_max no se
recalcula aca (este script no toca Earth Engine) ni se escribe como literal 2024 en ningun lado --
se lee del sidecar JSON que 20_trajectory_extract.py escribio en su misma corrida (el archivo de
diagnostics.txt es texto libre, no confiable para parsear una sola vez y usar en formulas). Con
k=2 y Y_max=2024, L_2=2023: un evento que EMPIEZA en 2024 no se puede confirmar (necesitaria ver
2025, que no existe en los datos). Entradas con año > L_k para un k dado no tienen seguimiento
observable para ESE k -- se excluyen de ese k especificamente (no de los otros), contadas aparte
en diagnostics.txt, nunca completadas con una duracion negativa o cero.

REGISTROS PRE-1985 EXCLUIDOS, MISMO FILTRO YA VERIFICADO EN D1: se reutiliza
load_occurrences()+filter_year_upper_bound() de 20_trajectory_extract.py via import dinamico
(mismo patron que 09_learned_record_weighting.py reutiliza 06_downstream_consequence.py, y por la
misma razon: esta logica ya esta escrita, verificada -- dio exactamente 135.021 ubicaciones,
491.228 registros -- y duplicarla aca a mano seria la misma logica dos veces con riesgo de que
diverjan con el tiempo).

SALIDAS SEPARADAS POR (k, buffer), SEIS ARCHIVOS -- asi lo pide D1 literalmente
("dataset_k{1,2,3}_buffer{250,500}m.csv"), aunque las columnas que NO dependen de k (H_i, año de
entrada, año/decada, n_entries_location, perdida_5y_previa, transiciones_previas, codigo
muestreado, soporte_forestal_inicial) se calculan UNA sola vez por (localidad, año, draw) y se
repiten identicas en los tres archivos de un mismo buffer -- solo evento/duracion cambian con k.
Evita triplicar el muestreo Monte Carlo (que es independiente de k) mientras respeta el formato de
archivo que el resto del plan (M1, B1, B2) espera.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
D1_PATH = SCRIPT_DIR / "20_trajectory_extract.py"

BUFFERS_DEFAULT = [250, 500]  # 250 principal, 500 sensibilidad (D1)
K_VALUES_DEFAULT = [2, 1, 3]  # k=2 principal, 1 y 3 derivados (orden de reporte, no de calculo)
N_DRAWS_DEFAULT = 20
RANDOM_SEED_DEFAULT = 42
YEAR_MAX_TOLERANCE_DEFAULT = 3  # misma convencion que 20_trajectory_extract.py
LOSS_WINDOW_YEARS_DEFAULT = 5  # "perdida en los 5 años previos" (D2, lista unica de covariables)
N_CROSSCHECK_SAMPLES = 500  # verificacion bit-aritmetica vs decode(), ver bloque POR QUE


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    obs_root = root
    parser = argparse.ArgumentParser(
        description="D2 - dataset de supervivencia localidad-año de entrada, derivado en pandas "
        "de la trayectoria de 20_trajectory_extract.py. Sin Earth Engine."
    )
    parser.add_argument("--occurrences-csv", type=Path, default=root / "data" / "processed" / "occurrences_with_cell_id.csv")
    parser.add_argument("--trajectory-dir", type=Path, default=obs_root / "results" / "trajectory",
                         help="Carpeta de 20_trajectory_extract.py -- debe contener .chunks/ y run_metadata.json.")
    parser.add_argument("--output-dir", type=Path, default=obs_root / "results" / "survival")
    parser.add_argument("--buffers-m", type=int, nargs="+", default=BUFFERS_DEFAULT)
    parser.add_argument("--k-values", type=int, nargs="+", default=K_VALUES_DEFAULT)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--year-max-tolerance", type=int, default=YEAR_MAX_TOLERANCE_DEFAULT)
    parser.add_argument("--loss-window-years", type=int, default=LOSS_WINDOW_YEARS_DEFAULT)
    parser.add_argument("--max-locations", type=int, default=None,
                         help="SOLO para pruebas: limitar a las primeras N ubicaciones (orden de "
                         "aparicion en el checkpoint). No usar para la corrida real del dataset.")
    return parser.parse_args()


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        missing = getattr(exc, "name", "a required package")
        raise SystemExit(f"Falta una dependencia: {missing}\nInstalar con:\n  pip install pandas numpy") from exc
    return {"np": np, "pd": pd}


def load_d1_module():
    """Carga 20_trajectory_extract.py como modulo -- mismo patron y misma razon que
    09_learned_record_weighting.py reutiliza 06_downstream_consequence.py: reusar load_occurrences,
    filter_year_upper_bound, load_buffer_trajectory, decode y encode YA VERIFICADOS, en vez de
    reimplementarlos (con el riesgo de que diverjan)."""
    if not D1_PATH.exists():
        raise SystemExit(f"No se encontro 20_trajectory_extract.py en {D1_PATH}")
    spec = importlib.util.spec_from_file_location("d1_trajectory_extract", D1_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run_metadata(path: Path) -> dict:
    meta_path = path / "run_metadata.json"
    if not meta_path.exists():
        raise SystemExit(
            f"No se encontro {meta_path} -- 20_trajectory_extract.py debe correr (y terminar) "
            "antes de este script. year_start/year_max NO se hardcodean aca."
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Paso 1: unidad localidad-año de entrada, deduplicada, con assert explicito.
# ---------------------------------------------------------------------------


def build_location_entry_years(pd, occurrences):
    unique_loc_years = occurrences[["loc_key", "year"]].drop_duplicates().rename(
        columns={"loc_key": "location_id", "year": "entry_year"}
    ).reset_index(drop=True)
    dup_mask = unique_loc_years.duplicated(subset=["location_id", "entry_year"])
    assert not dup_mask.any(), (
        f"Duplicados de localidad-año tras drop_duplicates() -- bug interno, no deberia ser "
        f"posible: {int(dup_mask.sum())} filas."
    )
    n_entries = unique_loc_years.groupby("location_id")["entry_year"].transform("size")
    unique_loc_years["n_entries_location"] = n_entries
    return unique_loc_years


# ---------------------------------------------------------------------------
# Paso 2: aritmetica de bits directa sobre el codigo -- ver bloque POR QUE.
# ---------------------------------------------------------------------------


def bit_at(code: int, idx: int) -> int:
    return (code >> idx) & 1


def find_event_bit(code: int, entry_idx: int, k: int, l_k_idx: int):
    """event, duration usando aritmetica de enteros -- ver bloque POR QUE. entry_idx/l_k_idx son
    OFFSETS desde year_start (indice de bit), no años calendario. Devuelve (None, None) si
    entry_idx > l_k_idx (sin seguimiento observable y confirmable para este k)."""
    if entry_idx > l_k_idx:
        return None, None
    k_mask = (1 << k) - 1
    for t in range(entry_idx + 1, l_k_idx + 1):
        if (code >> t) & k_mask == 0:  # k bits consecutivos en 0 = k años consecutivos NO forestales
            return 1, t - entry_idx
    return 0, l_k_idx - entry_idx


def find_event_slow_via_decode(decode_fn, code: int, entry_idx: int, k: int, l_k_idx: int, n_years: int):
    """Camino DE REFERENCIA usando decode() ya verificado en el piloto de D1 -- solo para
    verify_bit_arithmetic_matches_decode(), nunca en la ruta de calculo real (demasiado lento a
    escala completa, ver bloque POR QUE)."""
    if entry_idx > l_k_idx:
        return None, None
    bits = decode_fn(code, n_years)
    for t in range(entry_idx + 1, l_k_idx + 1):
        window = bits[t : t + k]
        if len(window) == k and all(b == 0 for b in window):
            return 1, t - entry_idx
    return 0, l_k_idx - entry_idx


def verify_bit_arithmetic_matches_decode(np_mod, decode_fn, all_codes: list, n_years: int, entry_idx_max: int, k_values: list, n_samples: int, seed: int) -> dict:
    """Chequeo OBLIGATORIO antes de confiar en find_event_bit() a escala completa -- ver bloque
    POR QUE. Muestrea (codigo, entry_idx, k) real de los datos y exige que la aritmetica de bits
    y decode()+busqueda den EXACTAMENTE el mismo (event, duration) en los tres casos: evento
    confirmado, censurado, y sin seguimiento observable."""
    rng = np_mod.random.default_rng(seed)
    if not all_codes:
        return {"n_verificados": 0, "n_fallas": 0, "ok": False, "motivo": "sin codigos para verificar"}
    codes_sample = rng.choice(all_codes, size=min(n_samples, len(all_codes)), replace=True)
    entry_idx_sample = rng.integers(0, entry_idx_max + 1, size=len(codes_sample))
    fails = []
    for code, entry_idx in zip(codes_sample, entry_idx_sample):
        code, entry_idx = int(code), int(entry_idx)
        for k in k_values:
            l_k_idx = (n_years - 1) - k + 1
            fast = find_event_bit(code, entry_idx, k, l_k_idx)
            slow = find_event_slow_via_decode(decode_fn, code, entry_idx, k, l_k_idx, n_years)
            if fast != slow:
                fails.append((code, entry_idx, k, fast, slow))
    return {
        "n_verificados": len(codes_sample) * len(k_values), "n_fallas": len(fails),
        "ejemplos_falla": fails[:10], "ok": len(fails) == 0,
    }


def frac_nonforest_before(code: int, entry_idx: int, window: int) -> float:
    """Fraccion NO forestal en los `window` años ANTES de entry_idx (bloque POR QUE: covariable
    AGREGADA, no depende del draw). NaN si entry_idx esta en year_start o antes (no hay historia
    previa observable en la ventana 1985-Y_max)."""
    lo = max(0, entry_idx - window)
    n = entry_idx - lo
    if n <= 0:
        return float("nan")
    mask = (1 << n) - 1
    segment = (code >> lo) & mask
    n_forest = bin(segment).count("1")
    return 1.0 - n_forest / n


def n_transitions_before(code: int, entry_idx: int) -> float:
    """Cantidad de transiciones bosque<->no-bosque en TODA la historia antes de entry_idx (bloque
    POR QUE: covariable agregada). NaN si entry_idx==0 (sin historia previa)."""
    if entry_idx <= 0:
        return float("nan")
    mask = (1 << entry_idx) - 1
    segment = (code >> 0) & mask  # bits [0, entry_idx)
    xor = segment ^ (segment >> 1)
    xor &= (1 << (entry_idx - 1)) - 1 if entry_idx >= 1 else 0
    return float(bin(xor).count("1"))


# ---------------------------------------------------------------------------
# Paso 3: por localidad -- H_i/covariables agregadas (deterministas) + draws Monte Carlo (evento).
# ---------------------------------------------------------------------------


def build_location_code_tables(pd, combined_buffer_df):
    """Agrupa (codigo, n_px) por localidad -- una sola vez por buffer, reusado para todas las
    entradas-año y todos los draws de esa localidad."""
    tables = {}
    for loc_key, group in combined_buffer_df.groupby("loc_key"):
        codes = group["traj_code"].to_numpy()
        weights = group["n_px"].to_numpy(dtype=float)
        tables[loc_key] = (codes, weights)
    return tables


def process_buffer(pd, np_mod, d1_mod, buffer_m, combined_buffer_df, loc_years, year_start, year_max, k_values, n_draws, seed, loss_window):
    n_years = year_max - year_start + 1
    l_k_idx = {k: (n_years - 1) - k + 1 for k in k_values}  # offset del ultimo año iniciable y confirmable

    loc_tables = build_location_code_tables(pd, combined_buffer_df)
    locations_with_trajectory = set(loc_tables.keys())
    locations_needed = set(loc_years["location_id"].unique())
    missing = locations_needed - locations_with_trajectory
    if missing:
        print(f"  AVISO buffer {buffer_m}m: {len(missing):,} localidades sin trayectoria en los "
              f"checkpoints (de {len(locations_needed):,} necesarias) -- se excluyen de este buffer.")

    all_codes_flat = [int(c) for codes, _ in loc_tables.values() for c in codes]
    crosscheck = verify_bit_arithmetic_matches_decode(
        np_mod, d1_mod.decode, all_codes_flat, n_years, n_years - 1, k_values, N_CROSSCHECK_SAMPLES, seed
    )
    print(f"  Verificacion bit-aritmetica vs decode() (buffer {buffer_m}m): "
          f"{crosscheck['n_verificados']} comparaciones, {crosscheck['n_fallas']} fallas -- "
          f"{'OK' if crosscheck['ok'] else '*** FALLO ***'}")
    if not crosscheck["ok"]:
        raise SystemExit(
            f"find_event_bit() no coincide con decode() -- buffer {buffer_m}m: {crosscheck['ejemplos_falla']}. "
            "No se construye el dataset con esta discrepancia sin resolver."
        )

    # --- H_i y covariables agregadas: UNA vez por (localidad, año de entrada) ---
    h_rows = []
    for row in loc_years.itertuples():
        loc_key = row.location_id
        if loc_key not in loc_tables:
            continue
        codes, weights = loc_tables[loc_key]
        entry_idx = row.entry_year - year_start
        total_w = 0.0
        forest_w = 0.0
        loss5_num, loss5_w = 0.0, 0.0
        trans_num, trans_w = 0.0, 0.0
        for c, w in zip(codes, weights):
            c = int(c)
            total_w += w
            if bit_at(c, entry_idx):
                forest_w += w
            loss5_val = frac_nonforest_before(c, entry_idx, loss_window)
            if not np_mod.isnan(loss5_val):
                loss5_num += loss5_val * w
                loss5_w += w
            trans_val = n_transitions_before(c, entry_idx)
            if not np_mod.isnan(trans_val):
                trans_num += trans_val * w
                trans_w += w
        h_i = forest_w / total_w if total_w > 0 else float("nan")
        loss5, trans = loss5_num, trans_num
        h_rows.append({
            "location_id": loc_key, "entry_year": row.entry_year, "decade_entry": (row.entry_year // 10) * 10,
            "n_entries_location": row.n_entries_location,
            "H_i": h_i, "n_px_valid_total": total_w,
            "frac_nonforest_last_5y": (loss5 / loss5_w) if loss5_w > 0 else float("nan"),
            "n_transitions_before_entry": (trans / trans_w) if trans_w > 0 else float("nan"),
        })
    h_table = pd.DataFrame(h_rows)

    # --- Draws Monte Carlo: UNO por localidad (misma posicion para todas sus entradas-año), ver POR QUE ---
    # entries_by_location precalculado UNA vez (groupby), no un filtro de loc_years POR localidad
    # dentro del bucle -- eso seria O(n_localidades x n_filas_loc_years), inviable a 135.021.
    entries_by_location = loc_years.groupby("location_id")["entry_year"].apply(list).to_dict()
    rng = np_mod.random.default_rng(seed)
    draw_records = []
    locations_sorted = sorted(loc_tables.keys())  # orden fijo -- reproducibilidad del rng
    for loc_key in locations_sorted:
        entries_this_loc = entries_by_location.get(loc_key)
        if not entries_this_loc:
            continue
        codes, weights = loc_tables[loc_key]
        p = weights / weights.sum()
        sampled_codes = rng.choice(codes, size=n_draws, p=p)
        for draw_id, code in enumerate(sampled_codes, start=1):
            code = int(code)
            for entry_year in entries_this_loc:
                entry_idx = int(entry_year) - year_start
                initial_support = bit_at(code, entry_idx)
                rec = {
                    "location_id": loc_key, "entry_year": int(entry_year), "draw_id": draw_id,
                    "sampled_traj_code": code, "initial_forest_support": initial_support,
                }
                for k in k_values:
                    if initial_support:
                        event, duration = find_event_bit(code, entry_idx, k, l_k_idx[k])
                    else:
                        event, duration = None, None  # sin soporte inicial -- fuera del analisis de supervivencia (D2)
                    rec[f"event_k{k}"] = event
                    rec[f"duration_k{k}"] = duration
                draw_records.append(rec)
    draws_table = pd.DataFrame(draw_records)

    dup_mask = draws_table.duplicated(subset=["location_id", "entry_year", "draw_id"])
    assert not dup_mask.any(), f"Duplicados de (localidad, año, draw) -- bug interno: {int(dup_mask.sum())} filas."

    full = draws_table.merge(h_table, on=["location_id", "entry_year"], how="left")
    return full, h_table, crosscheck, missing


# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    np, pd = rt["np"], rt["pd"]
    d1 = load_d1_module()

    print("D2 - Dataset de supervivencia localidad-año de entrada (PLAN_V3_EJECUTABLE.md)")
    print("====================================================================================")

    meta = load_run_metadata(args.trajectory_dir)
    year_start, year_max = meta["year_start"], meta["year_max"]
    print(f"Ventana (de run_metadata.json, NO hardcodeada): {year_start}-{year_max} "
          f"({year_max - year_start + 1} años). Asset: {meta['mapbiomas_asset']}.")
    if not meta.get("es_corrida_completa", False):
        print("*** AVISO: run_metadata.json indica que la extraccion de D1 NO fue la corrida "
              "completa (135.021 ubicaciones) -- este dataset heredaria esa limitacion. ***")

    df = d1.load_occurrences(pd, args.occurrences_csv, year_start)
    df = d1.filter_year_upper_bound(df, year_max, args.year_max_tolerance)
    print(f"Registros validos (year en [{year_start}, {year_max + args.year_max_tolerance}]): {len(df):,}")

    loc_years = build_location_entry_years(pd, df)
    print(f"Localidades-año de entrada UNICAS (assert de no-duplicados OK): {len(loc_years):,} "
          f"({loc_years['location_id'].nunique():,} localidades distintas)")
    multi = (loc_years["n_entries_location"] > 1).sum()
    print(f"  Localidades con mas de una entrada-año: {loc_years.loc[loc_years['n_entries_location']>1,'location_id'].nunique():,} "
          f"({multi:,} filas de esas localidades)")

    if args.max_locations is not None:
        keep_locs = loc_years["location_id"].drop_duplicates().head(args.max_locations)
        loc_years = loc_years.loc[loc_years["location_id"].isin(keep_locs)].reset_index(drop=True)
        print(f"*** --max-locations={args.max_locations} activo (SOLO PRUEBAS): {len(loc_years):,} filas. ***")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = [
        "D2 diagnostics - dataset de supervivencia localidad-año de entrada (PLAN_V3_EJECUTABLE.md)",
        "========================================================================================",
        f"Ventana: {year_start}-{year_max} (de run_metadata.json). k evaluados: {args.k_values}. "
        f"n_draws Monte Carlo: {args.n_draws}. seed: {args.random_seed}.",
        f"Registros validos: {len(df):,}. Localidades-año unicas: {len(loc_years):,} "
        f"({loc_years['location_id'].nunique():,} localidades).",
        "",
    ]

    for buffer_m in args.buffers_m:
        print(f"\n--- Buffer {buffer_m} m ---")
        combined = d1.load_buffer_trajectory(pd, args.trajectory_dir / ".chunks", buffer_m)
        if combined is None:
            print(f"  Sin checkpoints para buffer {buffer_m}m -- se omite.")
            continue
        full, h_table, crosscheck, missing = process_buffer(
            pd, np, d1, buffer_m, combined, loc_years, year_start, year_max,
            args.k_values, args.n_draws, args.random_seed, args.loss_window_years,
        )

        h_valid = h_table["H_i"].dropna()
        print(f"  H_i: media={h_valid.mean():.4f} mediana={h_valid.median():.4f} "
              f"(n={len(h_valid):,}); localidades con H_i=0 (sin soporte forestal inicial): "
              f"{(h_valid == 0).sum():,} ({(h_valid==0).mean():.1%})")

        diagnostics.append(f"BUFFER {buffer_m} m")
        diagnostics.append(f"  Localidades sin trayectoria en checkpoint: {len(missing):,}")
        diagnostics.append(f"  Verificacion bit-aritmetica vs decode(): {crosscheck}")
        diagnostics.append(f"  H_i -- media={h_valid.mean():.4f} mediana={h_valid.median():.4f} "
                            f"min={h_valid.min():.4f} max={h_valid.max():.4f} n={len(h_valid):,}")
        diagnostics.append(f"  Localidades-año con H_i=0 (sin soporte forestal inicial): "
                            f"{(h_valid==0).sum():,} ({(h_valid==0).mean():.1%})")

        for k in args.k_values:
            ev_col, du_col = f"event_k{k}", f"duration_k{k}"
            sub = full[[ev_col, du_col, "initial_forest_support"]].copy()
            n_no_support = (sub["initial_forest_support"] == 0).sum()
            n_no_followup = sub[ev_col].isna() & (sub["initial_forest_support"] == 1)
            evaluable = sub.loc[sub[ev_col].notna()]
            n_event = int((evaluable[ev_col] == 1).sum())
            n_censored = int((evaluable[ev_col] == 0).sum())
            print(f"  k={k}: evaluables={len(evaluable):,} (evento={n_event:,}, "
                  f"censurado_administrativo={n_censored:,}), sin_soporte_inicial={n_no_support:,}, "
                  f"sin_seguimiento_observable={int(n_no_followup.sum()):,}")
            diagnostics.append(f"  k={k}: evaluables={len(evaluable):,} evento={n_event:,} "
                                f"censurado={n_censored:,} sin_soporte_inicial={n_no_support:,} "
                                f"sin_seguimiento_observable={int(n_no_followup.sum()):,} "
                                f"L_k={year_max - k + 1}")

            out_cols = [
                "location_id", "entry_year", "decade_entry", "n_entries_location", "draw_id",
                "H_i", "n_px_valid_total", "frac_nonforest_last_5y", "n_transitions_before_entry",
                "sampled_traj_code", "initial_forest_support", ev_col, du_col,
            ]
            out = full[out_cols].rename(columns={ev_col: "event", du_col: "duration"})
            out_path = args.output_dir / f"dataset_k{k}_buffer{buffer_m}m.csv"
            out.to_csv(out_path, index=False, encoding="utf-8")
            print(f"  Guardado: {out_path} ({len(out):,} filas)")
        diagnostics.append("")

    (args.output_dir / "diagnostics.txt").write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
    print(f"\nGuardado: {args.output_dir / 'diagnostics.txt'}")


if __name__ == "__main__":
    main()
