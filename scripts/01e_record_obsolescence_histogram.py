# CCAI OBSOLESCENCIA - O1e: version de 01d con HISTOGRAMA completo de perdida por ubicacion,
# no solo el año maximo. Pensado para correr en el cluster, en un directorio plano -- TODAS las
# rutas de entrada y salida son flags de linea de comandos, sin ningun valor por defecto relativo
# al repo (a diferencia del resto de los scripts de ccai_obsolescencia/, que asumen la estructura
# local del proyecto). Consulta Earth Engine (a diferencia de casi todo lo demas en este repo).
"""
POR QUE
-------
01d_record_level_obsolescence.py usa ee.Reducer.max() sobre la imagen de primer año de perdida:
por ubicacion y buffer, se queda solo con el AÑO MAS RECIENTE de perdida detectado dentro del
buffer. Eso alcanza para la pregunta binaria "¿hubo perdida despues del registro, si/no?", pero
tira a la basura toda la informacion de CUANTOS pixeles del buffer se perdieron -- exactamente la
misma limitacion que 01d y 07_age_vs_habitat_change.py declaran explicitamente en sus propios
bloques POR QUE ("magnitud de cambio... 01d no calcula que FRACCION del area del buffer se
deforesto"). Ese hueco es real: O1c ya demostro, a nivel de CELDA, que un binario "cualquier
pixel" es extremadamente permisivo (61,5% de celdas 'afectadas' con un solo pixel de 30x30 m) y
que la curva de umbral de magnitud cambia la conclusion. La misma pregunta hace falta responderla
a nivel de REGISTRO/buffer, no solo de celda -- y para eso hace falta el histograma completo, no
el maximo.

ee.Reducer.frequencyHistogram() en vez de ee.Reducer.max() responde eso con UNA sola consulta
extra por chunk (no dos): el histograma completo {año: n_pixeles} de por si YA contiene el total
de pixeles validos (sumando todos los buckets, incluido el bucket año=0 = "nunca se detecto
perdida") -- no hace falta un reducer combinado aparte para contar, a diferencia de como
02a_mapbiomas_temporal_extract.py necesito combinar mean()+count() para la cobertura anual. Con
el histograma completo guardado, la fraccion de area perdida DESPUES de cualquier año de
referencia (el año del registro, distinto por registro) se puede derivar en pandas sumando los
buckets correspondientes -- sin volver a tocar GEE. Esa derivacion (frac_loss_after_record) es la
pieza que faltaba: convierte la pregunta binaria de 01d en una CONTINUA, analoga a
perdida_pct_celda de O1c pero a nivel de punto/buffer en vez de celda completa.

ORDEN DE BUFFERS 250,500,100,0 (no 0,100,250,500 como 01d): 01d proceso los buffers de mas chico
a mas grande, y se corto (stall aparente por cuota de Earth Engine restringida, ver la sesion de
trabajo del 13-14 ago) cuando ya llevaba los tres primeros completos y el ultimo (500 m, el mas
caro) a medio hacer. El buffer que O6/O1d terminaron usando como el "provisional" de producción
es 250 m -- si este script vuelve a cortarse por cuota, mejor que el buffer que ya se esta usando
rio abajo quede resuelto primero, no el ultimo.

DISCONTINUIDAD DELIBERADA CON EL RESTO DEL REPO: todos los demas scripts de ccai_obsolescencia/
asumen que corren dentro del repo local (rutas por defecto relativas a `Path(__file__).parents[N]`).
Este NO -- esta pensado para copiarse a un directorio plano en un cluster (sin la estructura
../data/, ccai_obsolescencia/data/, etc.), asi que CADA ruta de entrada y salida es un flag
obligatorio, sin default. Es la unica diferencia de diseño respecto al resto del repo, y es
intencional, no un descuido de estilo.

Supuesto mas fragil: frequencyHistogram() en Earth Engine devuelve las claves del histograma como
STRINGS de un numero (a veces con decimales, ej. "1994.0"), no como enteros -- se parsean con
`int(float(...))` como ya hacia 02a_mapbiomas_temporal_extract.py con su propio histograma de
perdida, por la misma razon. Si GEE cambiara ese formato de serializacion, el parseo fallaria
silenciosamente hacia valores incorrectos en vez de tirar un error -- no hay forma de validarlo
sin acceso a la documentacion interna de como frequencyHistogram serializa las claves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MAPBIOMAS_ASSET_DEFAULT = (
    "projects/mapbiomas-public/assets/brazil/lulc/collection9/"
    "mapbiomas_collection90_integration_v1"
)
FOREST_CLASSES_DEFAULT = [3, 4, 5, 49]  # Formacao Florestal, Savana Florestal, Mangue, Restinga Arborizada
GEE_PROJECT_DEFAULT = "YOUR-GEE-PROJECT-ID"
YEAR_START_DEFAULT = 1985
YEAR_END_DEFAULT = 2023
SCALE_M_DEFAULT = 30
CHUNK_SIZE_DEFAULT = 400
BUFFERS_M_DEFAULT = [250, 500, 100, 0]  # orden deliberado, ver bloque POR QUE
COORD_DECIMALS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="O1e - histograma completo de perdida por ubicacion/buffer (frequencyHistogram), "
        "pensado para correr en un directorio plano de cluster. Todas las rutas son obligatorias."
    )
    # --- Rutas: TODAS obligatorias, sin default relativo al repo (ver bloque POR QUE) ---
    parser.add_argument("--occurrences-csv", type=Path, required=True,
                         help="CSV con columnas species, decimalLatitude, decimalLongitude, year "
                              "(cell_id opcional, se preserva si esta).")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                         help="Carpeta para los checkpoints de histograma por chunk (resumible).")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Carpeta para las salidas derivadas (por buffer y combinada) y diagnostics.txt.")

    # --- Parametros de la consulta a Earth Engine ---
    parser.add_argument("--gee-project", default=GEE_PROJECT_DEFAULT)
    parser.add_argument("--mapbiomas-asset", default=MAPBIOMAS_ASSET_DEFAULT)
    parser.add_argument("--forest-classes", type=int, nargs="+", default=FOREST_CLASSES_DEFAULT)
    parser.add_argument("--year-start", type=int, default=YEAR_START_DEFAULT)
    parser.add_argument("--year-end", type=int, default=YEAR_END_DEFAULT)
    parser.add_argument("--scale-m", type=int, default=SCALE_M_DEFAULT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    parser.add_argument("--buffers-m", type=int, nargs="+", default=BUFFERS_M_DEFAULT,
                         help="Orden de procesamiento -- ver bloque POR QUE sobre por que 250 va primero.")
    parser.add_argument("--max-chunks", type=int, default=None,
                         help="Limitar chunks NUEVOS procesados por buffer en esta corrida (para pruebas). "
                              "Los ya completados en checkpoint no cuentan para este limite.")

    # --- Submuestreo opcional (pedido explicitamente; sin submuestrear por defecto) ---
    parser.add_argument("--sample-locations", type=int, default=None,
                         help="Si se pasa, procesar solo esta cantidad de ubicaciones unicas (muestra "
                              "aleatoria) en vez de todas -- para pruebas rapidas en el cluster antes de "
                              "lanzar la corrida completa. Por defecto: sin submuestrear (todas).")
    parser.add_argument("--sample-seed", type=int, default=42)

    return parser.parse_args()


def require_runtime():
    try:
        import ee
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        missing = getattr(exc, "name", "a required package")
        raise SystemExit(
            f"Falta una dependencia: {missing}\nInstalar con:\n  pip install earthengine-api pandas numpy"
        ) from exc
    return {"ee": ee, "np": np, "pd": pd}


# ---------------------------------------------------------------------------
# Paso 1: carga y deduplicacion por UBICACION (misma tecnica que 01d)
# ---------------------------------------------------------------------------


def load_occurrences(pd, path: Path):
    if not path.exists():
        raise SystemExit(f"No se encontro el csv de ocurrencias: {path}")
    df = pd.read_csv(path, dtype={"species": str})
    required = ["species", "decimalLatitude", "decimalLongitude", "year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"El csv de ocurrencias no tiene las columnas esperadas: {missing}")
    df = df.dropna(subset=required).copy()
    df["year"] = df["year"].astype(int)
    df["lat_r"] = df["decimalLatitude"].round(COORD_DECIMALS)
    df["lon_r"] = df["decimalLongitude"].round(COORD_DECIMALS)
    df["loc_key"] = df["lat_r"].map(lambda v: f"{v:.5f}") + "_" + df["lon_r"].map(lambda v: f"{v:.5f}")
    return df


def build_unique_locations(pd, np_mod, df, sample_locations: int | None, sample_seed: int):
    """Deduplicar por UBICACION sola, no por (ubicacion, year) -- el histograma de perdida es una
    propiedad fija del pixel/buffer, no depende de en que año se hizo la colecta (misma logica,
    y mismo ahorro de ~13% de consultas, que 01d_record_level_obsolescence.py documenta en su
    propio bloque POR QUE)."""
    unique_locs = df.drop_duplicates(subset=["loc_key"])[["loc_key", "lat_r", "lon_r"]].reset_index(drop=True)
    if sample_locations is not None and sample_locations < len(unique_locs):
        rng = np_mod.random.default_rng(sample_seed)
        idx = rng.choice(len(unique_locs), size=sample_locations, replace=False)
        unique_locs = unique_locs.iloc[sorted(idx)].reset_index(drop=True)
        print(f"--sample-locations activo: {len(unique_locs):,} ubicaciones de una muestra aleatoria "
              f"(semilla={sample_seed}), NO todas las disponibles.")
    return unique_locs


# ---------------------------------------------------------------------------
# Paso 2: extraccion via Earth Engine, ee.Reducer.frequencyHistogram()
# ---------------------------------------------------------------------------


def build_first_loss_year_image(ee, mapbiomas_asset: str, forest_classes: list[int], year_start: int, year_end: int):
    """Identica en logica a 01d/02a: recorre 1985-2023 y marca, por pixel, el PRIMER año de
    transicion bosque->no-bosque. 0 = sin transicion detectada en la ventana."""
    integration_img = ee.Image(mapbiomas_asset)
    years = range(year_start, year_end + 1)
    forest_by_year = {
        year: integration_img.select(f"classification_{year}")
        .remap(forest_classes, [1] * len(forest_classes), 0)
        .rename(f"forest_{year}")
        for year in years
    }
    years_sorted = sorted(years)
    first_loss = ee.Image(0).rename("first_loss_year")
    prev_forest = forest_by_year[years_sorted[0]]
    for year in years_sorted[1:]:
        curr_forest = forest_by_year[year]
        deforested_now = prev_forest.And(curr_forest.Not())
        first_loss = first_loss.where(first_loss.eq(0).And(deforested_now), ee.Image(year))
        prev_forest = curr_forest
    return first_loss.toInt16()


def chunk_frame(frame, chunk_size: int):
    for start in range(0, len(frame), chunk_size):
        yield frame.iloc[start : start + chunk_size]


def query_buffer_chunk_histogram(ee, first_loss_img, chunk_df, buffer_m: int, scale_m: int):
    """UNA consulta por chunk: frequencyHistogram() sobre la imagen de primer año de perdida. El
    histograma resultante {año_str: n_pixeles} ya contiene el total de pixeles validos (sumando
    todos los buckets, incluido año=0) -- no hace falta una segunda consulta de conteo, a
    diferencia de la cobertura anual de 02a (que combina mean()+count() porque promedia una
    mascara, no cuenta un histograma)."""
    features = []
    for row in chunk_df.itertuples():
        point = ee.Geometry.Point([row.lon_r, row.lat_r])
        geom = point if buffer_m == 0 else point.buffer(buffer_m)
        features.append(ee.Feature(geom, {"loc_key": row.loc_key}))
    fc = ee.FeatureCollection(features)
    reduced = first_loss_img.reduceRegions(collection=fc, reducer=ee.Reducer.frequencyHistogram(), scale=scale_m).getInfo()

    rows = []
    for feat in reduced["features"]:
        props = feat["properties"]
        loc_key = props["loc_key"]
        histogram = props.get("histogram", {}) or {}
        for year_str, px_count in histogram.items():
            # Ver "supuesto mas fragil" del bloque POR QUE: GEE serializa las claves del
            # histograma como strings numericos, a veces con decimales (ej. "1994.0").
            year = int(float(year_str))
            rows.append({"loc_key": loc_key, "loss_year": year, "n_px": px_count})
    return rows


def run_gee_extraction_for_buffer(ee, pd, unique_locs, buffer_m: int, first_loss_img, checkpoint_dir: Path, chunk_size: int, scale_m: int, max_chunks: int | None):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(chunk_frame(unique_locs, chunk_size))
    n_chunks_total = len(chunks)
    n_processed_this_run = 0
    n_done = 0
    failed_chunks = []

    for chunk_index, chunk_df in enumerate(chunks):
        ckpt = checkpoint_dir / f"hist_buffer{buffer_m}m_chunk_{chunk_index:04d}.csv"
        if ckpt.exists():
            n_done += len(chunk_df)
            continue
        if max_chunks is not None and n_processed_this_run >= max_chunks:
            print(f"  buffer {buffer_m}m: limite --max-chunks={max_chunks} alcanzado, resto queda para la proxima corrida.")
            break
        try:
            rows = query_buffer_chunk_histogram(ee, first_loss_img, chunk_df, buffer_m, scale_m)
        except Exception as exc:  # noqa: BLE001 - registrar y seguir con el resto de chunks
            failed_chunks.append(chunk_index)
            print(f"  buffer {buffer_m}m, chunk {chunk_index+1}/{n_chunks_total}: FALLO ({exc})")
            continue
        pd.DataFrame(rows).to_csv(ckpt, index=False, encoding="utf-8")
        n_processed_this_run += 1
        n_done += len(chunk_df)
        print(f"  buffer {buffer_m}m, chunk {chunk_index+1}/{n_chunks_total}: {len(chunk_df)} ubicaciones, "
              f"{len(rows)} filas de histograma")

    is_complete = n_done >= len(unique_locs) and not failed_chunks
    return n_done, n_chunks_total, is_complete, failed_chunks


def load_buffer_histogram(pd, checkpoint_dir: Path, buffer_m: int):
    files = sorted(checkpoint_dir.glob(f"hist_buffer{buffer_m}m_chunk_*.csv"))
    if not files:
        return None
    return pd.concat([pd.read_csv(f, dtype={"loc_key": str}) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------
# Paso 3: derivacion en PANDAS, sin volver a GEE
# ---------------------------------------------------------------------------


def derive_record_level(pd, occurrences_df, histogram_long: "pd.DataFrame"):
    """frac_loss_after_record y max_loss_year, calculados enteramente en pandas a partir del
    histograma ya bajado -- ninguna llamada nueva a Earth Engine.

    max_loss_year: el año maximo con n_px>0 entre los buckets > 0 (0 si nunca hubo perdida) --
    EXACTAMENTE lo que 01d_record_level_obsolescence.py obtenia con Reducer.max(), calculado aca
    a partir del histograma para compatibilidad con ese formato de salida.

    frac_loss_after_record: por (ubicacion, año del registro) -- no por ubicacion sola, porque el
    año de referencia cambia por registro -- la fraccion de pixeles validos del buffer cuyo
    PRIMER año de perdida es posterior al año del registro. Es la version continua de
    is_obsolete de 01d (que solo comparaba contra el MAXIMO, es decir, un binario 'hubo algun
    pixel perdido despues, si/no'). n_px_valid_total = suma de TODO el histograma de esa
    ubicacion, incluido el bucket año=0 (pixeles que nunca perdieron bosque en la ventana)."""
    n_px_valid = histogram_long.groupby("loc_key")["n_px"].sum().rename("n_px_valid_total")

    positive_years = histogram_long.loc[histogram_long["loss_year"] > 0]
    max_loss_year = positive_years.groupby("loc_key")["loss_year"].max().rename("max_loss_year")

    # Unico por (ubicacion, año de registro): evita recalcular la misma suma de histograma una
    # vez por cada registro que comparte exactamente esa combinacion (muchos registros de
    # distintas especies suelen compartir sitio y año de colecta).
    unique_loc_year = occurrences_df[["loc_key", "year"]].drop_duplicates().rename(columns={"year": "record_year"})
    joined = unique_loc_year.merge(positive_years, on="loc_key", how="left")
    after = joined.loc[joined["loss_year"] > joined["record_year"]]
    px_after = after.groupby(["loc_key", "record_year"])["n_px"].sum().rename("n_px_lost_after_record")

    per_loc_year = unique_loc_year.merge(px_after, on=["loc_key", "record_year"], how="left")
    per_loc_year["n_px_lost_after_record"] = per_loc_year["n_px_lost_after_record"].fillna(0.0)
    per_loc_year = per_loc_year.merge(n_px_valid, on="loc_key", how="left")
    per_loc_year["frac_loss_after_record"] = per_loc_year["n_px_lost_after_record"] / per_loc_year["n_px_valid_total"]
    per_loc_year = per_loc_year.merge(max_loss_year, on="loc_key", how="left")
    per_loc_year["max_loss_year"] = per_loc_year["max_loss_year"].fillna(0).astype(int)

    # Adjuntar a CADA registro original (species, cell_id si existe, etc.), no solo a los pares
    # (ubicacion, año) unicos -- el merge de vuelta es 1-a-muchos a proposito.
    join_cols = ["loc_key", "record_year", "n_px_valid_total", "n_px_lost_after_record", "frac_loss_after_record", "max_loss_year"]
    result = occurrences_df.rename(columns={"year": "record_year"}).merge(
        per_loc_year[join_cols], on=["loc_key", "record_year"], how="inner"
    )
    return result


# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    ee, np, pd = rt["ee"], rt["np"], rt["pd"]

    print("O1e - Histograma completo de perdida por ubicacion/buffer (frequencyHistogram)")
    print("====================================================================================")
    print(f"Ocurrencias: {args.occurrences_csv}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Salidas: {args.output_dir}")

    df = load_occurrences(pd, args.occurrences_csv)
    print(f"Registros validos: {len(df):,}")
    unique_locs = build_unique_locations(pd, np, df, args.sample_locations, args.sample_seed)
    print(f"Ubicaciones unicas a procesar: {len(unique_locs):,}")

    if args.sample_locations is not None:
        df = df.loc[df["loc_key"].isin(set(unique_locs["loc_key"]))].copy()
        print(f"Registros restringidos a la muestra de ubicaciones: {len(df):,}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ee.Initialize(project=args.gee_project)
    print(f"Earth Engine inicializado (project={args.gee_project})")
    first_loss_img = build_first_loss_year_image(
        ee, args.mapbiomas_asset, args.forest_classes, args.year_start, args.year_end
    )

    print(f"\nOrden de procesamiento de buffers: {args.buffers_m} (ver bloque POR QUE)")

    buffer_status = []
    derived_frames = []
    for buffer_m in args.buffers_m:
        print(f"\n--- Buffer {buffer_m} m ---")
        n_done, n_chunks_total, is_complete, failed_chunks = run_gee_extraction_for_buffer(
            ee, pd, unique_locs, buffer_m, first_loss_img, args.checkpoint_dir, args.chunk_size, args.scale_m, args.max_chunks
        )
        print(f"  Ubicaciones resueltas: {n_done:,}/{len(unique_locs):,} "
              f"({n_done/len(unique_locs):.1%}). Completo: {'SI' if is_complete else 'NO'}."
              + (f" Chunks fallidos: {failed_chunks}" if failed_chunks else ""))
        buffer_status.append({
            "buffer_m": buffer_m, "n_ubicaciones_resueltas": n_done, "n_ubicaciones_total": len(unique_locs),
            "completo": is_complete, "chunks_fallidos": len(failed_chunks),
        })

        if not is_complete:
            print(f"  Buffer {buffer_m}m incompleto -- no se deriva frac_loss_after_record todavia. "
                  "Correr el mismo comando de nuevo para retomar (los chunks ya hechos quedan en checkpoint).")
            continue

        histogram_long = load_buffer_histogram(pd, args.checkpoint_dir, buffer_m)
        derived = derive_record_level(pd, df, histogram_long)
        derived["buffer_m"] = buffer_m
        derived_path = args.output_dir / f"record_loss_derived_buffer{buffer_m}m.csv"
        derived.to_csv(derived_path, index=False, encoding="utf-8")
        derived_frames.append(derived)
        print(f"  Guardado: {derived_path} ({len(derived):,} registros, "
              f"frac_loss_after_record media={derived['frac_loss_after_record'].mean():.4f})")

    if derived_frames:
        combined = pd.concat(derived_frames, ignore_index=True)
        combined_path = args.output_dir / "record_loss_derived_all_buffers.csv"
        combined.to_csv(combined_path, index=False, encoding="utf-8")
        print(f"\nGuardado combinado: {combined_path} ({len(combined):,} filas, "
              f"{combined['buffer_m'].nunique()} buffers completos)")

    status_table = pd.DataFrame(buffer_status)
    diagnostics = [
        "O1e diagnostics - histograma completo de perdida por ubicacion/buffer",
        "===========================================================================",
        f"Ocurrencias: {args.occurrences_csv}",
        f"Checkpoints: {args.checkpoint_dir}",
        f"Salidas: {args.output_dir}",
        f"Asset MapBiomas: {args.mapbiomas_asset}, clases de bosque: {args.forest_classes}",
        f"Años: {args.year_start}-{args.year_end}, escala: {args.scale_m} m, chunk_size: {args.chunk_size}",
        f"Orden de buffers procesados: {args.buffers_m} (ver bloque POR QUE del script).",
        "",
        (f"--sample-locations={args.sample_locations} (semilla={args.sample_seed}) -- CORRIDA DE PRUEBA, "
         "NO todas las ubicaciones. No usar estos numeros como resultado final del paper."
         if args.sample_locations is not None else
         "Sin submuestrear: se proceso (o se intento procesar) la totalidad de ubicaciones unicas."),
        "",
        f"Ubicaciones unicas: {len(unique_locs):,}. Registros validos usados: {len(df):,}.",
        "",
        "Estado por buffer:",
        status_table.to_string(index=False),
        "",
        "Reductor usado: ee.Reducer.frequencyHistogram() (no ee.Reducer.max() como 01d) -- se "
        "guarda el histograma COMPLETO {año_perdida: n_pixeles} por ubicacion y buffer, del cual "
        "n_px_valid_total (suma de todos los buckets, incluido año=0) y max_loss_year (compatible "
        "con la salida de 01d) se derivan en pandas, sin volver a consultar Earth Engine -- ver "
        "bloque POR QUE y la funcion derive_record_level.",
        "",
        "frac_loss_after_record = fraccion de pixeles validos del buffer cuyo PRIMER año de "
        "perdida es posterior al año del registro -- calculado por (ubicacion, año del registro), "
        "no por ubicacion sola, porque el año de referencia cambia por registro. Es la version "
        "CONTINUA de is_obsolete de 01d (que era 'algun pixel perdido despues, si/no'): con esto "
        "se puede construir una curva de sensibilidad a nivel de registro/buffer, analoga a la "
        "curva de umbral de magnitud de 01c_loss_threshold_sensitivity.py pero a nivel de celda.",
    ]
    for row in buffer_status:
        if row["completo"]:
            frac = pd.concat(derived_frames)
            frac = frac.loc[frac["buffer_m"] == row["buffer_m"], "frac_loss_after_record"]
            diagnostics.append(
                f"  Buffer {row['buffer_m']}m: frac_loss_after_record media={frac.mean():.4f}, "
                f"mediana={frac.median():.4f}, frac_registros_con_perdida_alguna(>0)={ (frac>0).mean():.4f}."
            )
    (args.output_dir / "diagnostics.txt").write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
    print(f"\nGuardado: {args.output_dir / 'diagnostics.txt'}")


if __name__ == "__main__":
    main()
