# CCAI OBSOLESCENCIA - D1 (PLAN_V3_EJECUTABLE.md): trayectoria forestal COMPLETA 1985-Y_max
# codificada en bits de un unico entero Int64 por pixel, extraida por ubicacion y buffer via
# frequencyHistogram(). Pensado para correr en el cluster, directorio plano -- TODAS las rutas
# son flags de linea de comandos (misma discontinuidad deliberada que 01e_record_obsolescence_
# histogram.py respecto al resto del repo, y por la misma razon: sin estructura local del repo).
# Consulta Earth Engine.
"""
POR QUE
-------
PLAN_V3_EJECUTABLE.md, seccion D1, es la unica especificacion valida para este script (no
PLAN_PAPER_OBSOLESCENCIA.md ni PLAN_V2_SUPERVIVENCIA.md, que quedan como registro historico de
diseños superados). Cita textual del plan sobre por que hace falta la trayectoria COMPLETA y no
un evento unico: "Un raster de 'primer evento persistente desde 1985' no sirve: un pixel talado
en 1990, regenerado en 2000 y colectado en 2010 tendria evento en 1990 y quedaria mal
clasificado, aunque si era bosque en la colecta. El evento debe condicionarse al año de entrada."
01d/01e ya mostraron esa limitacion en la practica (01d con Reducer.max() del año de perdida, 01e
con el histograma de perdida): ninguno de los dos puede responder "¿era bosque en 2010?" para un
pixel que se taló antes y se regenero despues -- ambos asumen que una vez perdido, un pixel queda
perdido. Extraer la trayectoria completa (bosque/no-bosque en CADA año) resuelve eso de una vez,
sin multiplicar consultas a Earth Engine: en vez de N consultas (una por año de entrada posible),
UNA sola consulta por ubicacion/buffer que trae la trayectoria entera, de la que D2 deriva en
pandas (sin volver a GEE) el evento condicionado a CUALQUIER año de entrada, cualquier k
(persistencia de la conversion) y la regeneracion.

CODIFICACION EN BITS DE UN Int64, NO 40 BANDAS NI 40 CONSULTAS: la alternativa obvia --
descargar las 40 bandas classification_1985..Y_max y reducirlas en pandas -- multiplicaria el
volumen transferido de Earth Engine por 40 sin necesidad: reduceRegions ya puede devolver, por
CADA pixel unico de la union de ubicaciones+buffer, su combinacion completa de 40 años como una
sola clave de frequencyHistogram, si esos 40 años se empaquetan en un unico numero ANTES de
reducir. El bit j de ese numero codifica si el pixel era bosque en el año 1985+j (1=bosque,
0=no-bosque), calculado con bitwiseOr/leftShift sobre una imagen ya casteada a int64 -- el
casteo a int64 tiene que pasar ANTES del leftShift, no despues: si se corre en int32 (el default
implicito de muchas operaciones de EE) el shift hacia la posicion 32+ simplemente pierde esos
bits en vez de tirar un error, silenciosamente. Con Y_max=2024 (ver results/frozen/
cohort_manifest.txt, D0) la ventana 1985-2024 son 40 años -> 40 bits -> 2^40 ≈ 1,1×10^12,
comodamente por debajo de 2^53 (el limite de entero exactamente representable en IEEE-754 doble
precision) y muy por debajo de 2^63-1 (el rango positivo de un int64 con signo) -- no hay riesgo
de desbordar a valores negativos ni de perder precision si en algun punto de la cadena (GEE,
JSON, pandas) el numero pasara por un float de 64 bits.

NINGUNA FORMULA HARDCODEA EL ULTIMO AÑO: Y_max se descubre en tiempo de ejecucion leyendo las
bandas classification_YYYY del asset que se pase por --mapbiomas-asset (funcion
discover_year_max), no se asume 2024 en ningun lugar del codigo -- ver results/frozen/
cohort_manifest.txt (D0) para como se resolvio Collection 9 (Y_max=2023) vs Collection 10 v1/v2
(Y_max=2024) hoy: Collection 10 v2 es la mas reciente publicada (2025-07-12) y es el default de
--mapbiomas-asset, pero si mañana existe una Collection 11 accesible con Y_max=2025, correr este
script con --mapbiomas-asset apuntando ahi cambia Y_max automaticamente, sin tocar una sola
formula. discover_year_max tambien verifica que no haya AÑOS FALTANTES entre 1985 y el maximo
encontrado -- un hueco silencioso correria el bit j de todos los años posteriores al hueco.

MASCARA DE MAPBIOMAS -- VERIFICADO EN VIVO, NO ASUMIDO (ver results/frozen/cohort_manifest.txt):
Image.remap(from, to, default) preserva la mascara original del pixel -- un pixel fuera del
territorio de Brasil (oceano, paises vecinos) queda enmascarado ANTES del remap y sigue
enmascarado DESPUES, no se convierte silenciosamente en "no bosque" (default=0) solo porque su
valor de clase no esta en la lista `from`. Se verifico ademas que la mascara se propaga
correctamente a traves de los 40 bitwiseOr encadenados: un punto oceanico da un codigo
COMPLETAMENTE enmascarado (histograma vacio), no un codigo 0 con conteo de pixeles > 0. Esto
importa porque un buffer que cae parcialmente en el mar (ubicaciones costeras, frecuentes en
Mata Atlantica) tiene que excluir esos pixeles del conteo, no contarlos como "nunca fue bosque".

CLAVES DEL HISTOGRAMA COMO int(), NUNCA float(): identico supuesto fragil que 01e_record_
obsolescence_histogram.py ya declara para sus propias claves de año -- GEE serializa las claves
de frequencyHistogram como strings, a veces con un sufijo ".0" segun como se serialice
internamente el numero. Con codigos de hasta 2^40, convertir por `int(float(clave))` seria
matematicamente exacto (2^40 esta muy por debajo de los 2^53 que un float64 representa sin
perder precision) -- pero el plan pide explicitamente NUNCA pasar por float(), asi que
parse_trajectory_code() nunca invoca float(): separa a mano el sufijo ".0" si aparece (con un
assert de que la parte fraccionaria es literalmente "0"), y aplica int() directo sobre la parte
entera como string.

PILOTO OBLIGATORIO ANTES DE LA CORRIDA COMPLETA: --sample-locations activa, ademas del
submuestreo, una bateria completa de verificaciones (funcion run_pilot_verification) que el plan
exige correr ANTES de lanzar las 135.021 ubicaciones completas -- ver el bloque de verificacion
mas abajo para el detalle de cada chequeo. Cita del plan sobre por que: "Un error de orden de
bits descubierto tras doce horas de extraccion cuesta el paper." Este script NUNCA lanza la
corrida completa por si solo: --sample-locations es obligatorio para el piloto, y la corrida sin
submuestrear es una decision explicita de quien invoca el script (no un default), documentada en
diagnostics.txt de cada corrida.

Alternativa descartada -- imagen de "primer evento persistente" (como 01d): ver primer parrafo.
Alternativa descartada -- 40 consultas anuales en vez de una consulta empaquetada: multiplicaria
el trafico de Earth Engine por 40 sin necesidad, y con la cuota ya restringida (ver 01e, sesion
del 13-14 ago) eso es exactamente lo que hay que evitar.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MAPBIOMAS_ASSET_DEFAULT = (
    "projects/mapbiomas-public/assets/brazil/lulc/collection10/"
    "mapbiomas_brazil_collection10_integration_v2"
)  # ver results/frozen/cohort_manifest.txt (D0): version mas reciente publicada, Y_max=2024
FOREST_CLASSES_DEFAULT = [3, 4, 5, 49]  # Formacao Florestal, Savana Florestal, Manguezal, Restinga Arborizada
GEE_PROJECT_DEFAULT = "YOUR-GEE-PROJECT-ID"
YEAR_START_DEFAULT = 1985  # D0: registros anteriores a 1985 se EXCLUYEN, ver cohort_manifest.txt
YEAR_MAX_TOLERANCE_DEFAULT = 3  # cota superior year <= Y_max+tolerancia, misma convencion que 01d/06/07/09
SCALE_M_DEFAULT = 30
CHUNK_SIZE_DEFAULT = 400  # mismo chunking que 01e, ver bloque POR QUE
BUFFERS_M_DEFAULT = [250, 500, 100]  # 250 principal, 500 sensibilidad, 100 si sobra tiempo (D1)
COORD_DECIMALS = 5
PILOT_MANUAL_CHECK_N = 10  # "comparar a mano 10 pixeles contra las bandas originales" (D1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D1 - trayectoria forestal 1985-Y_max codificada en bits de un Int64 por "
        "ubicacion/buffer (frequencyHistogram). Todas las rutas son obligatorias (directorio plano)."
    )
    parser.add_argument("--occurrences-csv", type=Path, required=True,
                         help="CSV con columnas species, decimalLatitude, decimalLongitude, year "
                              "(cell_id opcional, se preserva si esta).")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                         help="Carpeta para los checkpoints de trayectoria por chunk (resumible).")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Carpeta para las salidas derivadas, diagnostics.txt y (en modo piloto) "
                              "el reporte de verificacion.")

    parser.add_argument("--gee-project", default=GEE_PROJECT_DEFAULT)
    parser.add_argument("--mapbiomas-asset", default=MAPBIOMAS_ASSET_DEFAULT)
    parser.add_argument("--forest-classes", type=int, nargs="+", default=FOREST_CLASSES_DEFAULT)
    parser.add_argument("--year-start", type=int, default=YEAR_START_DEFAULT)
    parser.add_argument("--year-max-tolerance", type=int, default=YEAR_MAX_TOLERANCE_DEFAULT,
                         help="Cota superior year <= Y_max+tolerancia (Y_max real, no hardcodeado).")
    parser.add_argument("--scale-m", type=int, default=SCALE_M_DEFAULT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    parser.add_argument("--buffers-m", type=int, nargs="+", default=BUFFERS_M_DEFAULT,
                         help="Orden de procesamiento -- 250 principal primero (D1).")
    parser.add_argument("--max-chunks", type=int, default=None,
                         help="Limitar chunks NUEVOS procesados por buffer en esta corrida (para pruebas). "
                              "Los ya completados en checkpoint no cuentan para este limite.")

    parser.add_argument("--sample-locations", type=int, default=None,
                         help="OBLIGATORIO para el piloto (D1 exige 100 antes de la corrida completa). "
                              "Si se pasa, se activa ademas run_pilot_verification() -- ver bloque POR QUE. "
                              "Por defecto: sin submuestrear (corrida completa, decision explicita).")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--manual-check-n", type=int, default=PILOT_MANUAL_CHECK_N,
                         help="Cuantos puntos comparar a mano contra las bandas crudas en modo piloto.")
    parser.add_argument(
        "--confirm-full-run", action="store_true",
        help="Requerido para correr SIN --sample-locations (las 135.021 ubicaciones completas). "
        "D1 exige el piloto aprobado primero -- ver bloque POR QUE. No es un flag para saltarse "
        "el piloto, es la confirmacion explicita de que ya se corrio y se aprobo.",
    )

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
# Y_max: SIEMPRE descubierto del asset real, nunca hardcodeado -- ver bloque POR QUE.
# ---------------------------------------------------------------------------


def discover_year_max(ee, mapbiomas_asset: str, year_start: int) -> int:
    img = ee.Image(mapbiomas_asset)
    band_names = img.bandNames().getInfo()
    years = sorted(
        int(m.group(1)) for b in band_names if (m := re.match(r"^classification_(\d{4})$", b))
    )
    if not years:
        raise SystemExit(f"No se encontraron bandas classification_YYYY en {mapbiomas_asset}")
    if years[0] != year_start:
        raise SystemExit(
            f"El primer año disponible en el asset ({years[0]}) no coincide con --year-start "
            f"({year_start}) -- revisar el asset o el flag."
        )
    expected = list(range(years[0], years[-1] + 1))
    if years != expected:
        missing = sorted(set(expected) - set(years))
        raise SystemExit(
            f"Bandas classification_YYYY con huecos en {mapbiomas_asset}: faltan los años {missing}. "
            "Un hueco correria el bit de todos los años posteriores -- no se puede continuar."
        )
    return years[-1]


# ---------------------------------------------------------------------------
# Paso 1: carga y deduplicacion por UBICACION (misma tecnica que 01d/01e)
# ---------------------------------------------------------------------------


def load_occurrences(pd, path: Path, year_start: int):
    """Filtra year >= year_start (D0: 'registros anteriores a 1985 se EXCLUYEN', ver
    cohort_manifest.txt) -- SOLO la cota inferior, que no depende de Y_max. La cota superior
    (year <= Y_max + tolerancia) se aplica aparte, en filter_year_upper_bound(), DESPUES de
    descubrir Y_max en vivo -- load_occurrences corre antes de inicializar Earth Engine, no
    puede depender todavia de un Y_max que no se conoce."""
    if not path.exists():
        raise SystemExit(f"No se encontro el csv de ocurrencias: {path}")
    df = pd.read_csv(path, dtype={"species": str})
    required = ["species", "decimalLatitude", "decimalLongitude", "year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"El csv de ocurrencias no tiene las columnas esperadas: {missing}")
    df = df.dropna(subset=required).copy()
    df["year"] = df["year"].astype(int)
    df = df.loc[df["year"] >= year_start].copy()
    df["lat_r"] = df["decimalLatitude"].round(COORD_DECIMALS)
    df["lon_r"] = df["decimalLongitude"].round(COORD_DECIMALS)
    df["loc_key"] = df["lat_r"].map(lambda v: f"{v:.5f}") + "_" + df["lon_r"].map(lambda v: f"{v:.5f}")
    return df


def filter_year_upper_bound(df, year_max: int, tolerance: int):
    """Cota superior year <= Y_max + tolerancia -- misma convencion que 01d/06/07/09
    (YEAR_MAX_TOLERANCE), pero atada al Y_max REAL descubierto en esta corrida (2024, no el 2023
    de Collection 9 que esos scripts todavia usan) en vez de un YEAR_END hardcodeado. Un registro
    fechado despues de Y_max+tolerancia no tiene forma de verificarse contra la trayectoria
    disponible."""
    return df.loc[df["year"] <= year_max + tolerance].copy()


def build_unique_locations(pd, df):
    """Deduplicar por UBICACION sola -- la trayectoria forestal es una propiedad fija del
    pixel/buffer, no depende del año de ninguna colecta particular (misma logica que 01e).
    Se conserva year_min (año de registro mas antiguo en esa ubicacion) solo para poder
    estratificar el piloto por 'años distintos' como pide D1 -- no participa en la extraccion."""
    year_min = df.groupby("loc_key")["year"].min().rename("year_min")
    unique_locs = df.drop_duplicates(subset=["loc_key"])[["loc_key", "lat_r", "lon_r"]].reset_index(drop=True)
    unique_locs = unique_locs.merge(year_min.reset_index(), on="loc_key", how="left")
    return unique_locs


def stratified_pilot_sample(pd, np_mod, unique_locs, n_sample: int, seed: int):
    """D1 exige '100 localidades de años y regiones distintas', no una muestra aleatoria simple
    que podria (por azar) quedar concentrada en una sola zona o decada. Estratifica por
    (bin geografico de 2 grados) x (decada de year_min) y recorre las estratas en round-robin
    (orden aleatorio de estratas, orden aleatorio DENTRO de cada estrata) hasta juntar n_sample
    -- maximiza la cantidad de estratas DISTINTAS representadas en vez de solo tomar las mas
    pobladas, que es lo que un muestreo proporcional haria."""
    df = unique_locs.copy()
    df["geo_bin"] = (
        np_mod.floor(df["lat_r"] / 2.0).astype(int).astype(str) + "_" +
        np_mod.floor(df["lon_r"] / 2.0).astype(int).astype(str)
    )
    df["decade"] = (df["year_min"] // 10 * 10).astype(int)
    df["stratum"] = df["geo_bin"] + "_" + df["decade"].astype(str)

    rng = np_mod.random.default_rng(seed)
    strata = df["stratum"].unique().tolist()
    rng.shuffle(strata)
    pools = {s: df.index[df["stratum"] == s].to_numpy() for s in strata}
    for s in pools:
        rng.shuffle(pools[s])

    picked = []
    i = 0
    while len(picked) < n_sample and any(len(pools[s]) > 0 for s in strata):
        s = strata[i % len(strata)]
        if len(pools[s]) > 0:
            picked.append(pools[s][-1])
            pools[s] = pools[s][:-1]
        i += 1
        if i > n_sample * 50:  # salvaguarda: no debe hacer falta tantas vueltas para n_sample chico
            break

    sample = df.loc[picked[:n_sample]].reset_index(drop=True)
    n_strata_used = sample["stratum"].nunique()
    print(f"--sample-locations activo: {len(sample):,} ubicaciones, estratificadas por region "
          f"(bin de 2°) x decada de year_min -- {n_strata_used} estratas distintas representadas "
          f"de {len(strata)} disponibles. Años cubiertos: {sample['year_min'].min()}-"
          f"{sample['year_min'].max()}. Bines geograficos distintos: {sample['geo_bin'].nunique()}.")
    return sample.drop(columns=["geo_bin", "decade", "stratum"])


# ---------------------------------------------------------------------------
# Paso 2: imagen de trayectoria empaquetada en bits + extraccion via frequencyHistogram
# ---------------------------------------------------------------------------


def build_trajectory_code_image(ee, mapbiomas_asset: str, forest_classes: list[int], year_start: int, year_max: int):
    """El bit j codifica el año year_start+j: 1=bosque, 0=no-bosque. remap() preserva la mascara
    original (verificado en vivo, ver bloque POR QUE) -- un pixel enmascarado en CUALQUIER año
    de la cadena deja el codigo combinado enmascarado para ese pixel, no lo convierte en 0."""
    integration_img = ee.Image(mapbiomas_asset)
    years = list(range(year_start, year_max + 1))
    code = ee.Image(0).toInt64()
    for j, year in enumerate(years):
        forest_bit = (
            integration_img.select(f"classification_{year}")
            .remap(forest_classes, [1] * len(forest_classes), 0)
            .toInt64()
        )
        code = code.bitwiseOr(forest_bit.leftShift(j))
    return code.rename("traj_code")


def chunk_frame(frame, chunk_size: int):
    for start in range(0, len(frame), chunk_size):
        yield frame.iloc[start : start + chunk_size]


def parse_trajectory_code(key: str) -> int:
    """NUNCA float() -- ver bloque POR QUE. Si GEE serializa con un sufijo '.0', se separa a
    mano (con assert de que la parte fraccionaria es literalmente cero) y se aplica int() solo
    sobre la parte entera, como string."""
    if "." in key:
        int_part, frac_part = key.split(".", 1)
        if set(frac_part) - {"0"}:
            raise ValueError(f"Clave de codigo de trayectoria con parte fraccionaria no nula: {key!r}")
        key = int_part
    code = int(key)
    if code < 0:
        raise ValueError(f"Codigo de trayectoria negativo -- no deberia ser posible: {code}")
    return code


def query_buffer_chunk_trajectory(ee, code_img, chunk_df, buffer_m: int, scale_m: int):
    features = []
    for row in chunk_df.itertuples():
        point = ee.Geometry.Point([row.lon_r, row.lat_r])
        geom = point if buffer_m == 0 else point.buffer(buffer_m)
        features.append(ee.Feature(geom, {"loc_key": row.loc_key}))
    fc = ee.FeatureCollection(features)
    reduced = code_img.reduceRegions(collection=fc, reducer=ee.Reducer.frequencyHistogram(), scale=scale_m).getInfo()

    rows = []
    for feat in reduced["features"]:
        props = feat["properties"]
        loc_key = props["loc_key"]
        histogram = props.get("histogram", {}) or {}
        for code_str, px_count in histogram.items():
            rows.append({"loc_key": loc_key, "traj_code": parse_trajectory_code(code_str), "n_px": px_count})
    return rows


def run_gee_extraction_for_buffer(ee, pd, unique_locs, buffer_m: int, code_img, checkpoint_dir: Path, chunk_size: int, scale_m: int, max_chunks: int | None):
    """Instrumentado con tiempo por chunk (time.perf_counter) -- pedido explicito para poder
    medir el costo real de un chunk de tamaño de PRODUCCION (chunk_size) y extrapolar la
    duracion total antes de lanzar la corrida completa, ver pilot_approval.txt / criterio (b)."""
    import time as time_mod
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(chunk_frame(unique_locs, chunk_size))
    n_chunks_total = len(chunks)
    n_processed_this_run = 0
    n_done = 0
    failed_chunks = []
    chunk_seconds = []

    for chunk_index, chunk_df in enumerate(chunks):
        ckpt = checkpoint_dir / f"traj_buffer{buffer_m}m_chunk_{chunk_index:04d}.csv"
        if ckpt.exists():
            n_done += len(chunk_df)
            continue
        if max_chunks is not None and n_processed_this_run >= max_chunks:
            print(f"  buffer {buffer_m}m: limite --max-chunks={max_chunks} alcanzado, resto queda para la proxima corrida.")
            break
        t0 = time_mod.perf_counter()
        try:
            rows = query_buffer_chunk_trajectory(ee, code_img, chunk_df, buffer_m, scale_m)
        except Exception as exc:  # noqa: BLE001 - registrar y seguir con el resto de chunks
            failed_chunks.append(chunk_index)
            print(f"  buffer {buffer_m}m, chunk {chunk_index+1}/{n_chunks_total}: FALLO ({exc})")
            continue
        elapsed = time_mod.perf_counter() - t0
        chunk_seconds.append(elapsed)
        pd.DataFrame(rows).to_csv(ckpt, index=False, encoding="utf-8")
        n_processed_this_run += 1
        n_done += len(chunk_df)
        print(f"  buffer {buffer_m}m, chunk {chunk_index+1}/{n_chunks_total}: {len(chunk_df)} ubicaciones, "
              f"{len(rows)} filas de codigo unico, {elapsed:.1f}s")

    is_complete = n_done >= len(unique_locs) and not failed_chunks
    return n_done, n_chunks_total, is_complete, failed_chunks, chunk_seconds


def load_buffer_trajectory(pd, checkpoint_dir: Path, buffer_m: int):
    files = sorted(checkpoint_dir.glob(f"traj_buffer{buffer_m}m_chunk_*.csv"))
    if not files:
        return None
    return pd.concat(
        [pd.read_csv(f, dtype={"loc_key": str, "traj_code": "int64"}) for f in files], ignore_index=True
    )


# ---------------------------------------------------------------------------
# encode/decode -- funciones puras, usadas tanto por D2 (derivacion en pandas) como por la
# verificacion del piloto (encode(decode(code)) == code).
# ---------------------------------------------------------------------------


def decode(code: int, n_years: int) -> list[int]:
    return [(code >> j) & 1 for j in range(n_years)]


def encode(bits: "list[int]") -> int:
    code = 0
    for j, b in enumerate(bits):
        if b:
            code |= 1 << j
    return code


# ---------------------------------------------------------------------------
# PILOTO OBLIGATORIO -- run_pilot_verification(). Ver bloque POR QUE. Cada chequeo imprime su
# resultado y queda en pilot_verification.txt; CUALQUIER falla se reporta con *** FALLO *** y no
# se recomienda avanzar a la corrida completa.
# ---------------------------------------------------------------------------


def verify_roundtrip(pd, np_mod, combined: "pd.DataFrame", n_years: int) -> dict:
    distinct_codes = combined["traj_code"].unique()
    bad = []
    for code in distinct_codes:
        code = int(code)
        if encode(decode(code, n_years)) != code:
            bad.append(code)
    return {
        "n_codigos_distintos_verificados": len(distinct_codes),
        "n_fallas_roundtrip": len(bad),
        "codigos_con_falla": bad[:20],
        "ok": len(bad) == 0,
    }


def verify_no_negative_and_range(combined: "pd.DataFrame", n_years: int) -> dict:
    codes = combined["traj_code"].astype("int64")
    max_valid = (1 << n_years) - 1
    n_negative = int((codes < 0).sum())
    n_over_range = int((codes > max_valid).sum())
    return {
        "min_codigo": int(codes.min()) if len(codes) else None,
        "max_codigo": int(codes.max()) if len(codes) else None,
        "max_codigo_valido_2elevado_n_years_menos_1": max_valid,
        "n_codigos_negativos": n_negative,
        "n_codigos_fuera_de_rango": n_over_range,
        "ok": n_negative == 0 and n_over_range == 0,
    }


def verify_csv_precision_roundtrip(pd, checkpoint_dir: Path, buffer_m: int) -> dict:
    """Vuelve a leer los checkpoints YA ESCRITOS con dtype int64 explicito (nunca float) y
    confirma que ningun valor parece haber pasado por notacion cientifica o perdido precision
    -- si pandas hubiera inferido float64 en algun momento de la cadena, valores grandes
    (cercanos a 2^40) perderian los ultimos digitos silenciosamente al volver a int."""
    files = sorted(checkpoint_dir.glob(f"traj_buffer{buffer_m}m_chunk_*.csv"))
    if not files:
        return {"ok": False, "motivo": "sin checkpoints"}
    n_checked = 0
    n_mismatch = 0
    for f in files:
        raw_text = f.read_text(encoding="utf-8")
        has_scientific = bool(re.search(r"\d[eE][+-]?\d", raw_text))
        df_int = pd.read_csv(f, dtype={"loc_key": str, "traj_code": "int64"})
        df_str = pd.read_csv(f, dtype={"loc_key": str, "traj_code": str})
        matches = (df_int["traj_code"].astype(str) == df_str["traj_code"]).all()
        n_checked += len(df_int)
        if has_scientific or not matches:
            n_mismatch += 1
    return {
        "n_filas_verificadas": n_checked,
        "n_archivos_con_problema": n_mismatch,
        "ok": n_mismatch == 0,
    }


def verify_frequency_sum_vs_valid_pixels(ee, pd, code_img, pilot_locs, buffer_m: int, scale_m: int, combined: "pd.DataFrame") -> dict:
    """Cross-check INDEPENDIENTE: suma de n_px del histograma vs. una consulta APARTE sobre la
    MISMA imagen de codigo.

    ADVERTENCIA -- 'apartado del mismo tipo', no cualquier reducer: la primera version de este
    chequeo comparaba contra ee.Reducer.count(), que resulto NO ser comparable -- count() cuenta
    pixeles TOCADOS (entero: 214), mientras frequencyHistogram() pondera cada pixel por su
    fraccion de solapamiento con la geometria (fraccionario: 219.45...) -- son dos convenciones
    de conteo DISTINTAS de GEE para el mismo borde de buffer, no un error de una de las dos.
    Verificado empiricamente: ee.Reducer.sum() sobre una imagen constante-1 CON LA MISMA MASCARA
    que el codigo (mismo area-weighting que frequencyHistogram) reproduce la suma del histograma
    hasta el redondeo de punto flotante (219.45098039215677 vs 219.45098039215685) -- ese es el
    cross-check correcto, no count()."""
    features = []
    for row in pilot_locs.itertuples():
        point = ee.Geometry.Point([row.lon_r, row.lat_r])
        geom = point if buffer_m == 0 else point.buffer(buffer_m)
        features.append(ee.Feature(geom, {"loc_key": row.loc_key}))
    fc = ee.FeatureCollection(features)
    weighted = ee.Image(1).updateMask(code_img.mask()).rename("w")
    reduced = weighted.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=scale_m).getInfo()
    sum_indep_by_loc = {f["properties"]["loc_key"]: f["properties"].get("sum", 0.0) for f in reduced["features"]}

    sum_hist_by_loc = combined.groupby("loc_key")["n_px"].sum()
    diffs = []
    for loc_key, indep_val in sum_indep_by_loc.items():
        hist_sum = float(sum_hist_by_loc.get(loc_key, 0.0))
        diffs.append(abs(hist_sum - float(indep_val)))
    max_diff = max(diffs) if diffs else float("nan")
    n_zero_valid = sum(1 for v in sum_indep_by_loc.values() if v == 0)
    return {
        "n_ubicaciones_comparadas": len(sum_indep_by_loc),
        "max_diferencia_suma_histograma_vs_sum_independiente": max_diff,
        "n_ubicaciones_sin_pixeles_validos": n_zero_valid,
        "ok": (max_diff < 1e-4) if diffs else False,
    }


def manual_pixel_comparison(ee, mapbiomas_asset: str, forest_classes: list[int], year_start: int, year_max: int, code_img, points_df, n_points: int) -> "list[dict]":
    """'Comparar a mano 10 pixeles contra las bandas originales' (D1): para cada punto, se
    muestrean TODAS las bandas classification_YYYY crudas en ese pixel exacto (buffer 0 m) y,
    por separado, el codigo empaquetado en ese mismo pixel -- se decodifica el codigo y se
    compara bit a bit (los 40 años, no solo 1985 y Y_max) contra si la clase cruda de esa banda
    esta en forest_classes. Devuelve una fila por punto con el resultado, para inspeccion
    literal 'a mano' ademas del assert automatico."""
    raw_img = ee.Image(mapbiomas_asset)
    years = list(range(year_start, year_max + 1))
    n_years = len(years)
    sample = points_df.head(n_points)

    results = []
    for row in sample.itertuples():
        pt = ee.Geometry.Point([row.lon_r, row.lat_r])
        raw_values = raw_img.reduceRegion(ee.Reducer.first(), pt, scale=30).getInfo()
        code_value = code_img.reduceRegion(ee.Reducer.first(), pt, scale=30).getInfo().get("traj_code")
        if code_value is None:
            results.append({
                "loc_key": row.loc_key, "lat": row.lat_r, "lon": row.lon_r,
                "codigo": None, "ok": False, "motivo": "pixel enmascarado en el punto exacto (sin dato)",
            })
            continue
        code_value = int(code_value)
        bits = decode(code_value, n_years)
        mismatches = []
        for j, year in enumerate(years):
            raw_class = raw_values.get(f"classification_{year}")
            expected_bit = 1 if raw_class in forest_classes else 0
            if bits[j] != expected_bit:
                mismatches.append((year, raw_class, expected_bit, bits[j]))
        bit_1985 = bits[0]
        bit_ymax = bits[-1]
        raw_1985 = raw_values.get(f"classification_{year_start}")
        raw_ymax = raw_values.get(f"classification_{year_max}")
        results.append({
            "loc_key": row.loc_key, "lat": row.lat_r, "lon": row.lon_r,
            "codigo": code_value, "n_mismatches_40_años": len(mismatches),
            "primer_mismatch": mismatches[0] if mismatches else None,
            "bit_1985": bit_1985, "clase_cruda_1985": raw_1985,
            "bit_Ymax": bit_ymax, "clase_cruda_Ymax": raw_ymax,
            "ok": len(mismatches) == 0,
        })
    return results


def run_pilot_verification(ee, pd, np_mod, args, code_img, year_start: int, year_max: int, unique_locs_sample, checkpoint_dir: Path, output_dir: Path):
    n_years = year_max - year_start + 1
    print("\n" + "=" * 60)
    print("VERIFICACION OBLIGATORIA DEL PILOTO (D1) -- antes de correr las 135.021 ubicaciones")
    print("=" * 60)

    report_lines = [
        "D1 - Verificacion del piloto (PLAN_V3_EJECUTABLE.md)",
        "=====================================================",
        f"n_years (1985-{year_max}): {n_years}. 2^n_years = {1 << n_years:,} (< 2^53: {(1 << n_years) < (1 << 53)}).",
        "",
    ]
    all_ok = True

    buffer_frames = {}
    for buffer_m in args.buffers_m:
        combined = load_buffer_trajectory(pd, checkpoint_dir, buffer_m)
        if combined is None:
            print(f"  buffer {buffer_m}m: sin checkpoints, se omite de la verificacion.")
            continue
        buffer_frames[buffer_m] = combined

        rt = verify_roundtrip(pd, np_mod, combined, n_years)
        rng_check = verify_no_negative_and_range(combined, n_years)
        csv_check = verify_csv_precision_roundtrip(pd, checkpoint_dir, buffer_m)
        freq_check = verify_frequency_sum_vs_valid_pixels(
            ee, pd, code_img, unique_locs_sample, buffer_m, args.scale_m, combined
        )

        for name, res in [
            ("encode(decode(code))==code", rt),
            ("sin codigos negativos / fuera de rango", rng_check),
            ("precision CSV (sin notacion cientifica, sin perdida)", csv_check),
            ("suma histograma == sum() independiente (mismo area-weighting)", freq_check),
        ]:
            status = "OK" if res["ok"] else "*** FALLO ***"
            print(f"  buffer {buffer_m}m -- {name}: {status}")
            report_lines.append(f"buffer {buffer_m}m -- {name}: {status}")
            report_lines.append(f"  {res}")
            all_ok = all_ok and res["ok"]

    if not buffer_frames:
        raise SystemExit("No hay checkpoints para verificar -- el piloto no escribio ningun chunk.")

    first_buffer = args.buffers_m[0]
    manual_results = manual_pixel_comparison(
        ee, args.mapbiomas_asset, args.forest_classes, year_start, year_max,
        code_img, unique_locs_sample, args.manual_check_n,
    )
    n_manual_ok = sum(1 for r in manual_results if r["ok"])
    print(f"\n  Comparacion manual de {len(manual_results)} pixeles contra bandas crudas "
          f"(buffer 0 m, punto exacto): {n_manual_ok}/{len(manual_results)} OK")
    report_lines.append("")
    report_lines.append(f"COMPARACION MANUAL ({len(manual_results)} puntos, bandas crudas vs. codigo decodificado, "
                         "los 40 años, no solo 1985 y Y_max):")
    for r in manual_results:
        report_lines.append(f"  {r}")
        if not r["ok"]:
            all_ok = False

    n_1985_ok = sum(1 for r in manual_results if r.get("ok") and r.get("bit_1985") == (1 if r.get("clase_cruda_1985") in args.forest_classes else 0))
    n_ymax_ok = sum(1 for r in manual_results if r.get("ok") and r.get("bit_Ymax") == (1 if r.get("clase_cruda_Ymax") in args.forest_classes else 0))
    print(f"  Posicion del bit de 1985 correcta: {n_1985_ok}/{len(manual_results)}")
    print(f"  Posicion del bit de {year_max} (Y_max) correcta: {n_ymax_ok}/{len(manual_results)}")
    report_lines.append(f"Bit de 1985 en posicion correcta: {n_1985_ok}/{len(manual_results)}")
    report_lines.append(f"Bit de {year_max} (Y_max) en posicion correcta: {n_ymax_ok}/{len(manual_results)}")

    print(f"\n{'*** TODAS LAS VERIFICACIONES PASARON. ***' if all_ok else '*** HAY FALLAS -- NO CORRER LA CORRIDA COMPLETA. ***'}")
    report_lines.insert(1, f"RESULTADO GLOBAL: {'TODAS LAS VERIFICACIONES PASARON' if all_ok else 'HAY FALLAS -- NO CORRER LA CORRIDA COMPLETA'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pilot_verification.txt").write_text("\n".join(str(l) for l in report_lines) + "\n", encoding="utf-8")
    print(f"\nGuardado: {output_dir / 'pilot_verification.txt'}")
    return all_ok


# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    ee, np, pd = rt["ee"], rt["np"], rt["pd"]

    print("D1 - Trayectoria forestal 1985-Y_max codificada en bits (PLAN_V3_EJECUTABLE.md)")
    print("====================================================================================")
    print(f"Ocurrencias: {args.occurrences_csv}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Salidas: {args.output_dir}")

    df = load_occurrences(pd, args.occurrences_csv, args.year_start)
    print(f"Registros con year >= {args.year_start}: {len(df):,}")

    ee.Initialize(project=args.gee_project)
    print(f"Earth Engine inicializado (project={args.gee_project})")

    year_max = discover_year_max(ee, args.mapbiomas_asset, args.year_start)
    print(f"Y_max descubierto del asset (NO hardcodeado): {year_max} "
          f"(ventana {args.year_start}-{year_max}, {year_max - args.year_start + 1} años/bits)")

    df = filter_year_upper_bound(df, year_max, args.year_max_tolerance)
    print(f"Registros con year <= {year_max}+{args.year_max_tolerance}={year_max + args.year_max_tolerance}: {len(df):,}")
    unique_locs = build_unique_locations(pd, df)
    print(f"Ubicaciones unicas totales (year en [{args.year_start}, {year_max + args.year_max_tolerance}]): {len(unique_locs):,}")

    if args.sample_locations is None and not args.confirm_full_run:
        raise SystemExit(
            "Este script requiere --sample-locations (piloto) o --confirm-full-run explicito. "
            "D1 prohibe lanzar las 135.021 ubicaciones completas sin antes correr y aprobar el "
            "piloto -- ver bloque POR QUE y results/frozen/... Correr primero con "
            "--sample-locations 100; recien despues de revisar pilot_verification.txt, correr "
            "sin --sample-locations pero CON --confirm-full-run."
        )
    if args.sample_locations is None:
        print("*** CORRIDA COMPLETA CONFIRMADA (--confirm-full-run): las 135.021 ubicaciones, "
              "sin submuestrear. Se asume que el piloto ya se corrio y se aprobo. ***")
        unique_locs_sample = unique_locs
    else:
        unique_locs_sample = stratified_pilot_sample(pd, np, unique_locs, args.sample_locations, args.sample_seed)

    code_img = build_trajectory_code_image(ee, args.mapbiomas_asset, args.forest_classes, args.year_start, year_max)

    print(f"\nOrden de procesamiento de buffers: {args.buffers_m}")

    buffer_status = []
    all_chunk_seconds = {}
    for buffer_m in args.buffers_m:
        print(f"\n--- Buffer {buffer_m} m ---")
        n_done, n_chunks_total, is_complete, failed_chunks, chunk_seconds = run_gee_extraction_for_buffer(
            ee, pd, unique_locs_sample, buffer_m, code_img, args.checkpoint_dir, args.chunk_size, args.scale_m, args.max_chunks
        )
        all_chunk_seconds[buffer_m] = chunk_seconds
        print(f"  Ubicaciones resueltas: {n_done:,}/{len(unique_locs_sample):,} "
              f"({n_done/len(unique_locs_sample):.1%}). Completo: {'SI' if is_complete else 'NO'}."
              + (f" Chunks fallidos: {failed_chunks}" if failed_chunks else ""))
        if chunk_seconds:
            print(f"  Tiempo por chunk (nuevos en esta corrida): {[f'{s:.1f}s' for s in chunk_seconds]}")
        buffer_status.append({
            "buffer_m": buffer_m, "n_ubicaciones_resueltas": n_done, "n_ubicaciones_total": len(unique_locs_sample),
            "completo": is_complete, "chunks_fallidos": len(failed_chunks),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_table = pd.DataFrame(buffer_status)
    diagnostics = [
        "D1 diagnostics - trayectoria forestal codificada en bits (PLAN_V3_EJECUTABLE.md)",
        "===================================================================================",
        f"Ocurrencias: {args.occurrences_csv}",
        f"Checkpoints: {args.checkpoint_dir}",
        f"Salidas: {args.output_dir}",
        f"Asset MapBiomas: {args.mapbiomas_asset}",
        f"Clases de bosque: {args.forest_classes}",
        f"Ventana: {args.year_start}-{year_max} ({year_max - args.year_start + 1} años/bits), "
        f"Y_max DESCUBIERTO del asset, no hardcodeado.",
        f"scale: {args.scale_m} m. crs: nativo del asset (EPSG:4326, sin forzar reproyeccion, "
        f"misma convencion que 01d/01e/02a). tileScale: 1 (default, no se forzo otro valor). "
        f"chunk_size de esta corrida: {args.chunk_size}.",
        f"Orden de buffers procesados: {args.buffers_m}.",
        "",
        (
            "*** CORRIDA COMPLETA (--confirm-full-run): las 135.021 ubicaciones, SIN submuestrear. "
            "El piloto que aprobo esta corrida esta en results/trajectory_pilot/pilot_verification.txt "
            "(y, si se corrio, results/trajectory_resume_test/pilot_verification.txt) -- ver tambien "
            "results/frozen/pilot_approval.txt. ***"
            if args.sample_locations is None else
            f"*** CORRIDA DE PILOTO: --sample-locations={args.sample_locations} (semilla={args.sample_seed}), "
            "estratificada por region x decada -- ver consola para el detalle de estratas cubiertas. "
            "NO son las 135.021 ubicaciones completas. ***"
        ),
        "",
        "Estado por buffer:",
        status_table.to_string(index=False),
    ]
    (args.output_dir / "diagnostics.txt").write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
    print(f"\nGuardado: {args.output_dir / 'diagnostics.txt'}")

    # Metadata MAQUINA-LEIBLE (JSON) -- D2 (21_build_survival_dataset.py) necesita year_start/
    # year_max sin volver a Earth Engine y SIN hardcodearlos (misma regla de D1: "no hardcodear
    # 2024 en ningun sitio"). diagnostics.txt es texto libre, no apto para parsear de forma
    # confiable -- este sidecar es la fuente de verdad machine-readable para scripts posteriores.
    import json
    metadata = {
        "year_start": args.year_start, "year_max": year_max,
        "mapbiomas_asset": args.mapbiomas_asset, "forest_classes": args.forest_classes,
        "scale_m": args.scale_m, "buffers_m": args.buffers_m,
        "es_corrida_completa": args.sample_locations is None,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Guardado: {args.output_dir / 'run_metadata.json'}")

    if args.sample_locations is None:
        # Corrida completa (--confirm-full-run): la bateria de verificacion es especifica del
        # piloto (re-consulta count() independiente por ubicacion, comparacion manual de pixeles)
        # -- correrla sobre 135.021 ubicaciones multiplicaria el trafico a Earth Engine sin
        # necesidad, el piloto ya la aprobo. Se omite a proposito, no por descuido.
        print("\nCorrida completa: se omite run_pilot_verification() (ya se corrio y aprobo en "
              "el piloto) -- ver pilot_verification.txt de esa corrida anterior.")
        return

    all_ok = run_pilot_verification(
        ee, pd, np, args, code_img, args.year_start, year_max, unique_locs_sample, args.checkpoint_dir, args.output_dir
    )
    if not all_ok:
        raise SystemExit(
            "\nEl piloto NO paso todas las verificaciones -- revisar pilot_verification.txt antes "
            "de considerar correr la corrida completa. NO se lanza automaticamente."
        )
    print("\nPiloto verificado OK. La corrida completa (135.021 ubicaciones, sin --sample-locations) "
          "NO se lanza desde aca -- requiere --confirm-full-run explicito, y confirmacion del usuario, "
          "antes de invocarse.")


if __name__ == "__main__":
    main()
