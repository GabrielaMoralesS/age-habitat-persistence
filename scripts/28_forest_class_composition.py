# CCAI OBSOLESCENCIA - PLAN_V4_VOZ_ECOLOGICA.md, Tier 1 punto 5: composicion de la cohorte por
# CLASE de MapBiomas (3 Formacao Florestal, 4 Formacao Savanica, 5 Manguezal, 49 Restinga
# Arborizada) en el pixel EXACTO de cada localidad-año de entrada, y el cruce clase x tasa de
# conversion posterior. UNA consulta nueva a Earth Engine (valor de clase cruda en el punto exacto,
# banda classification_{entry_year}), con el MISMO asset congelado y el MISMO patron de chunking
# con checkpoint que 20_trajectory_extract.py (D1) -- la tasa de conversion posterior NO vuelve a
# Earth Engine, se deriva en pandas de los checkpoints de trayectoria de D1 (buffer 250m) ya
# extraidos, igual que D2/R3 ya hacen.
# Lee data/processed/occurrences_with_cell_id.csv, results/trajectory/run_metadata.json y
# results/trajectory/.chunks/ (buffer 250m). Escribe checkpoints propios en
# results/interpretation/.chunks_forest_class/.
"""
POR QUE
-------
D1 (20_trajectory_extract.py) YA distingue bosque/no-bosque por año en cada pixel, pero LO HACE
COLAPSANDO las clases 3/4/5/49 a UN SOLO bit (remap a 1) -- por diseño, ver bloque POR QUE de D1:
"empaquetar 40 años en un Int64 vía bitwiseOr" exige un bit binario, no un codigo de clase de 6
valores. Esa decision (correcta para D1, que necesita la trayectoria COMPLETA de 40 años barata)
significa que NINGUN archivo ya extraido dice CUAL de las cuatro clases forestales tenia el pixel
en el año de entrada -- hace falta una consulta nueva, pero acotada: UN valor de clase cruda por
localidad-año de entrada (no 40 años, solo el año de entrada), exactamente lo que el plan pide como
"minimo, una consulta barata a GEE".

PIXEL EXACTO (buffer=0), NO EL BUFFER DE 250 M: el pedido es "que clase tenia SU pixel" -- el pixel
de la coordenada registrada, no una mezcla de clases dentro de un buffer de 250 m (que ademas
podria devolver MAS de una clase por localidad, sin una unica respuesta a "que clase es esta
localidad"). Es una definicion espacial DISTINTA de la que usa el resto del pipeline para medir
soporte/perdida (que si usa el buffer, con su propia sensibilidad declarada en 01f/01g) -- las dos
conviven, no se reconcilian: 250 m responde "cuanto bosque queda alrededor", el punto exacto
responde "que tipo de bosque es este sitio". Declarado explicitamente en diagnostics.txt: el numero
de localidades con clase forestal en el punto exacto puede diferir del 62.9% con soporte a 250 m que
reporta el paper (H_i>0 agregado de buffer), sin que eso sea una contradiccion.

CHUNKING POR AÑO DE ENTRADA, NO POR LOCALIDAD (a diferencia de D1): D1 arma UNA imagen con los 40
años ya empaquetados y consulta esa misma imagen para TODAS las localidades a la vez, sin importar
su año de entrada -- posible porque el codigo empaquetado ya tiene los 40 bits. Aca cada
localidad-año necesita la banda classification_{entry_year} ESPECIFICA de SU año de entrada (bandas
distintas para localidades con años de entrada distintos), asi que reduceRegions solo puede
agrupar en un mismo lote las localidades que comparten el MISMO año de entrada -- se agrupa primero
por entry_year (≤40 grupos posibles) y despues, DENTRO de cada año, se aplica el mismo chunk_size=
400 y el mismo patron de checkpoint por archivo que D1 (resumible: un chunk ya escrito en disco se
salta, no se vuelve a consultar).

LOCALIDADES-AÑO CON entry_year > Y_max SE EXCLUYEN DE LA CONSULTA, NO SE CONSULTAN CON LA BANDA MAS
CERCANA: filter_year_upper_bound() (D1) permite year <= Y_max+tolerancia (D0: tolerancia=3) para no
descartar registros con año de reporte ligeramente posterior a la ultima cobertura de MapBiomas --
pero el asset simplemente NO TIENE banda classification_{Y} para Y > Y_max=2024, asi que esas
localidades-año (año de entrada 2025-2027) no pueden clasificarse con este asset. Se cuentan y
excluyen explicitamente (nunca se sustituye por la banda de Y_max ni se descarta en silencio) --
mismo tratamiento, en espiritu, que D2 les da implicitamente (bit_at con indice fuera de rango
devuelve 0 = "sin soporte inicial", ver find_event_bit/bit_at en 21_build_survival_dataset.py).

TASA DE CONVERSION POSTERIOR: MISMA DEFINICION DE EVENTO QUE D2/R3, REUSADA EN PANDAS, SIN VOLVER A
GEE: "conversion posterior" usa el mismo evento que todo el proyecto -- primer spell persistente de
k=2 años consecutivos no-forestales despues de la entrada, censurado en L_k=Y_max-k+1 (misma
censura administrativa que D2 documenta). Se calcula sobre el buffer de 250 m (el buffer PRINCIPAL
del proyecto, D1), ponderando por n_px entre TODOS los codigos de esa localidad -- exactamente
como R3 calcula A_i_t, aca para el complemento (fraccion que SI tuvo un evento confirmado, no la
que retiene soporte). Localidades-año sin seguimiento observable para k=2 (entry_idx > L_k,idx) o
sin soporte forestal en el AGREGADO del buffer de 250 m (total_w==0, que puede diferir del punto
exacto por ser una definicion espacial distinta, ver bloque de arriba) se excluyen del cruce
clase x conversion, contadas aparte en diagnostics.txt -- nunca completadas con un valor inventado.

PROMEDIO SIMPLE ENTRE LOCALIDADES-ENTRADA POR CLASE, NO PONDERADO POR n_px OTRA VEZ: cada
localidad-año ya tiene su PROPIA tasa de conversion agregada (ponderada por n_px DENTRO de esa
localidad, parrafo anterior) -- promediar esas tasas entre localidades-entrada de la misma clase
con peso IGUAL (no volver a ponderar por n_px) le da a cada sitio el mismo voto,
consistente con como el resto del paper reporta promedios entre localidades (p.ej. "mean converted
fraction within 250m es 2.15%" en main.tex es un promedio simple entre localidades, no ponderado
por su n_px individual).

INTERVALOS POR BOOTSTRAP DE BLOQUE ESPACIAL DE 100 KM, MISMA CONVENCION QUE TODO EL PROYECTO: se
reusa assign_spatial_blocks (22_survival_models.py, EPSG:5880) para el IC95% de la tasa de
conversion media por clase -- ninguna consulta nueva a GEE hace falta para esto, los bloques se
derivan de location_id igual que en 24/25_*.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
D1_PATH = SCRIPT_DIR / "20_trajectory_extract.py"
D2_PATH = SCRIPT_DIR / "21_build_survival_dataset.py"
M1_PATH = SCRIPT_DIR / "22_survival_models.py"

CHUNK_SIZE_DEFAULT = 400  # mismo chunk_size que D1, ver bloque POR QUE
BUFFER_M_DEFAULT = 250  # buffer PRINCIPAL del proyecto (D1), usado para la tasa de conversion
K_DEFAULT = 2
N_BOOTSTRAP_DEFAULT = 1000
RANDOM_SEED_DEFAULT = 42
CLASS_LABELS = {
    3: "Formacao Florestal (Forest Formation)",
    4: "Formacao Savanica (Savanna Formation)",
    5: "Manguezal (Mangrove)",
    49: "Restinga Arborizada (Wooded Sandbank Vegetation)",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    obs_root = root
    parser = argparse.ArgumentParser(
        description="PLAN_V4 Tier 1.5 - composicion de la cohorte por clase de MapBiomas en el "
        "punto exacto de entrada, y cruce clase x tasa de conversion posterior (buffer 250m)."
    )
    parser.add_argument("--occurrences-csv", type=Path, default=root / "data" / "processed" / "occurrences_with_cell_id.csv")
    parser.add_argument("--trajectory-dir", type=Path, default=obs_root / "results" / "trajectory")
    parser.add_argument("--checkpoint-dir", type=Path, default=obs_root / "results" / "interpretation" / ".chunks_forest_class")
    parser.add_argument("--output-dir", type=Path, default=obs_root / "results" / "interpretation")
    parser.add_argument("--gee-project", default=None, help="Por defecto, el mismo de D1 (GEE_PROJECT_DEFAULT).")
    parser.add_argument("--mapbiomas-asset", default=None, help="Por defecto, el mismo asset congelado de D1/manifest.")
    parser.add_argument("--forest-classes", type=int, nargs="+", default=None, help="Por defecto, el mismo de D1: [3,4,5,49].")
    parser.add_argument("--buffer-m", type=int, default=BUFFER_M_DEFAULT)
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--max-year-groups", type=int, default=None, help="SOLO pruebas: limitar cuantos años de entrada distintos se consultan en esta corrida.")
    parser.add_argument("--max-chunks-per-year", type=int, default=None, help="SOLO pruebas: limitar chunks nuevos por año en esta corrida.")
    return parser.parse_args()


def require_runtime():
    try:
        import ee
        import numpy as np
        import pandas as pd
        import geopandas as gpd
    except ImportError as exc:
        missing = getattr(exc, "name", "a required package")
        raise SystemExit(
            f"Falta una dependencia: {missing}\nInstalar con:\n"
            "  pip install earthengine-api pandas numpy geopandas"
        ) from exc
    return {"ee": ee, "np": np, "pd": pd, "gpd": gpd}


def load_module(path: Path, name: str):
    import importlib.util
    if not path.exists():
        raise SystemExit(f"No se encontro {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Paso 1: consulta a Earth Engine -- clase cruda en el punto exacto, banda del año de entrada.
# Chunking por (entry_year, chunk_index), checkpoint resumible -- ver bloque POR QUE.
# ---------------------------------------------------------------------------


def query_class_chunk(ee, raw_img, chunk_df, entry_year: int, scale_m: int):
    """Reducer.first() en reduceRegions() nombra la propiedad de salida segun el REDUCER
    ("first"), NO segun el nombre de la banda -- verificado en vivo (a diferencia de
    reduceRegion() singular, que si usa el nombre de banda/rename): un punto de bosque conocido
    (Serra do Mar, cohort_manifest.txt) devuelve {'first': 15, ...}, nunca {'cls': ...} pese al
    rename() previo. Leer 'cls' en vez de 'first' hacia que TODO se interpretara como enmascarado
    -- bug real detectado en el smoke test (800/800 localidades 'enmascaradas'), corregido aca."""
    band = f"classification_{entry_year}"
    img = raw_img.select(band)
    features = []
    for row in chunk_df.itertuples():
        pt = ee.Geometry.Point([row.lon_r, row.lat_r])
        features.append(ee.Feature(pt, {"location_id": row.location_id}))
    fc = ee.FeatureCollection(features)
    reduced = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=scale_m).getInfo()
    rows = []
    for feat in reduced["features"]:
        props = feat["properties"]
        rows.append({
            "location_id": props["location_id"], "entry_year": entry_year,
            "mapbiomas_class": props.get("first"),  # None si el pixel esta enmascarado (fuera de Brasil)
        })
    return rows


def run_gee_class_query(ee, pd, raw_img, pairs_with_coords, checkpoint_dir: Path, chunk_size: int, scale_m: int, max_year_groups, max_chunks_per_year):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    years = sorted(pairs_with_coords["entry_year"].unique())
    if max_year_groups is not None:
        years = years[:max_year_groups]
    n_chunks_done, n_chunks_new, failed = 0, 0, []
    for entry_year in years:
        group = pairs_with_coords.loc[pairs_with_coords["entry_year"] == entry_year].reset_index(drop=True)
        n_chunks_this_year = 0
        for start in range(0, len(group), chunk_size):
            chunk_index = start // chunk_size
            ckpt = checkpoint_dir / f"class_entry{entry_year}_chunk_{chunk_index:04d}.csv"
            if ckpt.exists():
                n_chunks_done += 1
                continue
            if max_chunks_per_year is not None and n_chunks_this_year >= max_chunks_per_year:
                print(f"  año {entry_year}: limite --max-chunks-per-year={max_chunks_per_year} alcanzado.")
                break
            chunk_df = group.iloc[start : start + chunk_size]
            try:
                rows = query_class_chunk(ee, raw_img, chunk_df, entry_year, scale_m)
            except Exception as exc:  # noqa: BLE001 - registrar y seguir
                failed.append((entry_year, chunk_index))
                print(f"  año {entry_year}, chunk {chunk_index}: FALLO ({exc})")
                continue
            pd.DataFrame(rows).to_csv(ckpt, index=False, encoding="utf-8")
            n_chunks_new += 1
            n_chunks_this_year += 1
            n_chunks_done += 1
            print(f"  año {entry_year}, chunk {chunk_index}: {len(chunk_df)} localidades-entrada consultadas.")
    return n_chunks_done, n_chunks_new, failed


def load_class_checkpoints(pd, checkpoint_dir: Path):
    files = sorted(checkpoint_dir.glob("class_entry*_chunk_*.csv"))
    if not files:
        return None
    return pd.concat(
        [pd.read_csv(f, dtype={"location_id": str, "entry_year": "int64"}) for f in files], ignore_index=True
    )


# ---------------------------------------------------------------------------
# Paso 2: tasa de conversion posterior por localidad-entrada -- MISMO evento que D2/R3, derivado
# en pandas de los checkpoints de trayectoria de D1 (buffer 250m), sin volver a Earth Engine.
# ---------------------------------------------------------------------------


def compute_conversion_rate_by_pair(np_mod, pd_mod, d2, loc_code_tables: dict, pairs_df, year_start: int, l_k_idx: int, k: int):
    rows = []
    n_no_followup, n_no_code_table, n_no_buffer_support = 0, 0, 0
    for row in pairs_df.itertuples():
        loc, entry_year = row.location_id, row.entry_year
        entry_idx = entry_year - year_start
        if entry_idx > l_k_idx:
            n_no_followup += 1
            continue
        table = loc_code_tables.get(loc)
        if table is None:
            n_no_code_table += 1
            continue
        codes, weights = table
        total_w, conv_w = 0.0, 0.0
        for c, w in zip(codes, weights):
            c = int(c)
            if not d2.bit_at(c, entry_idx):
                continue
            total_w += w
            ev, _ = d2.find_event_bit(c, entry_idx, k, l_k_idx)
            if ev == 1:
                conv_w += w
        if total_w <= 0:
            n_no_buffer_support += 1
            continue
        rows.append({
            "location_id": loc, "entry_year": entry_year,
            "conversion_rate": conv_w / total_w, "total_w": total_w,
        })
    cols = ["location_id", "entry_year", "conversion_rate", "total_w"]
    return pd_mod.DataFrame(rows, columns=cols), {
        "n_no_followup_censored": n_no_followup,
        "n_no_code_table_250m": n_no_code_table,
        "n_no_buffer_support_at_entry": n_no_buffer_support,
    }


def spatial_block_bootstrap_mean_ci(np_mod, pd_mod, m1, gpd, values_by_location: "pd_mod.Series", n_bootstrap: int, seed: int):
    """IC95% por bootstrap de bloque espacial de 100km sobre la MEDIA de `values_by_location`
    (una fila por localidad-entrada; localidades con >1 entrada aportan >1 valor, se resamplean
    juntas por bloque igual que en el resto del proyecto)."""
    if len(values_by_location) == 0:
        return float("nan"), float("nan")
    blocks = m1.assign_spatial_blocks(gpd, np_mod, pd_mod, pd_mod.Series(values_by_location.index))
    unique_blocks = np_mod.unique(blocks)
    block_to_idx = {b: np_mod.where(blocks == b)[0] for b in unique_blocks}
    vals = values_by_location.to_numpy(dtype=float)
    rng = np_mod.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np_mod.concatenate([block_to_idx[b] for b in sampled_blocks])
        means.append(float(np_mod.mean(vals[idx])))
    means = np_mod.array(means)
    return float(np_mod.percentile(means, 2.5)), float(np_mod.percentile(means, 97.5))


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    ee, np, pd, gpd = rt["ee"], rt["np"], rt["pd"], rt["gpd"]

    print("PLAN_V4 Tier 1.5 - Composicion de la cohorte por clase de MapBiomas (punto exacto, año de entrada)")
    print("===========================================================================================================")

    d1 = load_module(D1_PATH, "d1_trajectory_extract")
    d2 = load_module(D2_PATH, "d2_build_survival_dataset")
    m1 = load_module(M1_PATH, "m1_survival_models")

    gee_project = args.gee_project or d1.GEE_PROJECT_DEFAULT
    mapbiomas_asset = args.mapbiomas_asset or d1.MAPBIOMAS_ASSET_DEFAULT
    forest_classes = args.forest_classes or d1.FOREST_CLASSES_DEFAULT
    print(f"Asset MapBiomas (mismo congelado del manifiesto/D1): {mapbiomas_asset}")
    print(f"Clases forestales: {forest_classes}")

    df = d1.load_occurrences(pd, args.occurrences_csv, d1.YEAR_START_DEFAULT)
    ee.Initialize(project=gee_project)
    print(f"Earth Engine inicializado (project={gee_project})")
    year_max = d1.discover_year_max(ee, mapbiomas_asset, d1.YEAR_START_DEFAULT)
    print(f"Y_max descubierto del asset: {year_max}")
    df = d1.filter_year_upper_bound(df, year_max, d1.YEAR_MAX_TOLERANCE_DEFAULT)

    entry_pairs = d2.build_location_entry_years(pd, df)
    print(f"Localidades-año de entrada (universo, year en [{d1.YEAR_START_DEFAULT}, {year_max + d1.YEAR_MAX_TOLERANCE_DEFAULT}]): {len(entry_pairs):,}")

    n_beyond_ymax = int((entry_pairs["entry_year"] > year_max).sum())
    pairs_queryable = entry_pairs.loc[entry_pairs["entry_year"] <= year_max].copy()
    print(f"Excluidas por año de entrada > Y_max ({year_max}), sin banda disponible en el asset: {n_beyond_ymax:,}.")
    print(f"Localidades-año a consultar: {len(pairs_queryable):,}")

    unique_locs = df.drop_duplicates(subset=["loc_key"])[["loc_key", "lat_r", "lon_r"]].rename(columns={"loc_key": "location_id"})
    pairs_with_coords = pairs_queryable.merge(unique_locs, on="location_id", how="left")
    assert pairs_with_coords["lat_r"].notna().all(), "Localidad-entrada sin coordenadas tras el merge -- bug interno."

    raw_img = ee.Image(mapbiomas_asset)
    n_done, n_new, failed = run_gee_class_query(
        ee, pd, raw_img, pairs_with_coords, args.checkpoint_dir, args.chunk_size, 30,
        args.max_year_groups, args.max_chunks_per_year,
    )
    print(f"\nChunks resueltos (acumulado en checkpoint): {n_done:,}. Nuevos en esta corrida: {n_new:,}. Fallidos: {len(failed)}.")
    if failed:
        print(f"  Chunks fallidos: {failed}")

    class_table = load_class_checkpoints(pd, args.checkpoint_dir)
    if class_table is None or len(class_table) == 0:
        raise SystemExit("Sin checkpoints de clase -- la consulta a Earth Engine no produjo filas.")
    class_table = class_table.drop_duplicates(subset=["location_id", "entry_year"])
    n_masked = int(class_table["mapbiomas_class"].isna().sum())
    print(f"Localidades-año clasificadas (checkpoint): {len(class_table):,}. Pixel enmascarado (sin dato, fuera de Brasil): {n_masked:,}.")

    class_table["mapbiomas_class"] = class_table["mapbiomas_class"].astype("Int64")
    is_forest = class_table["mapbiomas_class"].isin(forest_classes)
    n_not_forest_at_point = int((~is_forest & class_table["mapbiomas_class"].notna()).sum())
    print(f"Punto exacto NO forestal en el año de entrada (clase fuera de {forest_classes}): {n_not_forest_at_point:,}.")

    forest_table = class_table.loc[is_forest].copy()
    n_forest_cohort = len(forest_table)
    print(f"\nCohorte con soporte forestal al entrar (punto exacto, clase en {forest_classes}): {n_forest_cohort:,} localidades-entrada.")

    # --- Composicion de la cohorte ---
    composition = forest_table["mapbiomas_class"].value_counts().rename_axis("mapbiomas_class").reset_index(name="n_pairs")
    composition["pct_of_forest_cohort"] = composition["n_pairs"] / n_forest_cohort * 100.0
    composition["pct_of_all_queried"] = composition["n_pairs"] / len(class_table) * 100.0
    composition["class_label"] = composition["mapbiomas_class"].map(CLASS_LABELS)
    print("\nComposicion de la cohorte por clase (punto exacto, año de entrada):")
    print(composition.to_string(index=False))

    # --- Cruce clase x tasa de conversion posterior (buffer 250m, MISMO evento que D2/R3) ---
    print(f"\nCargando trayectoria de D1 (buffer {args.buffer_m}m) para la tasa de conversion posterior...")
    combined = d1.load_buffer_trajectory(pd, args.trajectory_dir / ".chunks", args.buffer_m)
    if combined is None:
        raise SystemExit(f"Sin checkpoints de trayectoria para buffer {args.buffer_m}m -- correr 20_trajectory_extract.py primero.")
    loc_code_tables = d2.build_location_code_tables(pd, combined)

    meta = d2.load_run_metadata(args.trajectory_dir)
    year_start_d1, year_max_d1 = meta["year_start"], meta["year_max"]
    assert year_max_d1 == year_max, (
        f"Y_max de run_metadata.json ({year_max_d1}) no coincide con el descubierto en vivo ({year_max}) -- "
        "el asset pudo cambiar entre corridas de D1 y esta."
    )
    l_k_idx = (year_max - args.k + 1) - year_start_d1
    print(f"Censura administrativa L_{args.k} = {year_max - args.k + 1} (idx={l_k_idx}).")

    conv_pairs, conv_diag = compute_conversion_rate_by_pair(
        np, pd, d2, loc_code_tables, forest_table[["location_id", "entry_year"]], year_start_d1, l_k_idx, args.k
    )
    print(f"Localidades-entrada con tasa de conversion calculable: {len(conv_pairs):,}. "
          f"Excluidas -- sin seguimiento observable (censura k={args.k}): {conv_diag['n_no_followup_censored']:,}; "
          f"sin trayectoria de 250m para esa localidad: {conv_diag['n_no_code_table_250m']:,}; "
          f"sin soporte forestal en el AGREGADO de 250m (total_w=0): {conv_diag['n_no_buffer_support_at_entry']:,}.")

    conv_with_class = conv_pairs.merge(
        forest_table[["location_id", "entry_year", "mapbiomas_class"]], on=["location_id", "entry_year"], how="left"
    )
    crosstab_rows = []
    for cls in sorted(forest_classes):
        sub = conv_with_class.loc[conv_with_class["mapbiomas_class"] == cls]
        if len(sub) == 0:
            crosstab_rows.append({
                "mapbiomas_class": cls, "n_pairs_conversion": 0, "mean_conversion_rate": float("nan"),
                "conversion_rate_ci95_lo": float("nan"), "conversion_rate_ci95_hi": float("nan"),
            })
            continue
        vals = sub.set_index("location_id")["conversion_rate"]
        mean_rate = float(vals.mean())
        ci_lo, ci_hi = spatial_block_bootstrap_mean_ci(np, pd, m1, gpd, vals, args.n_bootstrap, args.random_seed)
        crosstab_rows.append({
            "mapbiomas_class": cls, "n_pairs_conversion": len(sub), "mean_conversion_rate": mean_rate,
            "conversion_rate_ci95_lo": ci_lo, "conversion_rate_ci95_hi": ci_hi,
        })
    crosstab = pd.DataFrame(crosstab_rows)
    print("\nCruce clase x tasa de conversion posterior (buffer 250m, IC95% bootstrap de bloque espacial):")
    print(crosstab.to_string(index=False))

    # --- salida combinada ---
    out = composition.merge(crosstab, on="mapbiomas_class", how="outer").sort_values("mapbiomas_class")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "forest_class_composition.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nGuardado: {out_csv}")

    diagnostics = [
        "PLAN_V4 Tier 1.5 diagnostics - composicion de la cohorte por clase de MapBiomas",
        "===========================================================================================",
        f"Asset MapBiomas (congelado, mismo de results/frozen/cohort_manifest.txt): {mapbiomas_asset}",
        f"Clases forestales: {forest_classes}. Y_max descubierto: {year_max}.",
        f"Localidades-año de entrada, universo total: {len(entry_pairs):,}.",
        f"Excluidas por entry_year > Y_max (sin banda en el asset): {n_beyond_ymax:,}.",
        f"Localidades-año consultadas a Earth Engine (punto exacto, banda del año de entrada): {len(pairs_queryable):,}.",
        f"Chunks de checkpoint (acumulado): {n_done:,}. Nuevos en esta corrida: {n_new:,}. Fallidos: {len(failed)} {failed if failed else ''}.",
        f"Pixel enmascarado en el punto exacto (sin dato, fuera de Brasil): {n_masked:,}.",
        f"Punto exacto NO forestal en el año de entrada: {n_not_forest_at_point:,}.",
        f"Cohorte con soporte forestal al entrar (punto exacto): {n_forest_cohort:,} localidades-entrada "
        f"({n_forest_cohort/len(class_table):.1%} de las consultadas).",
        "",
        "NOTA -- esta cifra es una definicion espacial DISTINTA de la de D2/R3 (soporte agregado en "
        "el buffer de 250m, H_i>0): compara punto exacto vs. area, no deberian coincidir numericamente "
        "y su diferencia no es un error (ver bloque POR QUE).",
        "",
        "COMPOSICION DE LA COHORTE POR CLASE (punto exacto, año de entrada):",
        composition.to_string(index=False),
        "",
        f"TASA DE CONVERSION POSTERIOR (buffer {args.buffer_m}m, evento identico a D2/R3: primer spell "
        f"persistente de k={args.k} años no-forestales tras la entrada, censura L_{args.k}={year_max-args.k+1}):",
        f"  Localidades-entrada con tasa calculable: {len(conv_pairs):,} de {n_forest_cohort:,} en la cohorte forestal.",
        f"  Excluidas -- sin seguimiento observable (censura): {conv_diag['n_no_followup_censored']:,}.",
        f"  Excluidas -- sin trayectoria de {args.buffer_m}m extraida para esa localidad: {conv_diag['n_no_code_table_250m']:,}.",
        f"  Excluidas -- sin soporte forestal en el AGREGADO de {args.buffer_m}m (desacuerdo punto vs. buffer): {conv_diag['n_no_buffer_support_at_entry']:,}.",
        crosstab.to_string(index=False),
    ]
    diag_path = args.output_dir / "forest_class_composition_diagnostics.txt"
    diag_path.write_text("\n".join(str(l) for l in diagnostics) + "\n", encoding="utf-8")
    print(f"\nGuardado: {diag_path}")


if __name__ == "__main__":
    main()
