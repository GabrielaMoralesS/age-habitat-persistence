# CCAI OBSOLESCENCIA - B2 (PLAN_V3_EJECUTABLE.md): capa de DECISION sobre el backtest de R3 --
# "si tuvieramos presupuesto para re-verificar B localidades, cuales elegir, y cuanto mejor es eso
# que elegir al azar". Lee results/trajectory/.chunks/, results/trajectory/run_metadata.json,
# results/survival/dataset_k{K}_buffer{B}m.csv y data/processed/occurrences_with_cell_id.csv.
# Reusa (import dinamico) 20/21/22/24 -- no duplica extraccion, supervivencia ni landmark boosting.
# NO CONSULTA EARTH ENGINE.
"""
POR QUE
-------
SCORE r-bar-dagger_i: PROMEDIO SOBRE DRAWS DE (indicador de riesgo x prediccion), NO A_i(t0)
MULTIPLICADO APARTE. La formula de B2 en el plan esta escrita en notacion de razon de
supervivencia Rayleigh (heredada de una version del plan anterior a que existiera el landmark
model de R3) pero B1 ya establece explicitamente que la prediccion operativa es la del landmark
boosting, "NO P(T<=t0+5|T>t0,x) (diseno de supervivencia ya sustituido)". Se implementa por tanto:

    r_bar_i = (1/M) * sum_m  I_im(t0) * p_hat_im(t0,5)

M = numero TOTAL de draws Monte Carlo de esa localidad (20, fijo para todas las localidades --
los draws sin riesgo NO se excluyen del denominador, contribuyen 0). I_im(t0) = 1 si el draw m
esta en el risk set en t0 (misma definicion formal de A_i(t0) que B1/B2: forestal al entrar,
forestal en t0, sin evento persistente confirmado hasta L_k,t0). p_hat_im es la probabilidad del
landmark boosting YA VALIDADO en R3 (AUROC~0.79 en el test congelado), condicionada a estar en
riesgo (24_landmark_forecast.py solo genera filas para draws que pasan el gate de riesgo).

Por que (1/M)*sum_m I_im ya ES A_i(t0), sin necesidad de multiplicar aparte: es exactamente la
fraccion de los MISMOS 20 draws que produjeron las predicciones que caen en el risk set -- un
estimador de A_i(t0) consistente por construccion con p_hat, sin el riesgo de mezclar un A_i(t0)
calculado con una convencion distinta (p.ej. ponderado por n_px de TODOS los pixeles del buffer,
como hace la covariable A_i_t) con un p_hat calculado sobre otro universo. Multiplicar el score
final por A_i_t (la covariable) ademas del gate por draw DUPLICARIA la misma senal por dos vias
distintas -- ver el parrafo siguiente.

NO HAY DOBLE CONTEO CON LA COVARIABLE A_i_t: p_hat esta CONDICIONADA a estar en riesgo -- responde
"dado que esta posicion esta en riesgo, se convierte en 5 anos?". El factor (1/M)*sum I_im
responde "que fraccion de las posiciones muestreadas de esta localidad estan siquiera en riesgo?".
Son los dos factores de P(se convierte Y estaba en riesgo) = A_i(t0) x P(se convierte | en
riesgo). Como COVARIABLE de entrada del landmark boosting, A_i_t (fraccion ponderada por pixel de
TODO el buffer) le da al modelo senal ecologica adicional (que tan intacto esta el entorno
modula el hazard) -- un rol distinto, no redundante, del factor de riesgo por draw que entra en
r_bar_i desde afuera del modelo.

LINEAS BASE RAYLEIGH: MISMO ESQUEMA DE PROMEDIO POR DRAW, CON LA RAZON DE SUPERVIVENCIA EN VEZ DEL
BOOSTING. Rayleigh-30 y Rayleigh estimado (sigma_hat_t reestimado en el mismo landmark t0, sin
fuga, igual que en 24_landmark_forecast.py) usan la MISMA formula r_bar_i = (1/M)*sum_m I_im(t0) *
[1 - S(a_i+5|sigma) / S(a_i|sigma)], sustituyendo p_hat por la razon de supervivencia. El resultado
esperado es NULO (ya se demostro en R3 que Rayleigh es indistinguible de un ranking aleatorio,
AUROC~0.47) -- reportarlo aca es deliberado: cierra el argumento de que ponderar por edad tampoco
sirve para ASIGNAR PRESUPUESTO de reverificacion, no solo que no discrimina en abstracto.

"Mas antiguo" y "perdida reciente" (SIN PONDERAR) son rankings deterministas por localidad (edad y
perdida-5y no varian por draw, ver 24_landmark_forecast.py), asi que su score es directo, sin el
promedio por draw. Se reportan TAMBIEN sus variantes PONDERADAS
(mas_antiguo_ponderado, perdida_reciente_ponderado), con el MISMO esquema r_bar_i = (1/M)*sum_m
I_im(t0)*covariable que boosting/Rayleigh -- las dos versiones de cada baseline se reportan juntas
y rotuladas explicitamente en la columna "ponderacion" (sin_ponderar / ponderado_I_im / n/a para
aleatorio), nunca mezcladas en una sola fila. Diagnostico verificado empiricamente (ver
diagnostic_top100_A_i_check.csv): el top-100 SIN PONDERAR de perdida_reciente tiene A_i(t0) medio
0.015 contra 0.465 de landmark_boosting -- 81/100 localidades seleccionadas estan en riesgo en
solo 1 de los 20 draws, porque "mucha perdida reciente" y "casi sin soporte restante en t0" son la
MISMA propiedad vista de dos formas: un sitio que ya perdio casi todo su bosque no tiene mucho mas
que perder, y una posicion muestreada rara vez cae todavia forestal ahi. Es un resultado REAL de
esa construccion, no un bug -- la version ponderada corrige exactamente ese mecanismo. El universo
de localidades elegibles (union de las que aparecen en al menos un draw) es el mismo para TODOS
los metodos, porque todos se calculan sobre las mismas filas del dataset del landmark t0
(24.build_landmark_dataset), que ya excluye localidades nunca-en-riesgo.

DESEMPATE: SEEDED, NO ARBITRARIO POR ORDEN DE FILA. Con variables de baja cardinalidad (edad en
años enteros, o razon de supervivencia Rayleigh que es funcion monotona de esa misma edad), decenas
de localidades comparten el score exacto del corte B-esimo -- un `argsort` estable ordenaria esos
empates por el orden en que aparecen en el DataFrame, un artefacto de pipeline sin significado
(verificado: mas_antiguo tiene 141 localidades empatadas en el valor de corte para B=100).
Se resuelve con una clave aleatoria FIJA (semilla unica, una sola vez, reusada identica en la tabla
principal, el bootstrap espacial y el bloque de desempate) en vez de la determinista-por-orden-de-
fila. Ademas, para cada (metodo, B) se reporta CUANTAS localidades estan empatadas en el valor de
corte -- y cuando esas localidades exceden los cupos disponibles (empate real, no solo cosmetico),
Lift@B/CapturedEvents@B se promedian sobre 100 desempates aleatorios independientes (outcome
promediado entre draws, Y-barra_i, misma logica de tratabilidad computacional que el bootstrap
espacial) en vez de reportar un unico top-B arbitrario.

"Aleatorio" no es un score fijo: se simula por separado (>=1000 repeticiones de seleccion de B
unidades sin reemplazo) porque (a) el plan lo pide explicitamente como recuento de repeticiones,
no como expectativa cerrada, y (b) sirve dos propositos -- baseline propio con su distribucion, y
denominador E[bajo seleccion aleatoria] de Lift@B para cada draw m.

UNIDAD DE PRESUPUESTO -- DECIDIDO por el plan, no una eleccion de este script: PRINCIPAL =
localidades (B en {100,500,1000} localidades verificables); SENSIBILIDAD = celdas de 5 km
(mismo B, pero top-B CELDAS, agregando R_c^sum = suma de r_bar_i de las localidades de la celda);
VISUALIZACION = agregacion a la MISMA grilla de 5 km, de TODAS las celdas, no solo el top-B. Las
formulas del plan (Sigma_{c in TopB} Y_c) son genericas en "c": para el principal, c es la propia
localidad (celda unitaria); para sensibilidad, c es la celda de 5 km real. Una sola funcion de
evaluacion (evaluate_topb) sirve para ambos casos.

GRILLA DE 5 KM REUSADA, NO INVENTADA: se reusa la columna cell_id ya presente en
data/processed/occurrences_with_cell_id.csv (grilla EPSG:5880 de 02/06/09/data/grid_5km.gpkg), no
se recalcula un grid nuevo -- evita introducir una segunda definicion de "celda de 5km" en el
proyecto. Si esa columna faltara en el csv pasado por --occurrences-csv, se cae a un grid de
respaldo (mismo CRS EPSG:5880 que assign_spatial_blocks, floor-division por --grid-cell-size-m),
documentado explicitamente como respaldo en diagnostics.txt.

Y_i^(m) (outcome de una localidad en el draw m) = Z de esa fila en el dataset del landmark t0 si
la localidad esta en riesgo en ese draw (existe la fila), 0 en caso contrario -- no puede
"convertirse en la ventana (t0,t0+5]" una localidad que ya no estaba en riesgo en t0 para ese
draw especifico (o ya se habia perdido antes, o la posicion muestreada ya no es forestal). Esto
NO es lo mismo que excluirla del universo: sigue contando en el denominador de Recall@B.

DOS FUENTES DE INCERTIDUMBRE, REPORTADAS POR SEPARADO (nunca mezcladas en una unica distribucion):
(1) draws Monte Carlo de sensibilidad posicional -- CapturedEvents/Recall/Lift@B recalculados para
CADA uno de los 20 draws, con el MISMO ranking fijo (r_bar_i no cambia por draw, solo el outcome).
(2) bootstrap de bloque espacial (100 km, igual CRS/tamano que 22/24) -- remuestrea localidades
(agrupadas por bloque) con reemplazo y recalcula el ranking Y las metricas dentro de cada
remuestra, usando el outcome PROMEDIADO entre los 20 draws (Y-barra_i) como medida unica por
remuestra -- mezclar bootstrap espacial x draw completo (1000 x 20) es computacionalmente
excesivo para el beneficio marginal; promediar los draws en este eje concentra el computo en la
incertidumbre espacial, que es la que el bootstrap esta disenado para capturar. El bootstrap
espacial de la capa de decision se reporta SOLO para el presupuesto principal (localidades) --
extenderlo a celdas exigiria re-agregar celdas dentro de cada remuestra de bloques, no solo
reindexar, y no esta pedido explicitamente; queda como limitacion documentada.

MAPA PROSPECTIVO: EXTRAPOLACION NO OBSERVADA, NO "CASI GRATIS" (advertencia literal del plan).
Reentrena el landmark boosting con los MISMOS hiperparametros ya seleccionados en el backtest
(rolling-origin sobre {1993,1998,2003,2008}, nunca vueltos a buscar aca -- "el pipeline
prospectivo debe reproducir EXACTAMENTE el backtest") sobre los landmarks
{1993,1998,2003,2008,2013,2018} (Y_max=2024 permite hasta t=2018 con outcome completo; 2019-2023
NO se usan como landmarks de entrenamiento del prospectivo, siguiendo la secuencia literal del
plan). Predice el risk set en Y_max con covariables hasta Y_max -- sin bit de evento (Y_max+1..
Y_max+5 no existe todavia en la trayectoria: llamar a find_event_bit con un horizonte fuera de
rango daria un (evento, duracion) corrupto por desplazamiento de bits fuera de rango, no un error
visible, asi que el dataset prospectivo usa un constructor dedicado que NUNCA calcula Z). El
resultado se guarda con el rotulo explicito "EXTRAPOLACION PROSPECTIVA NO OBSERVADA" en cada
archivo -- el backtest valida el PROCEDIMIENTO, el mapa prospectivo es el producto operativo, pero
no esta validado directamente (no hay 2029 todavia).
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
D1_PATH = SCRIPT_DIR / "20_trajectory_extract.py"
D2_PATH = SCRIPT_DIR / "21_build_survival_dataset.py"
M1_PATH = SCRIPT_DIR / "22_survival_models.py"
R3_PATH = SCRIPT_DIR / "24_landmark_forecast.py"

K_DEFAULT = 2
BUFFER_M_DEFAULT = 250
HORIZON_YEARS_DEFAULT = 5
TRAIN_LANDMARKS_DEFAULT = [1993, 1998, 2003, 2008]
DECISION_LANDMARK_DEFAULT = 2014
PROSPECTIVE_TRAIN_LANDMARKS_DEFAULT = [1993, 1998, 2003, 2008, 2013, 2018]
YEAR_MAX_TOLERANCE_DEFAULT = 3
N_DRAWS_DEFAULT = 20
RANDOM_SEED_DEFAULT = 42
SIGMA_FIXED_DEFAULT = 30.0
BUDGETS_DEFAULT = [100, 500, 1000]
N_RANDOM_REPEATS_DEFAULT = 1000
N_BOOTSTRAP_DEFAULT = 1000
GRID_CELL_SIZE_M_DEFAULT = 5000
N_TIEBREAK_REPEATS_DEFAULT = 100
TIEBREAK_SEED_DEFAULT = 4242
DECISION_METHODS = [
    "landmark_boosting", "landmark_logistic", "rayleigh_30", "rayleigh_estimado",
    "mas_antiguo", "mas_antiguo_ponderado",
    "perdida_reciente", "perdida_reciente_ponderado",
]
METHOD_WEIGHTING_LABEL = {
    "landmark_boosting": "ponderado_I_im", "landmark_logistic": "ponderado_I_im",
    "rayleigh_30": "ponderado_I_im", "rayleigh_estimado": "ponderado_I_im",
    "mas_antiguo": "sin_ponderar", "mas_antiguo_ponderado": "ponderado_I_im",
    "perdida_reciente": "sin_ponderar", "perdida_reciente_ponderado": "ponderado_I_im",
    "aleatorio": "n/a",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    obs_root = root
    parser = argparse.ArgumentParser(
        description="B2 - capa de decision (CapturedEvents@B, Recall@B, Lift@B) sobre el backtest "
        "de R3, mas el mapa prospectivo declarado como extrapolacion no observada."
    )
    parser.add_argument("--occurrences-csv", type=Path, default=root / "data" / "processed" / "occurrences_with_cell_id.csv")
    parser.add_argument("--trajectory-dir", type=Path, default=obs_root / "results" / "trajectory")
    parser.add_argument("--survival-dir", type=Path, default=obs_root / "results" / "survival")
    parser.add_argument("--output-dir", type=Path, default=obs_root / "results" / "decision")
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--buffer-m", type=int, default=BUFFER_M_DEFAULT)
    parser.add_argument("--horizon-years", type=int, default=HORIZON_YEARS_DEFAULT)
    parser.add_argument("--train-landmarks", type=int, nargs="+", default=TRAIN_LANDMARKS_DEFAULT)
    parser.add_argument("--decision-landmark", type=int, default=DECISION_LANDMARK_DEFAULT)
    parser.add_argument("--prospective-train-landmarks", type=int, nargs="+", default=PROSPECTIVE_TRAIN_LANDMARKS_DEFAULT)
    parser.add_argument("--year-max-tolerance", type=int, default=YEAR_MAX_TOLERANCE_DEFAULT)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    parser.add_argument("--max-draws", type=int, default=None, help="SOLO pruebas.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--sigma-fixed", type=float, default=SIGMA_FIXED_DEFAULT)
    parser.add_argument("--budgets", type=int, nargs="+", default=BUDGETS_DEFAULT)
    parser.add_argument("--n-random-repeats", type=int, default=N_RANDOM_REPEATS_DEFAULT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--grid-cell-size-m", type=int, default=GRID_CELL_SIZE_M_DEFAULT)
    parser.add_argument("--n-tiebreak-repeats", type=int, default=N_TIEBREAK_REPEATS_DEFAULT)
    parser.add_argument("--tiebreak-seed", type=int, default=TIEBREAK_SEED_DEFAULT)
    parser.add_argument("--skip-prospective", action="store_true", help="SOLO pruebas: omite el mapa prospectivo.")
    return parser.parse_args()


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import lightgbm as lgb
        from sklearn.metrics import average_precision_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        missing = getattr(exc, "name", "a required package")
        raise SystemExit(
            f"Falta una dependencia: {missing}\nInstalar con:\n"
            "  pip install pandas numpy geopandas lightgbm scikit-learn"
        ) from exc
    return {
        "np": np, "pd": pd, "gpd": gpd, "lgb": lgb, "average_precision_score": average_precision_score,
        "LogisticRegression": LogisticRegression, "StandardScaler": StandardScaler,
        "SimpleImputer": SimpleImputer, "make_pipeline": make_pipeline,
    }


def load_module(path: Path, name: str):
    import importlib.util
    if not path.exists():
        raise SystemExit(f"No se encontro {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Grilla de 5 km: reusa cell_id si esta en el csv de ocurrencias, si no cae a un grid de respaldo
# en el mismo CRS (EPSG:5880) que assign_spatial_blocks -- ver bloque POR QUE.
# ---------------------------------------------------------------------------


def build_cell_id_lookup(pd, gpd, np_mod, df, grid_cell_size_m: int):
    diag = {}
    if "cell_id" in df.columns and df["cell_id"].notna().any():
        by_loc = df.dropna(subset=["cell_id"]).groupby("loc_key")["cell_id"]
        n_multi = int((by_loc.nunique() > 1).sum())
        cell_id_by_location = by_loc.agg(lambda s: s.mode().iat[0])
        diag["fuente"] = "cell_id de occurrences_with_cell_id.csv (grilla EPSG:5880 de data/grid_5km.gpkg)"
        diag["localidades_con_cell_id_inconsistente"] = n_multi
        missing_locs = set(df["loc_key"].unique()) - set(cell_id_by_location.index)
        if missing_locs:
            fallback = _fallback_cell_ids(pd, gpd, np_mod, df.loc[df["loc_key"].isin(missing_locs)], grid_cell_size_m)
            cell_id_by_location = pd.concat([cell_id_by_location, fallback])
            diag["localidades_sin_cell_id_en_csv_completadas_con_respaldo"] = len(missing_locs)
        else:
            diag["localidades_sin_cell_id_en_csv_completadas_con_respaldo"] = 0
    else:
        cell_id_by_location = _fallback_cell_ids(pd, gpd, np_mod, df, grid_cell_size_m)
        diag["fuente"] = f"RESPALDO: grid propio EPSG:5880, {grid_cell_size_m} m (cell_id ausente en el csv de ocurrencias)"
        diag["localidades_con_cell_id_inconsistente"] = 0
        diag["localidades_sin_cell_id_en_csv_completadas_con_respaldo"] = len(cell_id_by_location)
    return cell_id_by_location, diag


def _fallback_cell_ids(pd, gpd, np_mod, df, grid_cell_size_m: int):
    unique = df.drop_duplicates(subset=["loc_key"])[["loc_key", "lat_r", "lon_r"]]
    points = gpd.GeoSeries(gpd.points_from_xy(unique["lon_r"], unique["lat_r"]), crs="EPSG:4326").to_crs("EPSG:5880")
    cx = np_mod.floor(points.x.to_numpy() / grid_cell_size_m).astype(int)
    cy = np_mod.floor(points.y.to_numpy() / grid_cell_size_m).astype(int)
    ids = [f"grid5km_{a}_{b}" for a, b in zip(cx, cy)]
    return pd.Series(ids, index=unique["loc_key"].to_numpy())


# ---------------------------------------------------------------------------
# r_bar_i por metodo -- promedio sobre draws del indicador de riesgo x prediccion (o razon de
# supervivencia Rayleigh), y matriz de outcomes Y_i^(m) -- ver bloque POR QUE.
# ---------------------------------------------------------------------------


def build_r_bar_scores(pd, decision_table, m_total: int):
    r_bar = {}
    # Ponderados por I_im: MISMO esquema r_bar_i = (1/M)*sum_m I_im*valor que boosting/Rayleigh --
    # para mas_antiguo_ponderado/perdida_reciente_ponderado, "valor" es la covariable deterministica
    # (age_at_t/loss_5y), pero SOLO se suma sobre las filas que existen (I_im=1), igual que los
    # scores de prediccion.
    for method, col in [
        ("landmark_boosting", "score_landmark_boosting"), ("landmark_logistic", "score_landmark_logistic"),
        ("rayleigh_30", "score_rayleigh_30"), ("rayleigh_estimado", "score_rayleigh_estimado"),
        ("mas_antiguo_ponderado", "age_at_t"), ("perdida_reciente_ponderado", "loss_5y"),
    ]:
        r_bar[method] = decision_table.groupby("location_id")[col].sum() / float(m_total)
    # Sin ponderar: covariable directa, sin importar en cuantos draws esta la localidad en riesgo.
    for method, col in [("mas_antiguo", "age_at_t"), ("perdida_reciente", "loss_5y")]:
        r_bar[method] = decision_table.groupby("location_id")[col].first()
    return r_bar


def build_ai_and_ndraws_lookup(pd, decision_table):
    """A_i(t0) (covariable agregada, constante por localidad) y numero de draws (de M) en que la
    localidad esta en riesgo -- para el diagnostico top-100 (ver bloque POR QUE)."""
    a_i_t = decision_table.groupby("location_id")["A_i_t"].first()
    n_draws_at_risk = decision_table.groupby("location_id")["draw_id"].nunique()
    return a_i_t, n_draws_at_risk


def build_outcome_matrix(pd, decision_table, draws_available: list):
    pivot = decision_table.pivot_table(index="location_id", columns="draw_id", values="Z", aggfunc="first", fill_value=0)
    pivot = pivot.reindex(columns=draws_available, fill_value=0)
    return pivot.astype(int)


def aggregate_to_cells(pd, r_bar: dict, outcome_matrix, cell_id_by_location):
    universe = outcome_matrix.index
    cell_of = cell_id_by_location.reindex(universe)
    missing = cell_of.isna().sum()
    valid = cell_of.notna()
    cell_of = cell_of.loc[valid]
    outcome_c = outcome_matrix.loc[valid.index[valid]].groupby(cell_of).sum()
    r_bar_c = {}
    for method, series in r_bar.items():
        s = series.reindex(universe).loc[valid.index[valid]]
        r_bar_c[method] = s.groupby(cell_of).sum()
    return r_bar_c, outcome_c, int(missing)


# ---------------------------------------------------------------------------
# Evaluacion Top-B: CapturedEvents@B, Recall@B, Precision@B (secundaria) por draw, para un ranking
# FIJO (r_bar_i). "c" es generico: localidad (principal) o celda de 5 km (sensibilidad).
# ---------------------------------------------------------------------------


def generate_tiebreak_key(np_mod, pd, universe_index, seed: int):
    """Clave aleatoria FIJA (una sola vez) para desempatar rankings con baja cardinalidad de score
    -- ver bloque POR QUE. Reusada identica en la tabla principal, el bootstrap espacial y el
    bloque de desempate, para que el desempate sea consistente entre las tres piezas."""
    rng = np_mod.random.default_rng(seed)
    return pd.Series(rng.random(len(universe_index)), index=universe_index)


def evaluate_topb_fixed_ranking(np_mod, pd, r_bar_series, outcome_matrix, budgets: list, tiebreak_key):
    """Ranking construido UNA sola vez con r_bar_series, evaluado contra CADA columna (draw) de
    outcome_matrix por separado -- ver bloque POR QUE. outcome_matrix ya debe estar alineada al
    mismo universo (index) que r_bar_series. Empates resueltos con tiebreak_key (seeded), NO por
    orden de fila del DataFrame -- y diagnosticados (n_tied_at_cutoff) para cada B."""
    aligned = r_bar_series.reindex(outcome_matrix.index).fillna(0.0)
    values = aligned.to_numpy()
    tb = tiebreak_key.reindex(outcome_matrix.index).to_numpy()
    order = np_mod.lexsort((tb, -values))  # primario: valor descendente; empates: tiebreak_key ascendente
    sorted_values = values[order]
    outcome_vals = outcome_matrix.to_numpy(dtype=float)
    total_by_draw = outcome_vals.sum(axis=0)
    rows = []
    for B in budgets:
        b_eff = min(B, len(order))
        cutoff_val = sorted_values[b_eff - 1]
        n_tied_at_cutoff = int((values == cutoff_val).sum())
        n_above_cutoff = int((values > cutoff_val).sum())
        needs_tiebreak = n_tied_at_cutoff > (b_eff - n_above_cutoff)
        top_idx = order[:b_eff]
        sub = outcome_vals[top_idx, :]
        captured = sub.sum(axis=0)
        recall = np_mod.divide(captured, total_by_draw, out=np_mod.full_like(captured, np_mod.nan), where=total_by_draw > 0)
        precision = (sub > 0).mean(axis=0)
        rows.append({"B": B, "captured": captured, "recall": recall, "precision": precision,
                      "n_tied_at_cutoff": n_tied_at_cutoff, "needs_tiebreak": needs_tiebreak})
    return rows


# ---------------------------------------------------------------------------
# Desempate promediado (SOLO metodo/B con empate real en el corte, ver bloque POR QUE): >=100
# desempates aleatorios independientes de las localidades empatadas en el valor de corte, outcome
# promediado entre draws (Y-barra_i, misma logica de tratabilidad que el bootstrap espacial),
# denominador de Lift@B por expectativa CERRADA de seleccion aleatoria (exacta para muestreo
# simple sin reemplazo: B * media(Y-barra) sobre el universo), no simulada de nuevo.
# ---------------------------------------------------------------------------


def tiebreak_averaged_topb(np_mod, r_bar_series, outcome_mean_series, budgets: list, n_repeats: int, rng):
    aligned = r_bar_series.reindex(outcome_mean_series.index).fillna(0.0)
    values = aligned.to_numpy()
    y = outcome_mean_series.to_numpy()
    sorted_desc = np_mod.sort(values)[::-1]
    results = {}
    for B in budgets:
        b_eff = min(B, len(values))
        cutoff_val = sorted_desc[b_eff - 1]
        above_idx = np_mod.where(values > cutoff_val)[0]
        tied_idx = np_mod.where(values == cutoff_val)[0]
        remaining = b_eff - len(above_idx)
        e_random = b_eff * float(y.mean())
        if len(tied_idx) <= remaining:
            top_idx = np_mod.concatenate([above_idx, tied_idx[:remaining]]) if remaining > 0 else above_idx
            captured = float(y[top_idx].sum())
            lift = captured / e_random if e_random > 0 else float("nan")
            results[B] = {"n_tied_at_cutoff": int(len(tied_idx)), "needed_tiebreak": False,
                          "captured": {"media": captured, "sd": 0.0, "p2.5": captured, "p97.5": captured},
                          "lift": {"media": lift, "sd": 0.0, "p2.5": lift, "p97.5": lift}}
            continue
        captured_reps = np_mod.empty(n_repeats)
        for r in range(n_repeats):
            chosen = rng.choice(tied_idx, size=remaining, replace=False)
            top_idx = np_mod.concatenate([above_idx, chosen])
            captured_reps[r] = y[top_idx].sum()
        lift_reps = captured_reps / e_random if e_random > 0 else np_mod.full(n_repeats, np_mod.nan)
        results[B] = {"n_tied_at_cutoff": int(len(tied_idx)), "needed_tiebreak": True,
                      "captured": {"media": float(captured_reps.mean()), "sd": float(captured_reps.std()),
                                    "p2.5": float(np_mod.percentile(captured_reps, 2.5)), "p97.5": float(np_mod.percentile(captured_reps, 97.5))},
                      "lift": {"media": float(lift_reps.mean()), "sd": float(lift_reps.std()),
                                "p2.5": float(np_mod.percentile(lift_reps, 2.5)), "p97.5": float(np_mod.percentile(lift_reps, 97.5))}}
    return results


def summarize_across_draws(np_mod, values: "np.ndarray"):
    return {
        "media": float(np_mod.mean(values)), "sd": float(np_mod.std(values)),
        "p2.5": float(np_mod.percentile(values, 2.5)), "p97.5": float(np_mod.percentile(values, 97.5)),
        "min": float(np_mod.min(values)), "max": float(np_mod.max(values)),
    }


# ---------------------------------------------------------------------------
# Baseline "aleatorio": simulado con >=1000 repeticiones (no expectativa cerrada, pedido literal
# del plan) -- sirve de baseline propio Y de denominador de Lift@B por draw. Un solo permutation
# por repeticion cubre todos los B a la vez (prefijos anidados de una muestra sin reemplazo).
# ---------------------------------------------------------------------------


def random_baseline_captured(np_mod, outcome_matrix, budgets: list, n_repeats: int, rng):
    """Devuelve, por B: 'captured' (n_repeats, n_draws) y 'precision' (n_repeats, n_draws) --
    esta ultima via cumsum del indicador binario (Y>0) con el MISMO permutation, sin repetir el
    muestreo."""
    n_units = outcome_matrix.shape[0]
    max_b = min(max(budgets), n_units)
    outcome_vals = outcome_matrix.to_numpy(dtype=float)
    positive_vals = (outcome_vals > 0).astype(float)
    captured_by_b = {B: np_mod.empty((n_repeats, outcome_vals.shape[1])) for B in budgets}
    precision_by_b = {B: np_mod.empty((n_repeats, outcome_vals.shape[1])) for B in budgets}
    for r in range(n_repeats):
        perm = rng.permutation(n_units)[:max_b]
        cum = np_mod.cumsum(outcome_vals[perm, :], axis=0)
        cum_pos = np_mod.cumsum(positive_vals[perm, :], axis=0)
        for B in budgets:
            b_eff = min(B, max_b)
            captured_by_b[B][r] = cum[b_eff - 1]
            precision_by_b[B][r] = cum_pos[b_eff - 1] / b_eff
    return captured_by_b, precision_by_b  # dict B -> array (n_repeats, n_draws) each


# ---------------------------------------------------------------------------
# Bootstrap de bloque espacial (SOLO presupuesto principal, localidades) -- outcome promediado
# entre draws (Y-barra_i) para mantener el computo tratable, ver bloque POR QUE. El ranking se
# RECALCULA dentro de cada remuestra (mismos valores de r_bar_i, orden restringido al remuestreo).
# ---------------------------------------------------------------------------


def spatial_bootstrap_topb(np_mod, pd, gpd, r_bar_series, outcome_mean_series, location_ids, budgets: list, n_bootstrap: int, rng, m1, tiebreak_key):
    blocks = m1.assign_spatial_blocks(gpd, np_mod, pd, pd.Series(location_ids))
    unique_blocks = np_mod.unique(blocks)
    block_to_idx = {b: np_mod.where(blocks == b)[0] for b in unique_blocks}
    r_vals = r_bar_series.reindex(location_ids).fillna(0.0).to_numpy()
    y_vals = outcome_mean_series.reindex(location_ids).fillna(0.0).to_numpy()
    tb_vals = tiebreak_key.reindex(location_ids).to_numpy()
    results = {B: {"captured": [], "recall": [], "precision": []} for B in budgets}
    for _ in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np_mod.concatenate([block_to_idx[b] for b in sampled_blocks])
        r_sub, y_sub, tb_sub = r_vals[idx], y_vals[idx], tb_vals[idx]
        order = np_mod.lexsort((tb_sub, -r_sub))
        total = y_sub.sum()
        for B in budgets:
            b_eff = min(B, len(order))
            top_idx = order[:b_eff]
            captured = float(y_sub[top_idx].sum())
            results[B]["captured"].append(captured)
            results[B]["recall"].append(captured / total if total > 0 else float("nan"))
            results[B]["precision"].append(float((y_sub[top_idx] > 0).mean()))
    return results


# ---------------------------------------------------------------------------
# Orquestacion por unidad (localidad o celda): metricas por draw para los 5 metodos con ranking
# fijo, distribucion de "aleatorio" (media por draw sobre las repeticiones), Lift@B por draw, y
# (opcional, solo principal) bootstrap de bloque espacial.
# ---------------------------------------------------------------------------


def run_decision_evaluation(rt, m1, unit_label: str, r_bar: dict, outcome_matrix, budgets: list,
                             n_random_repeats: int, rng_random, do_spatial_bootstrap: bool,
                             n_bootstrap: int, rng_boot, tiebreak_key, n_tiebreak_repeats: int,
                             rng_tiebreak, location_ids_for_blocks=None):
    np_mod, pd, gpd = rt["np"], rt["pd"], rt["gpd"]
    random_captured, random_precision = random_baseline_captured(np_mod, outcome_matrix, budgets, n_random_repeats, rng_random)
    random_mean_per_draw = {B: random_captured[B].mean(axis=0) for B in budgets}
    random_precision_mean_per_draw = {B: random_precision[B].mean(axis=0) for B in budgets}

    rows = []
    outcome_mean = outcome_matrix.mean(axis=1)
    for method in DECISION_METHODS:
        per_b = evaluate_topb_fixed_ranking(np_mod, pd, r_bar[method], outcome_matrix, budgets, tiebreak_key)
        boot = None
        if do_spatial_bootstrap:
            boot = spatial_bootstrap_topb(np_mod, pd, gpd, r_bar[method], outcome_mean,
                                           location_ids_for_blocks, budgets, n_bootstrap, rng_boot, m1, tiebreak_key)
        tiebreak_result = tiebreak_averaged_topb(np_mod, r_bar[method], outcome_mean, budgets, n_tiebreak_repeats, rng_tiebreak)
        for entry in per_b:
            B = entry["B"]
            lift = entry["captured"] / np_mod.where(random_mean_per_draw[B] > 0, random_mean_per_draw[B], np_mod.nan)
            row = {"unidad": unit_label, "metodo": method, "ponderacion": METHOD_WEIGHTING_LABEL[method], "B": B,
                   "n_tied_en_cutoff": entry["n_tied_at_cutoff"], "necesita_desempate": entry["needs_tiebreak"],
                   "capturedevents_draws": summarize_across_draws(np_mod, entry["captured"]),
                   "recall_draws": summarize_across_draws(np_mod, entry["recall"]),
                   "precision_draws": summarize_across_draws(np_mod, entry["precision"]),
                   "lift_draws": summarize_across_draws(np_mod, lift)}
            if boot is not None:
                row["capturedevents_bootstrap_espacial"] = summarize_across_draws(np_mod, np_mod.array(boot[B]["captured"]))
                row["recall_bootstrap_espacial"] = summarize_across_draws(np_mod, np_mod.array(boot[B]["recall"]))
            if entry["needs_tiebreak"]:
                tb = tiebreak_result[B]
                row["capturedevents_desempate_promediado"] = tb["captured"]
                row["lift_desempate_promediado"] = tb["lift"]
            rows.append(row)

    # "aleatorio": media por draw sobre las repeticiones, mismo tratamiento de resumen que el resto.
    for B in budgets:
        captured_per_draw = random_mean_per_draw[B]
        total_by_draw = outcome_matrix.to_numpy(dtype=float).sum(axis=0)
        recall_per_draw = np_mod.divide(captured_per_draw, total_by_draw, out=np_mod.full_like(captured_per_draw, np_mod.nan), where=total_by_draw > 0)
        rows.append({"unidad": unit_label, "metodo": "aleatorio", "ponderacion": METHOD_WEIGHTING_LABEL["aleatorio"], "B": B,
                      "n_tied_en_cutoff": 0, "necesita_desempate": False,
                      "capturedevents_draws": summarize_across_draws(np_mod, captured_per_draw),
                      "recall_draws": summarize_across_draws(np_mod, recall_per_draw),
                      "precision_draws": summarize_across_draws(np_mod, random_precision_mean_per_draw[B]),
                      "lift_draws": {"media": 1.0, "sd": 0.0, "p2.5": 1.0, "p97.5": 1.0, "min": 1.0, "max": 1.0}})
    return pd.DataFrame(_flatten_rows(rows))


def _flatten_rows(rows: list) -> list:
    flat = []
    for r in rows:
        base = {"unidad": r["unidad"], "metodo": r["metodo"], "ponderacion": r["ponderacion"], "B": r["B"],
                "n_tied_en_cutoff": r["n_tied_en_cutoff"], "necesita_desempate": r["necesita_desempate"]}
        for block_name in ["capturedevents_draws", "recall_draws", "precision_draws", "lift_draws",
                            "capturedevents_bootstrap_espacial", "recall_bootstrap_espacial",
                            "capturedevents_desempate_promediado", "lift_desempate_promediado"]:
            if block_name not in r:
                continue
            for stat, val in r[block_name].items():
                base[f"{block_name}_{stat}"] = val
        flat.append(base)
    return flat


# ---------------------------------------------------------------------------
# Especies vinculadas a registros en localidades que subsecuentemente perdieron soporte forestal
# -- NUNCA "especies afectadas" (ver bloque POR QUE de B2 en el plan: perder soporte forestal no
# demuestra que la especie desaparecio). Localidad marcada si Y-barra_i > 0 (convertida en al
# menos uno de los M draws).
# ---------------------------------------------------------------------------


def build_species_linked_table(pd, df, outcome_matrix):
    outcome_mean = outcome_matrix.mean(axis=1)
    affected_locs = set(outcome_mean.loc[outcome_mean > 0].index)
    sub = df.loc[df["loc_key"].isin(affected_locs)]
    species_counts = sub.groupby("species")["loc_key"].nunique().rename("n_localidades_vinculadas").sort_values(ascending=False)
    return species_counts.reset_index(), len(affected_locs)


# ---------------------------------------------------------------------------
# Diagnostico top-100 por metodo: A_i(t0) medio, localidades con A_i=0, y cuantos draws (de M) esta
# cada localidad en riesgo -- el mecanismo que explica por que las variantes SIN PONDERAR
# seleccionan localidades marginales (ver bloque POR QUE).
# ---------------------------------------------------------------------------


def build_top100_ai_diagnostic(pd, np_mod, r_bar: dict, a_i_t: "pd.Series", n_draws_at_risk: "pd.Series",
                                methods: list, tiebreak_key, top_n: int = 100):
    rows = []
    for method in methods:
        s = r_bar[method].reindex(a_i_t.index).fillna(0.0)
        tb = tiebreak_key.reindex(a_i_t.index)
        order_idx = pd.DataFrame({"score": s, "tb": tb}).sort_values(["score", "tb"], ascending=[False, True]).index
        top = order_idx[:top_n]
        ait_top = a_i_t.reindex(top)
        ndraws_top = n_draws_at_risk.reindex(top)
        cutoff_val = s.reindex(top).iloc[-1]
        n_tied_universe = int((s == cutoff_val).sum())
        rows.append({
            "metodo": method, "ponderacion": METHOD_WEIGHTING_LABEL.get(method, "n/a"),
            "A_i_t0_media_top100": float(ait_top.mean()), "A_i_t0_mediana_top100": float(ait_top.median()),
            "n_A_i_t0_cero_top100": int((ait_top == 0).sum()),
            "n_draws_en_riesgo_media_top100_de_M": float(ndraws_top.mean()),
            "n_draws_en_riesgo_eq1_top100": int((ndraws_top == 1).sum()),
            "n_empatados_en_valor_de_corte_universo": n_tied_universe,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dataset PROSPECTIVO: SOLO risk set + covariables en t=Y_max, SIN Z (Y_max+1..Y_max+5 no existe
# todavia en la trayectoria -- ver bloque POR QUE, nunca llamar find_event_bit con horizonte fuera
# de rango).
# ---------------------------------------------------------------------------


def build_prospective_dataset(rt, d2, r3, t: int, k: int, year_start: int, year_max: int,
                               entries_by_location: dict, loc_code_tables: dict,
                               sampled_codes_by_draw: dict, loss_window: int):
    np_mod, pd = rt["np"], rt["pd"]
    t_idx = t - year_start
    l_k_t_idx = t_idx - k + 1

    entry_idx_by_location = {}
    for loc, entries in entries_by_location.items():
        y_star = r3.most_recent_entry_at_or_before(entries, t)
        if y_star is not None:
            entry_idx_by_location[loc] = y_star - year_start

    agg_rows = r3.compute_location_aggregates_at_t(np_mod, d2, loc_code_tables, entry_idx_by_location, t_idx, k, l_k_t_idx, loss_window)
    agg_table = pd.DataFrame(agg_rows)

    draw_rows = []
    for draw_id, codes_by_loc in sampled_codes_by_draw.items():
        for loc, code in codes_by_loc.items():
            entry_idx = entry_idx_by_location.get(loc)
            if entry_idx is None:
                continue
            if not d2.bit_at(code, entry_idx):
                continue
            # SOLO el gate de riesgo (hacia atras, dentro de rango) -- SIN calcular Z: t+h excede
            # Y_max, ver bloque POR QUE.
            ev_by_t, _ = d2.find_event_bit(code, entry_idx, k, l_k_t_idx)
            forest_t = d2.bit_at(code, t_idx)
            at_risk = forest_t and ev_by_t != 1
            if not at_risk:
                continue
            draw_rows.append({"location_id": loc, "draw_id": draw_id, "entry_idx": entry_idx})
    draw_table = pd.DataFrame(draw_rows)
    if len(draw_table) == 0 or len(agg_table) == 0:
        return pd.DataFrame()
    full = draw_table.merge(agg_table, on=["location_id", "entry_idx"], how="left")
    full["age_at_t"] = t - (year_start + full["entry_idx"])
    return full


# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    np, pd, gpd, lgb = rt["np"], rt["pd"], rt["gpd"], rt["lgb"]

    print("B2 - Capa de decision sobre el backtest de R3 (PLAN_V3_EJECUTABLE.md)")
    print("=================================================================")

    d1 = load_module(D1_PATH, "d1_trajectory_extract")
    d2 = load_module(D2_PATH, "d2_build_survival_dataset")
    m1 = load_module(M1_PATH, "m1_survival_models")
    r3 = load_module(R3_PATH, "r3_landmark_forecast")

    meta = r3.json_load_metadata(args.trajectory_dir)
    year_start, year_max = meta["year_start"], meta["year_max"]
    t_max_confirmable = year_max - args.horizon_years - args.k + 1
    print(f"Ventana: {year_start}-{year_max}. h={args.horizon_years}, k={args.k}. "
          f"t_max confirmable = {t_max_confirmable}. Landmark de decision (t0) = {args.decision_landmark}.")
    if args.decision_landmark > t_max_confirmable:
        raise SystemExit(f"Landmark de decision {args.decision_landmark} > t_max confirmable ({t_max_confirmable}).")
    if max(args.train_landmarks) >= args.decision_landmark:
        raise SystemExit("Los landmarks de entrenamiento del backtest deben ser estrictamente "
                          "anteriores al landmark de decision (fuga temporal si no).")
    for t in args.prospective_train_landmarks:
        if t > t_max_confirmable:
            raise SystemExit(f"Landmark prospectivo de entrenamiento {t} > t_max confirmable "
                              f"({t_max_confirmable}) -- outcome incompleto, no se puede usar.")

    df = d1.load_occurrences(pd, args.occurrences_csv, year_start)
    df = d1.filter_year_upper_bound(df, year_max, args.year_max_tolerance)
    print(f"Registros validos: {len(df):,}")
    entries_by_location = r3.build_entries_by_location(pd, df)
    print(f"Localidades unicas: {len(entries_by_location):,}")

    cell_id_by_location, cell_diag = build_cell_id_lookup(pd, gpd, np, df, args.grid_cell_size_m)
    print(f"Grilla de 5 km: {cell_diag['fuente']}. Localidades con cell_id inconsistente entre "
          f"registros: {cell_diag['localidades_con_cell_id_inconsistente']}. Completadas con "
          f"respaldo: {cell_diag['localidades_sin_cell_id_en_csv_completadas_con_respaldo']}.")

    sampled_codes_by_draw, draws_available = r3.load_sampled_codes(
        pd, args.survival_dir, args.k, args.buffer_m, args.n_draws, args.max_draws
    )
    m_total = len(draws_available)
    print(f"Draws Monte Carlo reusados de D2/R3: {m_total}")

    combined = d1.load_buffer_trajectory(pd, args.trajectory_dir / ".chunks", args.buffer_m)
    if combined is None:
        raise SystemExit(f"Sin checkpoints de trayectoria para buffer {args.buffer_m}m.")
    loc_code_tables = d2.build_location_code_tables(pd, combined)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # === Backtest: mismos landmarks/hiperparametros/refit que 24_landmark_forecast.py ===
    all_backtest_landmarks = args.train_landmarks + [args.decision_landmark]
    landmark_tables, sigma_hat_by_landmark = {}, {}
    for t in all_backtest_landmarks:
        print(f"\n--- Construyendo dataset del landmark {t} (ventana {t}->{t+args.horizon_years}) ---")
        table, sigma_hat_by_draw = r3.build_landmark_dataset(
            rt, d2, m1, t, args.horizon_years, args.k, year_start, year_max,
            entries_by_location, loc_code_tables, sampled_codes_by_draw, r3.LOSS_WINDOW_YEARS,
        )
        landmark_tables[t] = table
        sigma_hat_by_landmark[t] = sigma_hat_by_draw
        if len(table):
            print(f"  Filas (localidad x draw en riesgo): {len(table):,}. Z=1: {int(table['Z'].sum()):,} ({table['Z'].mean():.2%}).")
        else:
            print("  AVISO: dataset vacio para este landmark.")

    print("\n=== Seleccion de hiperparametros (rolling-origin, SOLO landmarks de entrenamiento del backtest) ===")
    best, grid_scores = r3.rolling_origin_select_hyperparams(
        rt, lambda params, tr, va: r3.fit_predict_lgb(lgb, params, tr, va),
        rt["average_precision_score"], landmark_tables, args.train_landmarks, r3.LGBM_GRID,
    )
    print(f"Mejor combinacion: {best['params']} (AUPRC rolling media={best['mean_auprc_rolling']:.4f})")

    train_df = pd.concat([landmark_tables[t] for t in args.train_landmarks if len(landmark_tables[t])], ignore_index=True)
    backtest_model = lgb.LGBMClassifier(**best["params"], random_state=42, verbose=-1)
    backtest_model.fit(train_df[r3.COVARIATE_COLS], train_df["Z"])
    print(f"Refit del landmark boosting del backtest sobre {len(train_df):,} filas.")

    # === Logistica: MISMO protocolo que el boosting -- mismos landmark_tables (mismos landmarks,
    # mismos draws posicionales), mismo rolling_origin_select_hyperparams (temporal, SOLO landmarks
    # de entrenamiento), mismo refit final -- ver bloque POR QUE de 24_landmark_forecast.py. ===
    print("\n=== Seleccion de hiperparametros (rolling-origin, logistica, SOLO landmarks de entrenamiento del backtest) ===")
    best_logistic, grid_scores_logistic = r3.rolling_origin_select_hyperparams(
        rt, lambda params, tr, va: r3.fit_predict_logistic(rt, params, tr, va),
        rt["average_precision_score"], landmark_tables, args.train_landmarks, r3.LOGISTIC_GRID,
    )
    print(f"Mejor combinacion (logistica): {best_logistic['params']} "
          f"(AUPRC rolling media={best_logistic['mean_auprc_rolling']:.4f})")
    backtest_model_logistic = r3.build_logistic_pipeline(rt, best_logistic["params"])
    backtest_model_logistic.fit(train_df[r3.COVARIATE_COLS], train_df["Z"])
    print(f"Refit de la logistica del backtest sobre las mismas {len(train_df):,} filas.")

    # === Score de decision r_bar_i en t0 ===
    decision_table = landmark_tables[args.decision_landmark].copy()
    if len(decision_table) == 0:
        raise SystemExit("Dataset del landmark de decision vacio -- no se puede construir la capa de decision.")
    row_scores = r3.score_methods(rt, np, m1, decision_table, sigma_hat_by_landmark[args.decision_landmark], args.sigma_fixed, args.horizon_years, args.random_seed)
    decision_table["score_landmark_boosting"] = backtest_model.predict_proba(decision_table[r3.COVARIATE_COLS])[:, 1]
    decision_table["score_landmark_logistic"] = backtest_model_logistic.predict_proba(decision_table[r3.COVARIATE_COLS])[:, 1]
    decision_table["score_rayleigh_30"] = row_scores["rayleigh_30"]
    decision_table["score_rayleigh_estimado"] = row_scores["rayleigh_estimado"]

    r_bar = build_r_bar_scores(pd, decision_table, m_total)
    outcome_matrix = build_outcome_matrix(pd, decision_table, draws_available)
    a_i_t, n_draws_at_risk = build_ai_and_ndraws_lookup(pd, decision_table)
    print(f"\nUniverso de decision (localidades en riesgo en >=1 de {m_total} draws en t0={args.decision_landmark}): {len(outcome_matrix):,}")

    tiebreak_key_loc = generate_tiebreak_key(np, pd, outcome_matrix.index, args.tiebreak_seed)

    # === Diagnostico: A_i(t0) del top-100, para las variantes ponderadas y sin ponderar (ver bloque POR QUE) ===
    ai_diag_table = build_top100_ai_diagnostic(pd, np, r_bar, a_i_t, n_draws_at_risk, DECISION_METHODS, tiebreak_key_loc, top_n=100)
    print("\n=== Diagnostico top-100 (localidades): A_i(t0) medio, localidades con A_i=0, draws en riesgo, empates ===")
    print(ai_diag_table.to_string(index=False))
    ai_diag_table.to_csv(args.output_dir / f"diagnostic_top100_A_i_check_{args.decision_landmark}.csv", index=False)

    # === PRINCIPAL: presupuesto B de localidades ===
    print("\n=== PRINCIPAL: presupuesto B de LOCALIDADES (bootstrap de bloque espacial incluido) ===")
    rng_random_loc = np.random.default_rng(args.random_seed)
    rng_boot_loc = np.random.default_rng(args.random_seed + 1)
    rng_tiebreak_loc = np.random.default_rng(args.tiebreak_seed + 1)
    principal_results = run_decision_evaluation(
        rt, m1, "localidad", r_bar, outcome_matrix, args.budgets, args.n_random_repeats, rng_random_loc,
        True, args.n_bootstrap, rng_boot_loc, tiebreak_key_loc, args.n_tiebreak_repeats, rng_tiebreak_loc,
        location_ids_for_blocks=outcome_matrix.index.to_numpy(),
    )
    print(principal_results.to_string(index=False))

    # === SENSIBILIDAD: presupuesto B de celdas de 5 km (sin bootstrap espacial, ver bloque POR QUE) ===
    print("\n=== SENSIBILIDAD: presupuesto B de CELDAS de 5 km ===")
    cell_r_bar, cell_outcome_matrix, n_missing_cell = aggregate_to_cells(pd, r_bar, outcome_matrix, cell_id_by_location)
    if n_missing_cell:
        print(f"AVISO: {n_missing_cell} localidades sin cell_id fueron excluidas de la sensibilidad por celdas.")
    tiebreak_key_cell = generate_tiebreak_key(np, pd, cell_outcome_matrix.index, args.tiebreak_seed + 2)
    rng_random_cell = np.random.default_rng(args.random_seed + 2)
    rng_tiebreak_cell = np.random.default_rng(args.tiebreak_seed + 3)
    sensitivity_results = run_decision_evaluation(
        rt, m1, "celda_5km", cell_r_bar, cell_outcome_matrix, args.budgets, args.n_random_repeats,
        rng_random_cell, False, 0, None, tiebreak_key_cell, args.n_tiebreak_repeats, rng_tiebreak_cell,
        location_ids_for_blocks=None,
    )
    print(sensitivity_results.to_string(index=False))

    # === VISUALIZACION: agregacion a TODAS las celdas de 5 km (no solo top-B) del score del landmark boosting ===
    viz_cells = r_bar["landmark_boosting"].reindex(outcome_matrix.index).fillna(0.0).groupby(cell_id_by_location.reindex(outcome_matrix.index)).agg(["sum", "mean", "count"])
    viz_cells.columns = ["r_sum", "r_mean", "n_localidades"]
    viz_cells = viz_cells.reset_index().rename(columns={"index": "cell_id"})
    viz_cells.to_csv(args.output_dir / f"viz_cells_5km_landmark_{args.decision_landmark}.csv", index=False)

    # === Especies vinculadas a registros en localidades subsecuentemente convertidas ===
    species_table, n_affected_locs = build_species_linked_table(pd, df, outcome_matrix)
    species_table.to_csv(args.output_dir / f"species_linked_to_affected_records_{args.decision_landmark}.csv", index=False)
    print(f"\nEspecies vinculadas a registros en localidades que subsecuentemente perdieron soporte "
          f"forestal (>=1 de {m_total} draws): {len(species_table):,} especies, {n_affected_locs:,} localidades.")

    # === guardar resultados del backtest/decision ===
    principal_results.to_csv(args.output_dir / f"decision_localidades_{args.decision_landmark}.csv", index=False)
    sensitivity_results.to_csv(args.output_dir / f"decision_celdas5km_{args.decision_landmark}.csv", index=False)
    for method in DECISION_METHODS:
        r_bar[method].rename("r_bar").reset_index().rename(columns={"index": "location_id"}).to_csv(
            args.output_dir / f"r_bar_{method}_{args.decision_landmark}.csv", index=False
        )

    diagnostics = build_diagnostics_text(
        year_start, year_max, args, t_max_confirmable, best, landmark_tables, all_backtest_landmarks,
        outcome_matrix, cell_diag, principal_results, sensitivity_results, species_table, n_affected_locs, m_total,
        ai_diag_table,
    )

    # === PROSPECTIVO: extrapolacion NO observada ===
    if not args.skip_prospective:
        print(f"\n=== MAPA PROSPECTIVO: reentrenamiento con {args.prospective_train_landmarks}, "
              f"prediccion {year_max}->{year_max + args.horizon_years} (EXTRAPOLACION NO OBSERVADA) ===")
        for t in args.prospective_train_landmarks:
            if t not in landmark_tables:
                print(f"--- Construyendo dataset del landmark prospectivo {t} (ventana {t}->{t+args.horizon_years}) ---")
                table, _ = r3.build_landmark_dataset(
                    rt, d2, m1, t, args.horizon_years, args.k, year_start, year_max,
                    entries_by_location, loc_code_tables, sampled_codes_by_draw, r3.LOSS_WINDOW_YEARS,
                )
                landmark_tables[t] = table
        prospective_train_df = pd.concat(
            [landmark_tables[t] for t in args.prospective_train_landmarks if len(landmark_tables[t])], ignore_index=True
        )
        # MISMOS hiperparametros del backtest, NO se vuelven a buscar -- ver bloque POR QUE.
        prospective_model = lgb.LGBMClassifier(**best["params"], random_state=42, verbose=-1)
        prospective_model.fit(prospective_train_df[r3.COVARIATE_COLS], prospective_train_df["Z"])
        print(f"Refit prospectivo sobre {len(prospective_train_df):,} filas ({len(args.prospective_train_landmarks)} landmarks).")

        prospective_table = build_prospective_dataset(
            rt, d2, r3, year_max, args.k, year_start, year_max,
            entries_by_location, loc_code_tables, sampled_codes_by_draw, r3.LOSS_WINDOW_YEARS,
        )
        if len(prospective_table) == 0:
            print("AVISO: dataset prospectivo vacio -- no se genera el mapa prospectivo.")
        else:
            prospective_table["score_landmark_boosting"] = prospective_model.predict_proba(prospective_table[r3.COVARIATE_COLS])[:, 1]
            r_bar_prospective = prospective_table.groupby("location_id")["score_landmark_boosting"].sum() / m_total
            print(f"Localidades en el universo prospectivo (en riesgo en >=1 draw en {year_max}): {len(r_bar_prospective):,}")

            prospective_out = r_bar_prospective.rename("r_bar_prospectivo_NO_OBSERVADO").reset_index().rename(columns={"index": "location_id"})
            prospective_out["extrapolacion"] = f"NO OBSERVADA: prediccion {year_max}->{year_max + args.horizon_years}, sin datos de verificacion disponibles"
            prospective_out.to_csv(args.output_dir / f"prospective_localidades_{year_max}_NO_OBSERVADO.csv", index=False)

            prospective_cells = r_bar_prospective.reindex(prospective_table["location_id"].unique()).groupby(
                cell_id_by_location.reindex(prospective_table["location_id"].unique())
            ).agg(["sum", "mean", "count"])
            prospective_cells.columns = ["r_sum", "r_mean", "n_localidades"]
            prospective_cells = prospective_cells.reset_index().rename(columns={"index": "cell_id"})
            prospective_cells["extrapolacion"] = f"NO OBSERVADA: prediccion {year_max}->{year_max + args.horizon_years}"
            prospective_cells.to_csv(args.output_dir / f"prospective_cells5km_{year_max}_NO_OBSERVADO.csv", index=False)

            top_b_prospective = {}
            order = r_bar_prospective.sort_values(ascending=False)
            for B in args.budgets:
                top_b_prospective[B] = order.head(B).index.tolist()
            import json
            (args.output_dir / f"prospective_topB_candidatos_{year_max}_NO_OBSERVADO.json").write_text(
                json.dumps({str(B): locs for B, locs in top_b_prospective.items()}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Guardado: predicciones prospectivas y candidatos top-B (declarados EXTRAPOLACION NO OBSERVADA).")
            diagnostics += [
                "",
                "*** MAPA PROSPECTIVO: EXTRAPOLACION PROSPECTIVA NO OBSERVADA (post-2024 forecast) ***",
                f"Reentrenado con landmarks {args.prospective_train_landmarks} (mismos hiperparametros del "
                f"backtest, NO re-seleccionados). Prediccion en ventana {year_max}->{year_max+args.horizon_years}, "
                "sin datos de verificacion disponibles todavia -- el backtest valida el PROCEDIMIENTO, este mapa "
                "es el producto operativo pero NO esta validado directamente.",
                f"Localidades en universo prospectivo: {len(r_bar_prospective):,}.",
                f"r_bar prospectivo: media={float(r_bar_prospective.mean()):.4f}, "
                f"p95={float(np.percentile(r_bar_prospective, 95)):.4f}, max={float(r_bar_prospective.max()):.4f}.",
            ]
    else:
        print("\n--skip-prospective: mapa prospectivo omitido (solo pruebas).")

    (args.output_dir / f"diagnostics_decision_{args.decision_landmark}.txt").write_text(
        "\n".join(str(l) for l in diagnostics) + "\n", encoding="utf-8"
    )
    print(f"\nGuardado: {args.output_dir / f'diagnostics_decision_{args.decision_landmark}.txt'}")


def build_diagnostics_text(year_start, year_max, args, t_max_confirmable, best, landmark_tables,
                            all_backtest_landmarks, outcome_matrix, cell_diag, principal_results,
                            sensitivity_results, species_table, n_affected_locs, m_total, ai_diag_table):
    lines = [
        "B2 diagnostics - capa de decision sobre el backtest de R3 (PLAN_V3_EJECUTABLE.md)",
        "===========================================================================",
        f"Ventana: {year_start}-{year_max}. h={args.horizon_years}, k={args.k}. t_max confirmable={t_max_confirmable}.",
        f"Landmark de decision (t0): {args.decision_landmark}. Landmarks de entrenamiento del backtest: {args.train_landmarks}.",
        f"Draws Monte Carlo (M, denominador fijo de r_bar_i): {m_total}.",
        f"Hiperparametros del landmark boosting (rolling-origin, reusados sin cambios en el "
        f"reentrenamiento prospectivo): {best['params']}.",
        f"Grilla de 5 km: {cell_diag['fuente']}.",
        f"Universo de decision en t0: {len(outcome_matrix):,} localidades (en riesgo en >=1 de {m_total} draws).",
        "",
        "SCORE r_bar_i = (1/M) * sum_m I_im(t0) * valor_im PARA landmark_boosting, rayleigh_30, "
        "rayleigh_estimado, mas_antiguo_ponderado y perdida_reciente_ponderado (ver columna "
        "'ponderacion'=ponderado_I_im). mas_antiguo y perdida_reciente (ponderacion=sin_ponderar) "
        "son la covariable deterministica directa, SIN el indicador de riesgo por draw -- las dos "
        "variantes de cada baseline se reportan juntas y rotuladas, ver bloque POR QUE.",
        "",
        "*** DIAGNOSTICO top-100 (localidades): A_i(t0) medio revela el mecanismo -- perdida_reciente "
        "sin ponderar selecciona sitios casi sin soporte restante (A_i(t0) bajo, pocos draws en "
        "riesgo), mientras que su variante ponderada corrige eso por construccion (ver bloque POR "
        "QUE, columna n_empatados_en_valor_de_corte_universo documenta empates de baja cardinalidad "
        "en variables discretas como la edad) ***",
        ai_diag_table.to_string(index=False),
        "",
        "*** PRESUPUESTO PRINCIPAL: localidades (captured/recall/precision/lift resumidos entre los "
        f"{m_total} draws Monte Carlo; bootstrap de bloque espacial de 100 km sobre outcome "
        "promediado entre draws; desempate SEEDED [no arbitrario por orden de fila] igual en la "
        "tabla principal y en el bootstrap; n_tied_en_cutoff/necesita_desempate documentan cuantas "
        "localidades comparten el valor de corte; cuando el empate excede los cupos disponibles, "
        "las columnas *_desempate_promediado reportan media+IC sobre "
        f"{args.n_tiebreak_repeats} desempates aleatorios independientes en vez de un top-B "
        "arbitrario) ***",
        principal_results.to_string(index=False),
        "",
        "*** SENSIBILIDAD: celdas de 5 km (mismo B, unidad = celda; sin bootstrap espacial, ver "
        "bloque POR QUE; mismo tratamiento de desempate seeded/promediado que el presupuesto "
        "principal) ***",
        sensitivity_results.to_string(index=False),
        "",
        f"Especies vinculadas a registros en localidades que subsecuentemente perdieron soporte "
        f"forestal (>=1 de {m_total} draws) -- NUNCA \"especies afectadas\": {len(species_table):,} "
        f"especies, {n_affected_locs:,} localidades.",
    ]
    return lines


if __name__ == "__main__":
    main()
