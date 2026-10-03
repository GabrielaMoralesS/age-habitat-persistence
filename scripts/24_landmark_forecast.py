# CCAI OBSOLESCENCIA - R3 (PLAN_V3_EJECUTABLE.md): landmark forecasting a cinco años -- "que
# localidades perderan soporte forestal en los proximos 5 años", evaluado con un backtest interno
# de rolling-origin (landmarks 1993/1998/2003/2008) y un test UNICO congelado (2014->2019). Lee
# results/trajectory/.chunks/, results/trajectory/run_metadata.json, results/survival/
# dataset_k{K}_buffer{B}m.csv (para reusar los draws Monte Carlo ya muestreados) y
# data/processed/occurrences_with_cell_id.csv. NO CONSULTA EARTH ENGINE.
"""
POR QUE
-------
R2 (22_survival_models.py) responde si el decaimiento gaussiano/Rayleigh esta CALIBRADO contra la
persistencia observada -- pero solo dentro del soporte temporal ya visto, sin producir una
prediccion individual util para decidir que localidad verificar primero. R3 es la pregunta
distinta y complementaria: "¿que localidades van a perder soporte forestal en los proximos 5
años?", con una prediccion POR LOCALIDAD (no una curva poblacional). Por eso R3 usa un modelo
supervisado (landmark gradient boosting) entrenado con el label binario Z_{i,t} = conversion
persistente en (t, t+5], no una funcion de supervivencia.

LA EDAD SI ES COVARIABLE ACA, A DIFERENCIA DE D2/R2: en el modelo de supervivencia la edad es el
EJE TEMPORAL del outcome (duracion hasta el evento) -- incluirla como covariable ahi duplicaria el
tiempo de supervivencia. En el landmark model el outcome ya NO es un tiempo, es un binario fijo a
horizonte fijo (Z_{i,t}), asi que la edad del registro en el corte t es una covariable legitima
mas, no una fuga. Esto no contradice la prohibicion de D2 -- son dos diseños distintos a proposito.

FECHAS DE LANDMARK -- LA CUENTA EXACTA, NO APROXIMADA: un landmark t esta completamente
etiquetado solo si t+h+k-1 <= Y_max (se necesitan los k años de confirmacion del evento mas
tardio posible, que puede empezar hasta t+h). Con h=5, k=2, Y_max=2024 (de run_metadata.json, NO
hardcodeado): t_max = 2024-5-2+1 = 2018. El test congelado de este script es t=2014 (h=5,
ventana 2014->2019) -- DENTRO del rango confirmable (2014 <= 2018), consistente con el ejemplo
del plan ("simulando una decision en 2014, t_max=2008 para el backtest de ENTRENAMIENTO" -- ver
mas abajo por que 2014 es el TEST y no entra en el t_max de 2008). Los landmarks de ENTRENAMIENTO
{1993,1998,2003,2008} son estrictamente anteriores a 2014 -- esto es CLAVE: si se usara 2014 (o
cualquier año >=2009) para ajustar hiperparametros, el resultado en el test congelado de 2014
dejaria de ser una evaluacion honesta (el modelo ya habria "visto" señal de esa misma epoca
temporal durante el ajuste). El test se toca UNA sola vez, al final, con los hiperparametros ya
fijados.

SIGMA-HAT DE RAYLEIGH REESTIMADO POR LANDMARK, NUNCA REUTILIZANDO EL SIGMA-HAT GLOBAL DE R2:
el sigma_hat que 22_survival_models.py reporta (~54) se estimo con TODA la ventana 1985-Y_max --
usarlo tal cual como baseline de prediccion en un landmark t significaria que el baseline "sabe"
informacion de años posteriores a t, una fuga sutil pero real (el equivalente aca del error que el
plan ya marco para 01d/D1: no mirar t+1 para confirmar un evento en t). Por eso RAYLEIGH_ESTIMADO
en este script reestima sigma_hat_t EN CADA landmark, usando SOLO trayectorias observables hasta
L_k,t = t-k+1 -- ni una consulta adicional a GEE, mismo MLE en forma cerrada ya verificado en
22_survival_models.py, recalculado con una ventana de datos distinta en cada corte. Se hace POR
DRAW (no pooled entre draws), igual que R2, para mantener "mismos draws entre modelos" y la misma
separacion de incertidumbre posicional vs. espacial en toda la tesis.

MISMOS DRAWS QUE D2/R2, REUTILIZADOS, NO REMUESTREADOS: cada localidad ya tiene, para cada uno de
los 20 draws Monte Carlo, UNA posicion muestreada (columna sampled_traj_code de
results/survival/dataset_k{K}_buffer{B}m.csv, ver 21_build_survival_dataset.py) -- la MISMA
posicion fisica se usa en TODOS los landmarks de un mismo draw (la localidad no se mueve entre
cortes temporales, solo cambia el año que se esta evaluando). Releer esa columna en vez de
remuestrear evita duplicar aleatoriedad y garantiza "mismos draws entre modelos" tal como M2 exige.

RIESGO EN t: EXACTAMENTE LA CONDICION DE B1, POR DRAW: una fila (localidad, draw) entra al
dataset de un landmark t solo si esa posicion muestreada especifica seguia con soporte forestal
en t Y sin evento persistente confirmado hasta L_k,t -- "Predictions are generated only for
spatially held-out Monte Carlo positional-sensitivity draws that remain forest-supported and
event-free at t, using only information confirmable by L_k,t" (B1, literal). A_i(t) SI se reporta
como covariable AGREGADA (ponderada por n_px sobre TODOS los codigos de la localidad, igual que
H_i en D2) -- describe el sitio completo, no varia por draw; el LABEL Z_{i,t} y el gate de riesgo
si varian por draw, porque dependen de que posicion especifica se este evaluando. Misma
distincion agregado-vs-per-draw que D2 ya establecio para H_i vs. evento/censura, aplicada aca.

VALIDACION ROLLING-ORIGIN, NO K-FOLD TEMPORAL: para elegir hiperparametros del landmark boosting
se usan SOLO los cuatro landmarks de entrenamiento, en tres particiones que respetan el orden
temporal (entrenar con landmarks mas antiguos, validar con uno mas nuevo, nunca al reves):
{1993}->valida 1998, {1993,1998}->valida 2003, {1993,1998,2003}->valida 2008. Un K-fold temporal
ingenuo (mezclar los cuatro landmarks al azar en folds) dejaria que el modelo "vea" el futuro
relativo de un fold de validacion durante el entrenamiento -- exactamente el tipo de fuga temporal
que el plan corta en D1 (no consultar t+1) y en B1 (no confirmar con años posteriores al corte).

TODAS LAS FILAS DE UNA LOCALIDAD EN EL MISMO BLOQUE ESPACIAL, SIEMPRE: se reutiliza
assign_spatial_blocks de 22_survival_models.py (mismo CRS EPSG:5880, mismo tamaño de bloque de
100 km) para el bootstrap de las metricas del test congelado -- al remuestrear bloques con
reemplazo, TODAS las filas de una localidad (sus hasta 20 draws) se mueven juntas, nunca se
separan. Esto aplica solo al bootstrap final: la particion rolling-origin ya separa por LANDMARK
(tiempo), no hace falta ademas partir por bloque espacial dentro de un landmark individual para
la eleccion de hiperparametros.

DIFERENCIAS ENTRE METODOS: BOOTSTRAP PAREADO, NO DOS IC INDEPENDIENTES RESTADOS: comparar si
landmark_boosting supera a perdida_reciente (o a rayleigh_30) restando sus IC95% individuales es
conservador e incorrecto -- dos intervalos que se solapan no implican que la diferencia cruce cero,
porque ignoran la correlacion entre los dos metodos evaluados sobre la MISMA remuestra. Por eso el
bootstrap de diferencias reusa, en cada una de las 1000 iteraciones, el mismo remuestreo de bloques
espaciales (misma seleccion de localidades) para calcular AUROC/AUPRC de los tres metodos
involucrados y resta dentro de esa iteracion -- la diferencia hereda la correlacion real entre
metodos evaluados sobre los mismos datos, y el IC95% resultante es mas angosto y mas correcto que
comparar dos intervalos marginales a ojo.

PREVALENCIA JUNTO AL AUPRC: el AUPRC de un clasificador sin señal es aprox. igual a la prevalencia
de la clase positiva, no a 0.5 (a diferencia del AUROC). Con Z=1 en ~1% de las filas del test
congelado, un AUPRC de 0.01 es el piso trivial, no una señal debil -- reportar la prevalencia (con
su propio IC95% bootstrap, porque el remuestreo por bloque espacial tambien mueve la tasa base) al
lado de cada AUPRC evita malinterpretar valores absolutos bajos como ausencia de señal.

LOS BASELINES DE RANKING (aleatorio, mas antiguo, perdida reciente) SOLO SE EVALUAN CON METRICAS
DE RANKING (AUROC, AUPRC) -- M2 lo dice explicito: "no meterlos en Brier ni calibracion salvo que
antes se conviertan a probabilidades usando EXCLUSIVAMENTE el entrenamiento". Este script no hace
esa conversion (fuera de alcance de lo pedido), asi que Brier-5 años, calibracion por deciles e
ICI se calculan SOLO para los cuatro esquemas que ya son probabilidades: Rayleigh-30, Rayleigh
estimado, landmark boosting y landmark logistica.

BASELINE LOGISTICO, MISMO PROTOCOLO QUE EL BOOSTING, NO UNO NUEVO: landmark_logistic usa
EXACTAMENTE las mismas COVARIATE_COLS, los mismos landmark_tables (mismos landmarks, mismos draws
posicionales -- ya construidos una sola vez y compartidos por ambos modelos, no se recalculan) y
pasa por el MISMO rolling_origin_select_hyperparams que el boosting (splits temporales
{1993}->1998, {1993,1998}->2003, {1993,1998,2003}->2008, SOLO landmarks de entrenamiento, el test
congelado nunca participa) -- se generalizo esa funcion para recibir un fit_predict_fn y una grilla
en vez de asumir LightGBM, sin cambiar la logica de particion en si. La grilla logistica barre
fuerza de regularizacion (C de LogisticRegression, penalizacion L2 por defecto), NO class_weight
"balanced": reponderar por clase distorsionaria las probabilidades predichas hacia la clase
minoritaria y invalidaria la lectura de Brier/ICI, que dependen de que la probabilidad predicha se
acerque a la prevalencia real (~1%) -- ver el bloque POR QUE de mas arriba sobre por que el AUPRC
se lee junto a la prevalencia. Imputacion (mediana) + escalado (Z-score) van DENTRO de un pipeline
de sklearn ajustado SOLO con el fold de entrenamiento de cada particion (Pipeline con
SimpleImputer+StandardScaler+LogisticRegression) -- necesarios porque, a diferencia de LightGBM
(que maneja NaN y distintas escalas de forma nativa), una regresion logistica regularizada por L2
es sensible a la escala relativa de las covariables (age_at_t llega a ~40, A_i_t/loss_5y estan en
[0,1]) y no acepta NaN directamente.

"MISMOS FOLDS ESPACIALES QUE EL BOOSTING": no hay una particion espacial separada para elegir
hiperparametros (ni para el boosting ni para el logistico -- la seleccion es temporal por diseño,
ver bloque POR QUE de rolling-origin mas arriba). Los folds espaciales que SI comparten ambos
modelos son los del bootstrap de bloque de 100 km sobre el TEST CONGELADO: landmark_logistic entra
al mismo diccionario `scores` que los otros seis metodos, asi que el bucle de bootstrap existente
(una sola remuestra de bloques por iteracion, reusada para TODOS los metodos) lo cubre
automaticamente sin duplicar codigo -- la comparacion landmark_boosting-landmark_logistic en
DIFF_PAIRS hereda la misma correlacion pareada que las demas comparaciones.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
D1_PATH = SCRIPT_DIR / "20_trajectory_extract.py"
D2_PATH = SCRIPT_DIR / "21_build_survival_dataset.py"
M1_PATH = SCRIPT_DIR / "22_survival_models.py"

K_DEFAULT = 2
BUFFER_M_DEFAULT = 250
HORIZON_YEARS_DEFAULT = 5
TRAIN_LANDMARKS_DEFAULT = [1993, 1998, 2003, 2008]
TEST_LANDMARK_DEFAULT = 2014
YEAR_MAX_TOLERANCE_DEFAULT = 3
N_DRAWS_DEFAULT = 20
N_BOOTSTRAP_DEFAULT = 1000
RANDOM_SEED_DEFAULT = 42
SIGMA_FIXED_DEFAULT = 30.0
LOSS_WINDOW_YEARS = 5
LGBM_GRID = [
    {"num_leaves": 7, "learning_rate": 0.05, "n_estimators": 150, "min_child_samples": 50},
    {"num_leaves": 7, "learning_rate": 0.10, "n_estimators": 100, "min_child_samples": 50},
    {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 150, "min_child_samples": 50},
    {"num_leaves": 15, "learning_rate": 0.10, "n_estimators": 100, "min_child_samples": 50},
]
LOGISTIC_GRID = [
    {"C": 0.01},
    {"C": 0.1},
    {"C": 1.0},
    {"C": 10.0},
]
COVARIATE_COLS = ["age_at_t", "A_i_t", "loss_5y", "n_transitions_before_t"]
DIFF_PAIRS = [
    ("landmark_boosting", "perdida_reciente"),
    ("landmark_boosting", "rayleigh_30"),
    ("perdida_reciente", "rayleigh_30"),
    ("landmark_boosting", "landmark_logistic"),
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    obs_root = root
    parser = argparse.ArgumentParser(
        description="R3 - landmark forecasting a 5 años: rolling-origin en landmarks de "
        "entrenamiento, test unico congelado 2014->2019, siete metodos."
    )
    parser.add_argument("--occurrences-csv", type=Path, default=root / "data" / "processed" / "occurrences_with_cell_id.csv")
    parser.add_argument("--trajectory-dir", type=Path, default=obs_root / "results" / "trajectory")
    parser.add_argument("--survival-dir", type=Path, default=obs_root / "results" / "survival")
    parser.add_argument("--output-dir", type=Path, default=obs_root / "results" / "landmark")
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--buffer-m", type=int, default=BUFFER_M_DEFAULT)
    parser.add_argument("--horizon-years", type=int, default=HORIZON_YEARS_DEFAULT)
    parser.add_argument("--train-landmarks", type=int, nargs="+", default=TRAIN_LANDMARKS_DEFAULT)
    parser.add_argument("--test-landmark", type=int, default=TEST_LANDMARK_DEFAULT)
    parser.add_argument("--year-max-tolerance", type=int, default=YEAR_MAX_TOLERANCE_DEFAULT)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    parser.add_argument("--max-draws", type=int, default=None, help="SOLO pruebas.")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--sigma-fixed", type=float, default=SIGMA_FIXED_DEFAULT)
    return parser.parse_args()


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
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
        "np": np, "pd": pd, "gpd": gpd, "lgb": lgb,
        "roc_auc_score": roc_auc_score, "average_precision_score": average_precision_score,
        "brier_score_loss": brier_score_loss,
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
# Paso 1: entradas por localidad (year*_i(t)) y draws ya muestreados (reusados de D2).
# ---------------------------------------------------------------------------


def build_entries_by_location(pd, occurrences) -> dict:
    unique = occurrences[["loc_key", "year"]].drop_duplicates()
    out = {}
    for loc, group in unique.groupby("loc_key"):
        out[loc] = sorted(group["year"].tolist())
    return out


def most_recent_entry_at_or_before(entries: list, t: int):
    candidates = [y for y in entries if y <= t]
    return max(candidates) if candidates else None


def load_sampled_codes(pd, survival_dir: Path, k: int, buffer_m: int, n_draws: int, max_draws):
    """(location_id, draw_id) -> codigo muestreado -- el MISMO draw que D2/R2 ya usaron, leido de
    su salida en vez de remuestrear (ver bloque POR QUE)."""
    path = survival_dir / f"dataset_k{k}_buffer{buffer_m}m.csv"
    if not path.exists():
        raise SystemExit(f"No se encontro {path} -- correr 21_build_survival_dataset.py primero.")
    df = pd.read_csv(path, usecols=["location_id", "draw_id", "sampled_traj_code"])
    df = df.drop_duplicates(subset=["location_id", "draw_id"])
    draws_available = sorted(df["draw_id"].unique())
    if max_draws is not None:
        draws_available = draws_available[:max_draws]
        df = df.loc[df["draw_id"].isin(draws_available)]
    lookup = {}
    for row in df.itertuples():
        lookup.setdefault(row.draw_id, {})[row.location_id] = int(row.sampled_traj_code)
    return lookup, draws_available


# ---------------------------------------------------------------------------
# Paso 2: covariables agregadas A_i(t)/perdida-5y/transiciones -- deterministas, por localidad.
# ---------------------------------------------------------------------------


def compute_location_aggregates_at_t(np_mod, d2, loc_code_tables: dict, entry_idx_by_location: dict, t_idx: int, k: int, l_k_t_idx: int, loss_window: int):
    rows = []
    for loc, entry_idx in entry_idx_by_location.items():
        table = loc_code_tables.get(loc)
        if table is None:
            continue
        codes, weights = table
        total_w = 0.0
        a_num = 0.0
        loss_num, loss_w = 0.0, 0.0
        trans_num, trans_w = 0.0, 0.0
        for c, w in zip(codes, weights):
            c = int(c)
            total_w += w
            forest_entry = d2.bit_at(c, entry_idx)
            forest_t = d2.bit_at(c, t_idx)
            ev, _ = d2.find_event_bit(c, entry_idx, k, l_k_t_idx)
            # ev es None cuando entry_idx > l_k_t_idx -- SOLO posible si entry_idx esta a menos
            # de (k-1) años de t (para k=2: entry_year==t exactamente). En ese caso no hubo TIEMPO
            # para que ocurra un evento entre la entrada y t, asi que "sin evento confirmado" es
            # la respuesta correcta por construccion, no un dato faltante -- de ahi que `ev != 1`
            # (None != 1 es True) sea deliberado, no un descuido de comparar contra None.
            if forest_entry and forest_t and ev != 1:
                a_num += w
            lv = d2.frac_nonforest_before(c, t_idx, loss_window)
            if not np_mod.isnan(lv):
                loss_num += lv * w
                loss_w += w
            tv = d2.n_transitions_before(c, t_idx)
            if not np_mod.isnan(tv):
                trans_num += tv * w
                trans_w += w
        rows.append({
            "location_id": loc, "entry_idx": entry_idx,
            "A_i_t": (a_num / total_w) if total_w > 0 else float("nan"),
            "loss_5y": (loss_num / loss_w) if loss_w > 0 else float("nan"),
            "n_transitions_before_t": (trans_num / trans_w) if trans_w > 0 else float("nan"),
        })
    return rows


# ---------------------------------------------------------------------------
# Paso 3: filas por draw -- gate de riesgo (B1), label Z, e insumos para sigma_hat_t por draw.
# ---------------------------------------------------------------------------


def build_draw_rows_and_sigma_inputs(np_mod, d2, sampled_codes_by_draw: dict, entry_idx_by_location: dict, t_idx: int, h_end_idx: int, k: int, l_k_t_idx: int):
    rows = []
    sigma_inputs = {}  # draw_id -> (durations list, events list)
    for draw_id, codes_by_loc in sampled_codes_by_draw.items():
        durations, events = [], []
        for loc, code in codes_by_loc.items():
            entry_idx = entry_idx_by_location.get(loc)
            if entry_idx is None:
                continue
            forest_entry = d2.bit_at(code, entry_idx)
            if not forest_entry:
                continue
            ev_by_t, dur_by_t = d2.find_event_bit(code, entry_idx, k, l_k_t_idx)
            # ev_by_t es None cuando entry_idx > l_k_t_idx -- localidades cuya UNICA entrada
            # disponible es el propio año t (o muy cercana), sin seguimiento observable todavia
            # para estimar sigma_hat_t en este landmark. Se excluyen del insumo del MLE (NO se
            # agregan como NaN) -- incluirlas corrompia la suma de duration^2 y devolvia
            # sigma_hat=nan para TODOS los draws, bug real detectado en el smoke test.
            if ev_by_t is not None:
                durations.append(dur_by_t)
                events.append(ev_by_t)
            forest_t = d2.bit_at(code, t_idx)
            at_risk = forest_t and ev_by_t != 1  # ev_by_t=None -> True, mismo razonamiento de arriba
            if not at_risk:
                continue
            z, _ = d2.find_event_bit(code, t_idx, k, h_end_idx)
            rows.append({"location_id": loc, "draw_id": draw_id, "entry_idx": entry_idx, "Z": z})
        sigma_inputs[draw_id] = (np_mod.array(durations, dtype=float), np_mod.array(events, dtype=bool))
    return rows, sigma_inputs


# ---------------------------------------------------------------------------
# Construccion completa del dataset de un landmark.
# ---------------------------------------------------------------------------


def build_landmark_dataset(rt, d2, m1, t: int, h: int, k: int, year_start: int, year_max: int,
                            entries_by_location: dict, loc_code_tables: dict,
                            sampled_codes_by_draw: dict, loss_window: int):
    np_mod, pd = rt["np"], rt["pd"]
    t_idx = t - year_start
    l_k_t_idx = t_idx - k + 1
    h_end_idx = (t + h) - year_start

    entry_idx_by_location = {}
    for loc, entries in entries_by_location.items():
        y_star = most_recent_entry_at_or_before(entries, t)
        if y_star is not None:
            entry_idx_by_location[loc] = y_star - year_start

    agg_rows = compute_location_aggregates_at_t(np_mod, d2, loc_code_tables, entry_idx_by_location, t_idx, k, l_k_t_idx, loss_window)
    agg_table = pd.DataFrame(agg_rows)

    draw_rows, sigma_inputs = build_draw_rows_and_sigma_inputs(
        np_mod, d2, sampled_codes_by_draw, entry_idx_by_location, t_idx, h_end_idx, k, l_k_t_idx
    )
    draw_table = pd.DataFrame(draw_rows)
    if len(draw_table) == 0 or len(agg_table) == 0:
        return pd.DataFrame(), {}

    full = draw_table.merge(agg_table, on=["location_id", "entry_idx"], how="left")
    full["age_at_t"] = t - (year_start + full["entry_idx"])

    sigma_hat_by_draw = {}
    for draw_id, (durations, events) in sigma_inputs.items():
        sigma_hat_by_draw[draw_id] = m1.rayleigh_mle_sigma(np_mod, durations, events)

    return full, sigma_hat_by_draw


# ---------------------------------------------------------------------------
# Puntajes de los 6 metodos.
# ---------------------------------------------------------------------------


def score_methods(rt, np_mod, m1, table, sigma_hat_by_draw: dict, sigma_fixed: float, h: int, seed: int):
    """rayleigh_survival de 22_survival_models.py espera un sigma escalar en la mayoria de sus
    usos ahi, pero numpy permite sigma como array via broadcasting -- se reusa tal cual, funciona
    identico con sigma vector (un sigma_hat_t distinto por fila, segun el draw de esa fila)."""
    rng = np_mod.random.default_rng(seed)
    age = table["age_at_t"].to_numpy(dtype=float)
    scores = {}
    scores["aleatorio"] = rng.uniform(size=len(table))
    scores["mas_antiguo"] = age.copy()
    scores["rayleigh_30"] = 1.0 - m1.rayleigh_survival(np_mod, sigma_fixed, age + h) / m1.rayleigh_survival(np_mod, sigma_fixed, age)
    sigma_per_row = table["draw_id"].map(sigma_hat_by_draw).to_numpy(dtype=float)
    scores["rayleigh_estimado"] = 1.0 - m1.rayleigh_survival(np_mod, sigma_per_row, age + h) / m1.rayleigh_survival(np_mod, sigma_per_row, age)
    scores["perdida_reciente"] = table["loss_5y"].to_numpy(dtype=float)
    return scores


AGE_BASELINES = ["mas_antiguo", "rayleigh_30", "rayleigh_estimado"]
CONVERGENCE_TOL = 1e-9


def audit_age_baselines(rt, table, scores: dict, sigma_hat_by_draw: dict):
    """Auditoria de los TRES baselines de edad -- corre inmediatamente despues de score_methods(),
    sobre `table` (localidad x draw) TAL CUAL sale de build_landmark_dataset: sin multiplicar por
    A_i, sin promediar entre draws, sin agregar por localidad, sin mascaras ni desempate (esas
    cinco operaciones son de B2/25_resurvey_decision.py -- este archivo, R3, nunca las ejecuta; el
    audit corre en el UNICO punto de este archivo donde existen los tres scores crudos).

    POR QUE mas_antiguo == rayleigh_30 esta GARANTIZADO, no es un hallazgo empirico: mas_antiguo es
    age_at_t identico. rayleigh_30 usa sigma_fixed ESCALAR, el MISMO para las filas de todos los
    draws (broadcasting, ver score_methods) -- 1 - S(age+h)/S(age) con sigma constante es una
    funcion ESTRICTAMENTE CRECIENTE de age_at_t sola, igual para cualquier fila sin importar el
    draw: una transformacion monotona GLOBAL. AUROC y AUPRC dependen solo del ORDEN inducido por el
    score (Mann-Whitney / precision-recall por umbral), no de sus valores -- son invariantes a
    cualquier transformacion monotona estricta. Por construccion matematica, mas_antiguo y
    rayleigh_30 DEBEN producir el mismo AUROC y el mismo AUPRC, fila por fila, exactamente. Se
    assertea (no se reporta como resultado incierto): que falle seria una regresion real en
    score_methods o en sigma_fixed, no ruido de muestreo esperable.

    POR QUE rayleigh_estimado PUEDE diferir: sigma_hat_t = table["draw_id"].map(sigma_hat_by_draw)
    NO es un escalar global, varia POR DRAW (un sigma_hat distinto por landmark-draw, reestimado
    con las trayectorias observables hasta L_k,t -- ver bloque POR QUE del modulo). DENTRO de un
    draw fijo, sigma_hat_t es constante, asi que la transformacion sigue siendo estrictamente
    creciente en age_at_t ahi -- se assertea tambien, localizando el mecanismo: si este segundo
    assert fallara, el bug estaria en el CALCULO de rayleigh_estimado (sigma no constante dentro de
    un draw), no en el mecanismo esperado. Lo que NO esta garantizado es el orden ENTRE filas de
    DRAWS DISTINTOS: dos filas con la MISMA age_at_t en dos draws con sigma_hat_t diferente reciben
    scores distintos, y como AUROC/AUPRC se calculan sobre la tabla COMPLETA (todos los draws
    mezclados), esas inversiones cruzadas pueden (y en la practica, si sigma_hat_t varia entre
    draws, van a) cambiar el AUROC/AUPRC global respecto de mas_antiguo/rayleigh_30. Esta
    comparacion NO se assertea -- se mide, y se decide "converged" en base a la tolerancia.
    """
    np_mod, roc_auc_score, average_precision_score = rt["np"], rt["roc_auc_score"], rt["average_precision_score"]

    y_true = table["Z"].to_numpy()
    n = len(table)
    row_id = (
        table["location_id"].astype(str) + "_" + table["draw_id"].astype(str) + "_"
        + table["entry_idx"].astype(str)
    ).to_numpy()
    assert len(set(row_id)) == n, (
        f"row_id (location_id+draw_id+entry_idx) no es unico en {n} filas -- el audit exige una "
        "clave por fila-draw para poder afirmar 'las mismas filas' entre los tres baselines."
    )
    for name in AGE_BASELINES:
        assert name in scores, f"Falta el score '{name}' -- score_methods no lo genero."
        assert len(scores[name]) == n, (
            f"{name}: {len(scores[name])} valores, se esperaban {n} filas-draw de `table` -- los "
            "tres baselines deben evaluarse sobre exactamente las mismas filas, sin ponderar por "
            "A_i, promediar entre draws ni agregar por localidad (eso es B2, no R3)."
        )
    # Los tres se evaluan contra el MISMO y_true = table["Z"] por construccion (una sola variable,
    # nunca una copia por baseline) -- reafirmado explicitamente, no asumido.

    def _auc(name):
        if len(np_mod.unique(y_true)) < 2:
            return float("nan"), float("nan")
        s = scores[name]
        return float(roc_auc_score(y_true, s)), float(average_precision_score(y_true, s))

    auroc = {b: _auc(b)[0] for b in AGE_BASELINES}
    auprc = {b: _auc(b)[1] for b in AGE_BASELINES}

    assert abs(auroc["mas_antiguo"] - auroc["rayleigh_30"]) < CONVERGENCE_TOL, (
        f"mas_antiguo y rayleigh_30 DEBEN dar AUROC identico (transformacion monotona GLOBAL de "
        f"age_at_t, sigma escalar) pero difieren: {auroc['mas_antiguo']!r} vs "
        f"{auroc['rayleigh_30']!r}. Regresion real -- revisar sigma_fixed / score_methods."
    )
    assert abs(auprc["mas_antiguo"] - auprc["rayleigh_30"]) < CONVERGENCE_TOL, (
        f"mas_antiguo y rayleigh_30 DEBEN dar AUPRC identico pero difieren: "
        f"{auprc['mas_antiguo']!r} vs {auprc['rayleigh_30']!r}. Regresion real."
    )

    draw_ids = table["draw_id"].to_numpy()
    age = table["age_at_t"].to_numpy(dtype=float)
    r_est = scores["rayleigh_estimado"]
    for draw_id in np_mod.unique(draw_ids):
        mask = draw_ids == draw_id
        order_age = np_mod.argsort(age[mask], kind="stable")
        order_rest = np_mod.argsort(r_est[mask], kind="stable")
        assert np_mod.array_equal(order_age, order_rest), (
            f"rayleigh_estimado no es monotono en age_at_t DENTRO del draw {draw_id!r} -- esto "
            "apuntaria a un bug en el calculo de sigma_per_row (deberia ser constante dentro de "
            "un draw), no al mecanismo de divergencia ENTRE draws que este audit espera."
        )

    converged = (
        abs(auroc["mas_antiguo"] - auroc["rayleigh_estimado"]) < CONVERGENCE_TOL
        and abs(auprc["mas_antiguo"] - auprc["rayleigh_estimado"]) < CONVERGENCE_TOL
    )
    sigma_vals = np_mod.array(list(sigma_hat_by_draw.values()), dtype=float)
    return {
        "n_rows": n, "auroc": auroc, "auprc": auprc, "converged": converged,
        "unified_auroc": auroc["mas_antiguo"], "unified_auprc": auprc["mas_antiguo"],
        "sigma_hat_media": float(sigma_vals.mean()), "sigma_hat_de": float(sigma_vals.std()),
        "sigma_hat_min": float(sigma_vals.min()), "sigma_hat_max": float(sigma_vals.max()),
    }


# ---------------------------------------------------------------------------
# LightGBM y logistica: rolling-origin para hiperparametros, refit final, prediccion.
# ---------------------------------------------------------------------------


def fit_predict_lgb(lgb, params, train_df, predict_df):
    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    model.fit(train_df[COVARIATE_COLS], train_df["Z"])
    return model.predict_proba(predict_df[COVARIATE_COLS])[:, 1]


def build_logistic_pipeline(rt, params: dict):
    """Imputacion (mediana) + escalado (Z-score) + LogisticRegression -- las tres etapas se
    ajustan SOLO con el fold de entrenamiento que reciba .fit() (Pipeline de sklearn, sin fuga).
    Ver bloque POR QUE sobre por que hacen falta (LightGBM no las necesita, logistica si)."""
    return rt["make_pipeline"](
        rt["SimpleImputer"](strategy="median"),
        rt["StandardScaler"](),
        rt["LogisticRegression"](C=params["C"], max_iter=2000, random_state=42),
    )


def fit_predict_logistic(rt, params, train_df, predict_df):
    model = build_logistic_pipeline(rt, params)
    model.fit(train_df[COVARIATE_COLS], train_df["Z"])
    return model.predict_proba(predict_df[COVARIATE_COLS])[:, 1]


def rolling_origin_select_hyperparams(rt, fit_predict_fn, average_precision_score, landmark_tables: dict, train_landmarks: list, grid: list):
    """landmark_tables: {landmark_year: DataFrame ya construido}. Tres particiones que respetan
    el orden temporal -- ver bloque POR QUE. Generalizada sobre fit_predict_fn(params, train_df,
    predict_df) y una grilla de hiperparametros para poder reusarse tanto para LightGBM como para
    la regresion logistica, sin duplicar la logica de particion temporal."""
    pd = rt["pd"]
    splits = []
    for i in range(1, len(train_landmarks)):
        splits.append((train_landmarks[:i], train_landmarks[i]))

    grid_scores = []
    for params in grid:
        split_scores = []
        for train_lms, val_lm in splits:
            train_df = pd.concat([landmark_tables[t] for t in train_lms if len(landmark_tables[t])], ignore_index=True)
            val_df = landmark_tables[val_lm]
            if len(train_df) == 0 or len(val_df) == 0 or val_df["Z"].nunique() < 2:
                continue
            pred = fit_predict_fn(params, train_df, val_df)
            split_scores.append({"landmark_val": val_lm, "auprc": average_precision_score(val_df["Z"], pred)})
        mean_auprc = float(pd.DataFrame(split_scores)["auprc"].mean()) if split_scores else float("nan")
        grid_scores.append({"params": params, "mean_auprc_rolling": mean_auprc, "per_landmark": split_scores})

    grid_scores.sort(key=lambda r: (r["mean_auprc_rolling"] if r["mean_auprc_rolling"] == r["mean_auprc_rolling"] else -1), reverse=True)
    return grid_scores[0], grid_scores


# ---------------------------------------------------------------------------
# Metricas de R3 (individuales): Brier-5, deciles/ICI, AUROC, AUPRC.
# ---------------------------------------------------------------------------


def calibration_deciles(pd, np_mod, y_true, y_pred, n_bins: int = 10):
    order = np_mod.argsort(y_pred)
    bins = np_mod.array_split(order, n_bins)
    rows = []
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        rows.append({
            "decil": i + 1, "n": len(idx),
            "pred_medio": float(np_mod.mean(y_pred[idx])), "observado": float(np_mod.mean(y_true[idx])),
        })
    table = pd.DataFrame(rows)
    ici = float(np_mod.average(np_mod.abs(table["pred_medio"] - table["observado"]), weights=table["n"])) if len(table) else float("nan")
    return table, ici


def evaluate_method(rt, name, y_true, y_pred, is_probability: bool):
    np_mod, roc_auc_score, average_precision_score, brier_score_loss, pd = (
        rt["np"], rt["roc_auc_score"], rt["average_precision_score"], rt["brier_score_loss"], rt["pd"]
    )
    out = {"metodo": name}
    if len(np_mod.unique(y_true)) < 2:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y_true, y_pred))
        out["auprc"] = float(average_precision_score(y_true, y_pred))
    out["prevalencia"] = float(np_mod.mean(y_true))
    if is_probability:
        out["brier_5y"] = float(brier_score_loss(y_true, y_pred))
        deciles, ici = calibration_deciles(pd, np_mod, y_true, y_pred)
        out["ici"] = ici
    else:
        out["brier_5y"] = float("nan")
        out["ici"] = float("nan")
        deciles = pd.DataFrame()
    return out, deciles


# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    np, pd, gpd, lgb = rt["np"], rt["pd"], rt["gpd"], rt["lgb"]

    print("R3 - Landmark forecasting a 5 años (PLAN_V3_EJECUTABLE.md)")
    print("=================================================================")

    d1 = load_module(D1_PATH, "d1_trajectory_extract")
    d2 = load_module(D2_PATH, "d2_build_survival_dataset")
    m1 = load_module(M1_PATH, "m1_survival_models")

    meta = json_load_metadata(args.trajectory_dir)
    year_start, year_max = meta["year_start"], meta["year_max"]
    t_max_confirmable = year_max - args.horizon_years - args.k + 1
    print(f"Ventana: {year_start}-{year_max} (run_metadata.json). h={args.horizon_years}, k={args.k}. "
          f"t_max confirmable = Y_max-h-k+1 = {t_max_confirmable}.")
    for t in args.train_landmarks + [args.test_landmark]:
        if t > t_max_confirmable:
            raise SystemExit(f"Landmark {t} > t_max confirmable ({t_max_confirmable}) -- ventana "
                              f"({t},{t+args.horizon_years}] incompleta con Y_max={year_max}. Corregir landmarks.")
    if max(args.train_landmarks) >= args.test_landmark:
        raise SystemExit("Los landmarks de entrenamiento deben ser estrictamente anteriores al "
                          "landmark de test (fuga temporal si no) -- ver bloque POR QUE.")

    df = d1.load_occurrences(pd, args.occurrences_csv, year_start)
    df = d1.filter_year_upper_bound(df, year_max, args.year_max_tolerance)
    print(f"Registros validos: {len(df):,}")
    entries_by_location = build_entries_by_location(pd, df)
    print(f"Localidades unicas: {len(entries_by_location):,}")

    sampled_codes_by_draw, draws_available = load_sampled_codes(
        pd, args.survival_dir, args.k, args.buffer_m, args.n_draws, args.max_draws
    )
    print(f"Draws Monte Carlo reusados de D2: {len(draws_available)}")

    combined = d1.load_buffer_trajectory(pd, args.trajectory_dir / ".chunks", args.buffer_m)
    if combined is None:
        raise SystemExit(f"Sin checkpoints de trayectoria para buffer {args.buffer_m}m.")
    loc_code_tables = d2.build_location_code_tables(pd, combined)

    all_landmarks = args.train_landmarks + [args.test_landmark]
    landmark_tables, sigma_hat_by_landmark = {}, {}
    for t in all_landmarks:
        print(f"\n--- Construyendo dataset del landmark {t} (ventana {t}->{t+args.horizon_years}) ---")
        table, sigma_hat_by_draw = build_landmark_dataset(
            rt, d2, m1, t, args.horizon_years, args.k, year_start, year_max,
            entries_by_location, loc_code_tables, sampled_codes_by_draw, LOSS_WINDOW_YEARS,
        )
        landmark_tables[t] = table
        sigma_hat_by_landmark[t] = sigma_hat_by_draw
        if len(table):
            print(f"  Filas (localidad x draw en riesgo): {len(table):,}. "
                  f"Z=1: {int(table['Z'].sum()):,} ({table['Z'].mean():.2%}). "
                  f"sigma_hat_t por draw: media={np.mean(list(sigma_hat_by_draw.values())):.2f}")
        else:
            print("  AVISO: dataset vacio para este landmark.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Hiperparametros del landmark boosting: SOLO rolling-origin sobre landmarks de entrenamiento ---
    print("\n=== Seleccion de hiperparametros (rolling-origin, SOLO landmarks de entrenamiento) ===")
    best, grid_scores = rolling_origin_select_hyperparams(
        rt, lambda params, tr, va: fit_predict_lgb(lgb, params, tr, va),
        rt["average_precision_score"], landmark_tables, args.train_landmarks, LGBM_GRID,
    )
    print(f"Mejor combinacion: {best['params']} (AUPRC rolling media={best['mean_auprc_rolling']:.4f})")
    rolling_by_landmark_rows = []
    for combo in grid_scores:
        for s in combo["per_landmark"]:
            rolling_by_landmark_rows.append({"params": str(combo["params"]), **s})
    rolling_table = pd.DataFrame(rolling_by_landmark_rows)
    rolling_table.to_csv(args.output_dir / "rolling_origin_by_landmark.csv", index=False)
    print("Desempeño por landmark de VALIDACION interna (todas las combinaciones probadas):")
    print(rolling_table.to_string(index=False))

    # --- Hiperparametros del baseline logistico: MISMO protocolo rolling-origin, misma restriccion
    # temporal (solo landmarks de entrenamiento, test congelado nunca participa) -- ver bloque POR QUE ---
    print("\n=== Seleccion de hiperparametros (rolling-origin, logistica, SOLO landmarks de entrenamiento) ===")
    best_logistic, grid_scores_logistic = rolling_origin_select_hyperparams(
        rt, lambda params, tr, va: fit_predict_logistic(rt, params, tr, va),
        rt["average_precision_score"], landmark_tables, args.train_landmarks, LOGISTIC_GRID,
    )
    print(f"Mejor combinacion (logistica): {best_logistic['params']} "
          f"(AUPRC rolling media={best_logistic['mean_auprc_rolling']:.4f})")
    rolling_by_landmark_rows_logistic = []
    for combo in grid_scores_logistic:
        for s in combo["per_landmark"]:
            rolling_by_landmark_rows_logistic.append({"params": str(combo["params"]), **s})
    rolling_table_logistic = pd.DataFrame(rolling_by_landmark_rows_logistic)
    rolling_table_logistic.to_csv(args.output_dir / "rolling_origin_by_landmark_logistic.csv", index=False)
    print("Desempeño por landmark de VALIDACION interna, logistica (todas las combinaciones probadas):")
    print(rolling_table_logistic.to_string(index=False))

    # --- Refit final con TODOS los landmarks de entrenamiento, hiperparametros ya fijados ---
    train_df_final = pd.concat([landmark_tables[t] for t in args.train_landmarks if len(landmark_tables[t])], ignore_index=True)
    print(f"\nRefit final del landmark boosting sobre {len(train_df_final):,} filas "
          f"({len(args.train_landmarks)} landmarks de entrenamiento combinados).")
    final_model = lgb.LGBMClassifier(**best["params"], random_state=42, verbose=-1)
    final_model.fit(train_df_final[COVARIATE_COLS], train_df_final["Z"])

    print(f"Refit final de la logistica sobre las mismas {len(train_df_final):,} filas.")
    final_model_logistic = build_logistic_pipeline(rt, best_logistic["params"])
    final_model_logistic.fit(train_df_final[COVARIATE_COLS], train_df_final["Z"])

    # --- TEST UNICO CONGELADO ---
    print(f"\n=== TEST CONGELADO: landmark {args.test_landmark} -> {args.test_landmark + args.horizon_years} (tocado UNA sola vez) ===")
    test_table = landmark_tables[args.test_landmark]
    if len(test_table) == 0:
        raise SystemExit("Dataset del landmark de test vacio -- no se puede evaluar.")
    scores = score_methods(rt, np, m1, test_table, sigma_hat_by_landmark[args.test_landmark], args.sigma_fixed, args.horizon_years, args.random_seed)
    scores["landmark_boosting"] = final_model.predict_proba(test_table[COVARIATE_COLS])[:, 1]
    scores["landmark_logistic"] = final_model_logistic.predict_proba(test_table[COVARIATE_COLS])[:, 1]

    # --- Auditoria: los tres baselines de edad, ¿AUROC/AUPRC identicos? Ver bloque POR QUE de
    # audit_age_baselines -- corre AQUI porque es el unico punto de este archivo donde existen los
    # tres scores crudos (localidad x draw), antes de cualquier operacion de B2. ---
    print("\n=== Auditoria: baselines de edad (mas_antiguo, rayleigh_30, rayleigh_estimado) ===")
    audit = audit_age_baselines(rt, test_table, scores, sigma_hat_by_landmark[args.test_landmark])
    print(f"  Filas-draw auditadas: {audit['n_rows']:,}. row_id y_true unicos: OK.")
    print(f"  ASSERT OK: mas_antiguo == rayleigh_30 (AUROC={audit['auroc']['mas_antiguo']:.10f}, "
          f"AUPRC={audit['auprc']['mas_antiguo']:.10f}) -- transformacion monotona GLOBAL de "
          "age_at_t (sigma escalar), garantizado matematicamente.")
    print("  ASSERT OK: rayleigh_estimado es monotono en age_at_t DENTRO de cada draw (mismo "
          "sigma_hat_t ahi).")
    diff_pairs = list(DIFF_PAIRS)
    if audit["converged"]:
        print(f"  CONVERGEN LOS TRES: rayleigh_estimado tambien coincide "
              f"(AUROC={audit['auroc']['rayleigh_estimado']:.10f}, "
              f"AUPRC={audit['auprc']['rayleigh_estimado']:.10f}). Se reporta un unico baseline "
              "'edad_unificada' y la diferencia pareada se recalcula contra el.")
        scores["edad_unificada"] = scores["rayleigh_30"]  # alias: orden identico por construccion
        diff_pairs = [
            (a if a not in AGE_BASELINES else "edad_unificada", b if b not in AGE_BASELINES else "edad_unificada")
            for (a, b) in diff_pairs
        ]
        diff_pairs = list(dict.fromkeys(diff_pairs))  # sin duplicados si dos pares colapsan al mismo
    else:
        print(f"  NO CONVERGEN: rayleigh_estimado difiere -- "
              f"AUROC={audit['auroc']['rayleigh_estimado']:.10f} (vs {audit['unified_auroc']:.10f} "
              f"de mas_antiguo/rayleigh_30, delta={audit['auroc']['rayleigh_estimado']-audit['unified_auroc']:+.10f}), "
              f"AUPRC={audit['auprc']['rayleigh_estimado']:.10f} (vs {audit['unified_auprc']:.10f}, "
              f"delta={audit['auprc']['rayleigh_estimado']-audit['unified_auprc']:+.10f}).")
        print(f"  Mecanismo (no un bug): sigma_hat_t varia POR DRAW -- media={audit['sigma_hat_media']:.2f}, "
              f"DE={audit['sigma_hat_de']:.2f}, rango=[{audit['sigma_hat_min']:.2f}, {audit['sigma_hat_max']:.2f}] "
              "-- en vez de un escalar global como sigma_fixed. El orden coincide con age_at_t DENTRO "
              "de cada draw (ya verificado) pero NO entre draws con sigma_hat_t distinto: eso rompe "
              "la monotonia GLOBAL que rayleigh_30 si preserva. No se consolida -- se reportan los "
              "tres baselines por separado, sin cambios en diff_pairs.")

    y_true = test_table["Z"].to_numpy()
    is_probability = {"aleatorio": False, "mas_antiguo": False, "perdida_reciente": False,
                       "rayleigh_30": True, "rayleigh_estimado": True, "landmark_boosting": True,
                       "landmark_logistic": True, "edad_unificada": False}
    metric_rows, deciles_by_method = [], {}
    for name, y_pred in scores.items():
        row, deciles = evaluate_method(rt, name, y_true, y_pred, is_probability[name])
        metric_rows.append(row)
        deciles_by_method[name] = deciles
    metric_table = pd.DataFrame(metric_rows)
    print(metric_table.to_string(index=False))

    # --- Bootstrap de bloque espacial sobre el test congelado -- ver bloque POR QUE ---
    print("\nBootstrap de bloque espacial (localidad -> bloque de 100 km, IC95%) sobre el test congelado...")
    blocks = m1.assign_spatial_blocks(gpd, np, pd, test_table["location_id"])
    unique_blocks = np.unique(blocks)
    block_to_idx = {b: np.where(blocks == b)[0] for b in unique_blocks}
    rng = np.random.default_rng(args.random_seed)
    boot_rows = {name: [] for name in scores}
    diff_samples = {pair: {"auroc": [], "auprc": []} for pair in diff_pairs}
    prevalence_samples = []
    for _ in range(args.n_bootstrap):
        # Mismos bloques/localidades remuestreados UNA vez por iteracion, reusados para los siete
        # metodos y para las diferencias pareadas -- asi la diferencia entre metodos se calcula
        # dentro de la MISMA remuestra (pareada), no restando dos IC independientes.
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_to_idx[b] for b in sampled_blocks])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        iter_auroc, iter_auprc = {}, {}
        for name, y_pred in scores.items():
            a = float(rt["roc_auc_score"](yt, y_pred[idx]))
            p = float(rt["average_precision_score"](yt, y_pred[idx]))
            iter_auroc[name], iter_auprc[name] = a, p
            boot_rows[name].append({"auroc": a, "auprc": p})
        prevalence_samples.append(float(np.mean(yt)))
        for pair in diff_pairs:
            a_name, b_name = pair
            diff_samples[pair]["auroc"].append(iter_auroc[a_name] - iter_auroc[b_name])
            diff_samples[pair]["auprc"].append(iter_auprc[a_name] - iter_auprc[b_name])
    ci_rows = []
    for name, samples in boot_rows.items():
        if not samples:
            continue
        auroc_vals = np.array([s["auroc"] for s in samples])
        auprc_vals = np.array([s["auprc"] for s in samples])
        ci_rows.append({
            "metodo": name,
            "auroc_ic95_lo": float(np.percentile(auroc_vals, 2.5)), "auroc_ic95_hi": float(np.percentile(auroc_vals, 97.5)),
            "auprc_ic95_lo": float(np.percentile(auprc_vals, 2.5)), "auprc_ic95_hi": float(np.percentile(auprc_vals, 97.5)),
        })
    ci_table = pd.DataFrame(ci_rows)
    print(ci_table.to_string(index=False))

    prevalence_vals = np.array(prevalence_samples)
    prevalence_point = float(y_true.mean())
    prevalence_ic_lo, prevalence_ic_hi = float(np.percentile(prevalence_vals, 2.5)), float(np.percentile(prevalence_vals, 97.5))
    print(f"\nPrevalencia del outcome (Z=1) en el test congelado: {prevalence_point:.4%} "
          f"(IC95% bootstrap de bloque espacial: [{prevalence_ic_lo:.4%}, {prevalence_ic_hi:.4%}]). "
          "Referencia para leer los AUPRC: un ranking sin señal tiene AUPRC esperado ~= prevalencia.")

    print("\nBootstrap PAREADO de diferencias entre metodos (mismos bloques espaciales y las mismas "
          "localidades remuestreadas a la vez en los tres metodos de cada comparacion), IC95%:")
    diff_rows = []
    for pair in diff_pairs:
        a_name, b_name = pair
        auroc_d = np.array(diff_samples[pair]["auroc"])
        auprc_d = np.array(diff_samples[pair]["auprc"])
        diff_rows.append({
            "comparacion": f"{a_name} - {b_name}",
            "auroc_diff_media": float(np.mean(auroc_d)) if len(auroc_d) else float("nan"),
            "auroc_diff_ic95_lo": float(np.percentile(auroc_d, 2.5)) if len(auroc_d) else float("nan"),
            "auroc_diff_ic95_hi": float(np.percentile(auroc_d, 97.5)) if len(auroc_d) else float("nan"),
            "auprc_diff_media": float(np.mean(auprc_d)) if len(auprc_d) else float("nan"),
            "auprc_diff_ic95_lo": float(np.percentile(auprc_d, 2.5)) if len(auprc_d) else float("nan"),
            "auprc_diff_ic95_hi": float(np.percentile(auprc_d, 97.5)) if len(auprc_d) else float("nan"),
        })
    diff_table = pd.DataFrame(diff_rows)
    print(diff_table.to_string(index=False))

    # --- salidas ---
    metric_table = metric_table.merge(ci_table, on="metodo", how="left")
    metric_table.to_csv(args.output_dir / f"test_landmark_{args.test_landmark}_metrics.csv", index=False)
    diff_table.to_csv(args.output_dir / f"test_landmark_{args.test_landmark}_bootstrap_diffs.csv", index=False)
    for name, deciles in deciles_by_method.items():
        if len(deciles):
            deciles.to_csv(args.output_dir / f"test_landmark_{args.test_landmark}_deciles_{name}.csv", index=False)
    test_table.assign(**{f"score_{n}": s for n, s in scores.items()}).to_csv(
        args.output_dir / f"test_landmark_{args.test_landmark}_predictions.csv", index=False
    )

    diagnostics = [
        "R3 diagnostics - landmark forecasting a 5 años (PLAN_V3_EJECUTABLE.md)",
        "===========================================================================",
        f"Ventana: {year_start}-{year_max}. h={args.horizon_years}, k={args.k}. "
        f"t_max confirmable={t_max_confirmable}.",
        f"Landmarks de entrenamiento (rolling-origin interno): {args.train_landmarks}.",
        f"Landmark de TEST (congelado, tocado una sola vez): {args.test_landmark} -> "
        f"{args.test_landmark + args.horizon_years}.",
        f"Draws Monte Carlo reusados de D2 (mismos que R2): {len(draws_available)}.",
        "",
        "SELECCION DE HIPERPARAMETROS -- LANDMARK BOOSTING (rolling-origin, SOLO landmarks de "
        "entrenamiento, el test congelado NUNCA participa de esto):",
        f"  Mejor combinacion: {best['params']}, AUPRC rolling media={best['mean_auprc_rolling']:.4f}",
        rolling_table.to_string(index=False),
        "",
        "SELECCION DE HIPERPARAMETROS -- LOGISTICA (mismo protocolo rolling-origin, mismos "
        "landmarks de entrenamiento, mismas COVARIATE_COLS y mismos draws que el boosting -- ver "
        "bloque POR QUE):",
        f"  Mejor combinacion: {best_logistic['params']}, AUPRC rolling media={best_logistic['mean_auprc_rolling']:.4f}",
        rolling_table_logistic.to_string(index=False),
        "",
        f"Filas por landmark (localidad x draw en riesgo, ver bloque POR QUE sobre el gate de B1):",
    ]
    diagnostics += [
        "",
        "*** AUDITORIA: baselines de edad (mas_antiguo, rayleigh_30, rayleigh_estimado) sobre las "
        "MISMAS filas-draw crudas, ANTES de A_i/promedio-entre-draws/agregacion-por-localidad/"
        "mascaras/desempate (esas operaciones son de B2, 25_resurvey_decision.py) ***",
        f"  Filas-draw auditadas: {audit['n_rows']:,}. row_id y_true unicos: OK (asserts).",
        f"  mas_antiguo == rayleigh_30 (garantizado matematicamente, sigma escalar global): "
        f"AUROC={audit['auroc']['mas_antiguo']:.10f}, AUPRC={audit['auprc']['mas_antiguo']:.10f}.",
        f"  rayleigh_estimado, monotono en age_at_t DENTRO de cada draw (assert OK) pero NO entre "
        f"draws (sigma_hat_t media={audit['sigma_hat_media']:.2f}, DE={audit['sigma_hat_de']:.2f}, "
        f"rango=[{audit['sigma_hat_min']:.2f}, {audit['sigma_hat_max']:.2f}]): "
        f"AUROC={audit['auroc']['rayleigh_estimado']:.10f}, AUPRC={audit['auprc']['rayleigh_estimado']:.10f}.",
        (f"  CONVERGEN LOS TRES -- se reporta 'edad_unificada' y diff_pairs se recalculo contra ese "
         "baseline unico." if audit["converged"] else
         f"  NO CONVERGEN: delta AUROC={audit['auroc']['rayleigh_estimado']-audit['unified_auroc']:+.10f}, "
         f"delta AUPRC={audit['auprc']['rayleigh_estimado']-audit['unified_auprc']:+.10f} respecto de "
         "mas_antiguo/rayleigh_30. Mecanismo: sigma_hat_t varia POR DRAW en vez de ser un escalar "
         "global -- rompe la monotonia GLOBAL (entre draws) que preserva rayleigh_30, no un bug de "
         "calculo (el orden SI coincide dentro de cada draw individual). Los tres baselines se "
         "reportan por separado, sin cambios en las comparaciones pareadas."),
    ]
    for t in all_landmarks:
        tbl = landmark_tables[t]
        if len(tbl):
            diagnostics.append(f"  {t}: {len(tbl):,} filas, Z=1 en {tbl['Z'].mean():.2%}")
        else:
            diagnostics.append(f"  {t}: vacio")
    diagnostics += [
        "",
        f"*** TEST CONGELADO {args.test_landmark}->{args.test_landmark+args.horizon_years}, siete metodos "
        "(Brier/deciles/ICI SOLO para los cuatro que son probabilidades: Rayleigh-30, Rayleigh "
        "estimado, landmark boosting, landmark logistica -- los tres de ranking se evaluan solo "
        "con AUROC/AUPRC, ver bloque POR QUE): ***",
        metric_table.to_string(index=False),
        "",
        "IC95% por bootstrap de BLOQUE ESPACIAL (100 km, todas las filas de una localidad se "
        "mueven juntas al remuestrear):",
        ci_table.to_string(index=False),
        "",
        f"Prevalencia del outcome (Z=1) en el test congelado: {prevalence_point:.4%} (IC95% "
        f"bootstrap de bloque espacial: [{prevalence_ic_lo:.4%}, {prevalence_ic_hi:.4%}]). Se "
        "reporta junto a los AUPRC porque el AUPRC de un ranking sin señal es aprox. igual a la "
        "prevalencia (no a 0.5 como el AUROC) -- sin esta referencia un AUPRC de 0.037 parece bajo "
        "en terminos absolutos, pero hay que compararlo contra la prevalencia real de ~1%.",
        "",
        "Bootstrap PAREADO de diferencias entre metodos (mismos bloques espaciales y las mismas "
        "localidades remuestreadas a la vez en las tres comparaciones, IC95%; un intervalo que no "
        "cruza 0 indica diferencia consistente bajo la incertidumbre espacial):",
        diff_table.to_string(index=False),
    ]
    (args.output_dir / f"diagnostics_test_landmark_{args.test_landmark}.txt").write_text(
        "\n".join(str(l) for l in diagnostics) + "\n", encoding="utf-8"
    )
    print(f"\nGuardado: {args.output_dir / f'diagnostics_test_landmark_{args.test_landmark}.txt'}")


def json_load_metadata(trajectory_dir: Path) -> dict:
    import json
    meta_path = trajectory_dir / "run_metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"No se encontro {meta_path} -- correr 20_trajectory_extract.py primero.")
    return json.loads(meta_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
