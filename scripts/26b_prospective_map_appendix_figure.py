# CCAI OBSOLESCENCIA - version de apendice (mapa solo, a mayor tamaño) de fig_prospective_map.
# Lee el MISMO CSV/JSON que 26_prospective_map_figure.py (results/decision/...), NO RECALCULA
# NINGUN SCORE. Es el mismo mapa del cuerpo, redibujado mas grande para el apendice.
"""
POR QUE
-------
La figura compacta del cuerpo (fig_prospective_map.pdf) prioriza legibilidad a tamaño de columna
de paper; la version de apendice puede permitirse mas espacio para el mismo mapa (marcadores mas
grandes, mas aire alrededor de las etiquetas de estado) sin cambiar ningun dato ni criterio de
color respecto de la version del cuerpo.

REVERTIDO A UN SOLO PANEL (pasada 2026-08-24): una version anterior de este script agregaba un
segundo panel con barras de composicion por estado (universo vs top-100). Se quito a pedido -- el
apendice vuelve a ser solo el mapa, sin el panel de barras ni los titulos internos "(a) ..." /
"(b) ...". El rotulo de la barra de color tambien se acorta a "Predicted risk" (se quita el sufijo
"(p99 clip)"): el recorte en el percentil 99 se sigue aplicando exactamente igual (vmax = p99,
colorbar con extend='max'), solo cambia el texto visible.

MISMO CONTORNO DE BIOMA Y RECORTE DE EJES QUE fig_prospective_map.pdf (ver el bloque POR QUE de
26_prospective_map_figure.py, "CUARTA PASADA" para la causa raiz del bounding box de la cohorte):
xlim=(-54.3,-43.2), ylim=(-26.3,-19.0), mismas cinco etiquetas de estado. Aqui se dibuja mas grande
(marcadores s=1.5 / top-100 s=22 en vez de s=0.8 / s=12) porque el apendice tiene mas espacio que
la figura compacta del cuerpo.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

SEQUENTIAL_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
TOPB_HIGHLIGHT_COLOR = "#eb6834"
TOPB_BUDGET_DEFAULT = 100

MAIN_XLIM = (-54.3, -43.2)
MAIN_YLIM = (-26.3, -19.0)

STATE_LABELS = [
    ("SÃO PAULO", -22.2, -49.3),
    ("PARANÁ", -24.6, -51.6),
    ("MINAS GERAIS", -20.2, -45.6),
    ("RIO DE J.", -22.4, -43.8),
    ("M. G. DO SUL", -21.3, -53.4),
]


def parse_args() -> argparse.Namespace:
    obs_root = SCRIPT_DIR.parent
    decision_dir = obs_root / "results" / "decision"
    parser = argparse.ArgumentParser(
        description="Version de apendice (un panel, mas grande) del mapa prospectivo post-2024: "
        "el mismo mapa del cuerpo, redibujado a mayor tamaño."
    )
    parser.add_argument(
        "--localidades-csv", type=Path,
        default=decision_dir / "prospective_localidades_2024_NO_OBSERVADO.csv",
    )
    parser.add_argument(
        "--topb-json", type=Path,
        default=decision_dir / "prospective_topB_candidatos_2024_NO_OBSERVADO.json",
    )
    parser.add_argument("--topb-budget", type=int, default=TOPB_BUDGET_DEFAULT)
    parser.add_argument(
        "--biome-boundary", type=Path,
        default=obs_root / "data" / "mata_atlantica.gpkg",
    )
    parser.add_argument(
        "--output-figure", type=Path,
        default=obs_root / "figures" / "fig_prospective_map_appendix.pdf",
    )
    return parser.parse_args()


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as patheffects
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:
        missing = getattr(exc, "name", "a required package")
        raise SystemExit(
            f"Falta una dependencia: {missing}\nInstalar con:\n"
            "  pip install pandas numpy geopandas matplotlib"
        ) from exc
    return {
        "np": np, "pd": pd, "gpd": gpd, "plt": plt,
        "patheffects": patheffects, "LinearSegmentedColormap": LinearSegmentedColormap,
    }


def split_location_id(np_mod, pd_mod, location_ids):
    parts = pd_mod.Series(location_ids).str.split("_", n=1, expand=True)
    lat = parts[0].to_numpy(dtype=float)
    lon = parts[1].to_numpy(dtype=float)
    return lat, lon


def load_biome_boundary(gpd_mod, path: Path):
    if not path.exists():
        print(f"AVISO: no se encontro {path} -- se grafica sin contorno de referencia.")
        return None
    biome = gpd_mod.read_file(path)
    if biome.crs is not None and biome.crs.to_epsg() != 4326:
        biome = biome.to_crs(4326)
    print(f"Contorno del bioma: {path} ({len(biome)} geometria(s), reproyectado a EPSG:4326).")
    return biome


def draw_state_labels(ax, patheffects_mod, fontsize: float, color: str = "0.3") -> None:
    for text, label_lat, label_lon in STATE_LABELS:
        ax.annotate(
            text, xy=(label_lon, label_lat), ha="center", va="center",
            fontsize=fontsize, color=color, zorder=6,
            path_effects=[patheffects_mod.withStroke(linewidth=1.5, foreground="white")],
        )


def draw_map_panel(ax, np, gpd, plt, patheffects, LinearSegmentedColormap,
                    lat, lon, r_bar, is_topb, biome, vmax, topb_budget,
                    point_size: float, topb_size: float, label_fontsize: float) -> "object":
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE_RAMP)
    order = np.argsort(r_bar)

    if biome is not None:
        biome.plot(ax=ax, facecolor="none", edgecolor="0.6", linewidth=0.4, zorder=1)

    sc = ax.scatter(
        lon[order], lat[order], c=r_bar[order], cmap=cmap, vmin=0.0, vmax=vmax,
        s=point_size, alpha=0.5, linewidths=0, rasterized=True, zorder=2,
    )
    ax.scatter(
        lon[is_topb], lat[is_topb], facecolors="none", edgecolors=TOPB_HIGHLIGHT_COLOR,
        s=topb_size, linewidths=0.6, marker="o", label=f"Top-{topb_budget} priority",
        zorder=5,
    )

    draw_state_labels(ax, patheffects, fontsize=label_fontsize, color="0.3")

    ax.set_xlim(*MAIN_XLIM)
    ax.set_ylim(*MAIN_YLIM)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.9, handlelength=1.0, handletextpad=0.3,
              borderaxespad=0.25, markerscale=1.3)
    return sc


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    np, pd, gpd, plt = rt["np"], rt["pd"], rt["gpd"], rt["plt"]
    patheffects = rt["patheffects"]
    LinearSegmentedColormap = rt["LinearSegmentedColormap"]

    print("Figura (apendice, un panel) - mapa prospectivo post-2024")
    print("======================================================================")

    df = pd.read_csv(args.localidades_csv)
    extrap_label = df["extrapolacion"].iloc[0]
    print(f"Localidades: {len(df):,} ({args.localidades_csv})")
    print(f"Rotulo de extrapolacion (fuente): {extrap_label}")

    topb_raw = json.loads(args.topb_json.read_text(encoding="utf-8"))
    topb_key = str(args.topb_budget)
    if topb_key not in topb_raw:
        raise SystemExit(f"Presupuesto {args.topb_budget} no esta en {args.topb_json} "
                          f"(disponibles: {list(topb_raw.keys())}).")
    topb_ids = set(topb_raw[topb_key])
    print(f"Top-{args.topb_budget} candidatos: {len(topb_ids):,} localidades ({args.topb_json})")

    lat, lon = split_location_id(np, pd, df["location_id"])
    r_bar = df["r_bar_prospectivo_NO_OBSERVADO"].to_numpy(dtype=float)
    is_topb = df["location_id"].isin(topb_ids).to_numpy()

    vmax = float(np.percentile(r_bar, 99))
    print(f"r_bar prospectivo: media={r_bar.mean():.4f}, p50={np.median(r_bar):.4f}, "
          f"p99={vmax:.4f}, max={r_bar.max():.4f} (escala de color recortada en p99).")

    biome = load_biome_boundary(gpd, args.biome_boundary)

    n_outside_main = int(np.sum(
        (lon < MAIN_XLIM[0]) | (lon > MAIN_XLIM[1]) | (lat < MAIN_YLIM[0]) | (lat > MAIN_YLIM[1])
    ))
    pct_outside_main = 100.0 * n_outside_main / len(df)

    plt.rcParams.update({
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
        'lines.linewidth': 1.0, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'savefig.dpi': 300, 'pdf.fonttype': 42,
    })

    fig, ax_map = plt.subplots(figsize=(3.6, 3.0), constrained_layout=True)

    sc = draw_map_panel(
        ax_map, np, gpd, plt, patheffects, LinearSegmentedColormap,
        lat, lon, r_bar, is_topb, biome, vmax, args.topb_budget,
        point_size=1.5, topb_size=22, label_fontsize=6,
    )
    cbar = fig.colorbar(sc, ax=ax_map, extend="max", orientation="horizontal", shrink=0.75, pad=0.02)
    cbar.set_label("Predicted risk")
    cbar.ax.tick_params(labelsize=6)

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = args.output_figure.parent / "_backup"
    if args.output_figure.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{args.output_figure.stem}_{stamp}{args.output_figure.suffix}"
        shutil.copy2(args.output_figure, backup_path)
        print(f"Respaldo (figura anterior): {backup_path}")

    fig.canvas.draw()
    fig.savefig(args.output_figure, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(args.output_figure.with_suffix(".png"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"Guardado (figura): {args.output_figure}")
    print(f"Guardado (figura, png): {args.output_figure.with_suffix('.png')}")

    diag_path = args.output_figure.with_name(args.output_figure.stem + "_diagnostics.txt")
    diagnostics = [
        "Figura (apendice, un panel) - mapa prospectivo post-2024 - diagnostics",
        "==============================================================================",
        f"Fuente: {args.localidades_csv}",
        f"Rotulo de extrapolacion: {extrap_label}",
        f"Localidades: {len(df):,}.",
        f"Top-{args.topb_budget} candidatos (resaltados): {len(topb_ids):,} ({args.topb_json}).",
        f"r_bar prospectivo: media={r_bar.mean():.6f}, p50={np.median(r_bar):.6f}, "
        f"p99={vmax:.6f} (techo de la escala de color, colorbar sin sufijo de texto), "
        f"max={r_bar.max():.6f}.",
        f"xlim={MAIN_XLIM}, ylim={MAIN_YLIM}. "
        f"Localidades fuera del recuadro: {n_outside_main:,} de {len(df):,} ({pct_outside_main:.2f}%).",
    ]
    diag_path.write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
    print(f"Guardado (diagnostics): {diag_path}")


if __name__ == "__main__":
    main()
