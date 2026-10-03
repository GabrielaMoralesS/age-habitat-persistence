# CCAI OBSOLESCENCIA - B2 auxiliar (RENDER-ONLY): re-dibuja paper/figures/fig_cumulative_gain.pdf
# al tamaño físico exacto del panel C de fig:main (0.32\textwidth = 1.76in), leyendo UNICAMENTE
# paper/figures/fig_cumulative_gain_data.csv -- la curva de ganancia acumulada a nivel de localidad
# (presupuesto absoluto 0-2000) que 25b_cumulative_gain_locations_figure.py YA CALCULO Y GUARDO
# (bootstrap de bloque espacial, 1000 iteraciones, ver el bloque POR QUE de ese script). NO CONSULTA
# EARTH ENGINE, NO ENTRENA NINGUN MODELO, NO RECALCULA NADA: no importa geopandas/22_survival_models,
# solo pandas+matplotlib sobre el CSV ya persistido.
"""
POR QUE
-------
25b_cumulative_gain_locations_figure.py corre un bootstrap de bloque espacial de 1000 iteraciones
(ordenar ~32.000 localidades x 3 metodos por iteracion) solo para producir la curva que YA GUARDA
en fig_cumulative_gain_data.csv (chance_diagonal + *_mean/*_ci95_lo/*_ci95_hi por metodo, malla de
presupuesto 1-2000 localidades). Repetir ese bootstrap solo para cambiar tamaño de fuente y figsize
es trabajo (y costo) innecesario. Este script es la mitad "solo grafica" de 25b, separada para que
regenerar la figura al tamaño de impresion final no dependa de repetir el bootstrap.

TAMAÑO FISICO: panel C de fig:main se inserta a 0.32\\textwidth = 0.32*5.5in = 1.76in
(\\includegraphics[width=\\linewidth] dentro de un minipage de ese ancho) -- figsize se fija a ese
ancho exacto para que LaTeX no reescale la figura al compilar.

ETIQUETAS DE METODO SIN CAMBIOS: "Chance baseline", "Landmark boosting", "Weighted recent loss" y
"Rayleigh sigma=30" ya eran suficientemente cortas para 1.76in de ancho a 6pt -- no estaban en la
lista de leyendas a acortar, se mantienen igual que en 25b_cumulative_gain_locations_figure.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

METHODS = ["landmark_boosting", "perdida_reciente_ponderado", "rayleigh_30"]
METHOD_LABELS = {
    "landmark_boosting": "Landmark boosting",
    "perdida_reciente_ponderado": "Weighted recent loss",
    "rayleigh_30": r"Rayleigh $\sigma=30$",
}
METHOD_COLORS = {
    "landmark_boosting": "#4c72b0",
    "perdida_reciente_ponderado": "#8172b2",
    "rayleigh_30": "#c44e52",
}


def parse_args() -> argparse.Namespace:
    obs_root = SCRIPT_DIR.parent
    fig_dir = obs_root / "figures"
    parser = argparse.ArgumentParser(
        description="Figura B2 (panel C, RENDER-ONLY) - curva de ganancia acumulada a nivel de "
        "localidad, desde fig_cumulative_gain_data.csv ya calculado por "
        "25b_cumulative_gain_locations_figure.py."
    )
    parser.add_argument("--input-csv", type=Path, default=fig_dir / "fig_cumulative_gain_data.csv")
    parser.add_argument("--output-figure", type=Path, default=fig_dir / "fig_cumulative_gain.pdf")
    return parser.parse_args()


def require_runtime():
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        missing = getattr(exc, "name", "a required package")
        raise SystemExit(f"Falta una dependencia: {missing}\nInstalar con:\n  pip install pandas matplotlib") from exc
    return {"pd": pd, "plt": plt}


def main() -> None:
    args = parse_args()
    rt = require_runtime()
    pd, plt = rt["pd"], rt["plt"]

    print("Figura B2 (panel C, RENDER-ONLY) - curva de ganancia acumulada a nivel de localidad")
    print("=========================================================================================")
    if not args.input_csv.exists():
        raise SystemExit(f"No se encontro {args.input_csv} -- correr "
                          "25b_cumulative_gain_locations_figure.py primero.")
    df = pd.read_csv(args.input_csv).sort_values("budget_locations")
    print(f"Entrada (ya calculada): {args.input_csv} -- {len(df)} puntos de malla "
          f"(presupuesto {int(df['budget_locations'].min())}-{int(df['budget_locations'].max())}).")

    plt.rcParams.update({
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
        'lines.linewidth': 1.0, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'savefig.dpi': 300, 'pdf.fonttype': 42,
    })

    budgets = df["budget_locations"].to_numpy()
    fig, ax = plt.subplots(figsize=(1.76, 1.95), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05)
    ax.plot(budgets, df["chance_diagonal"], color="black", linestyle=":", linewidth=1.2,
            label="Chance baseline")
    for method in METHODS:
        color = METHOD_COLORS[method]
        ax.fill_between(budgets, df[f"{method}_ci95_lo"], df[f"{method}_ci95_hi"],
                         color=color, alpha=0.20)
        ax.plot(budgets, df[f"{method}_mean"], color=color, linewidth=1.5, label=METHOD_LABELS[method])
    ax.set_xlim(0, float(budgets.max()))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Resurvey budget (localities)")
    ax.set_ylabel("Losses captured")
    ax.grid(alpha=0.25, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=1,
               handlelength=1.3, handletextpad=0.4, columnspacing=1.0, labelspacing=0.25,
               frameon=False)

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(args.output_figure, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(args.output_figure.with_suffix(".png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Guardado (figura): {args.output_figure}")
    print(f"Guardado (figura, png): {args.output_figure.with_suffix('.png')}")


if __name__ == "__main__":
    main()
