# CCAI OBSOLESCENCIA - O1g: figura de sensibilidad al buffer, paper/figures/fig_buffer_sensitivity.pdf
# -- las dos series (binario y continuo) contra el radio de buffer, cohorte comun de O1e (40.000
# localidades), con bandas IC95% de bootstrap de bloque espacial y el factor de rango anotado en
# cada serie. Lee results/record_level_histogram/buffer_sensitivity_common_cohort.csv y
# buffer_sensitivity_range_factor.csv (ya generados por 01f_buffer_sensitivity_common_cohort.py).
# NO CONSULTA EARTH ENGINE, NO ENTRENA NINGUN MODELO.
"""
POR QUE
-------
Panel A de fig:main en el paper. Las dos series difieren en un orden de magnitud (binario
1.7-54.9%, continuo 1.7-2.15%) -- graficarlas en el mismo eje Y aplanaria la serie continua contra
el 0. Se usan dos ejes Y (twin axis) sobre el mismo eje X (radio de buffer), cada uno con su propio
color y su propia banda IC95%, y el factor de rango de CADA serie anotado junto a su ultimo punto
(500 m) -- son los mismos numeros que 01f ya calculo y verifico con bootstrap PAREADO de bloque
espacial (buffer_sensitivity_range_factor.csv / buffer_sensitivity_range_factor_paired_log_bootstrap.csv),
esta figura no recalcula nada, solo grafica.

RENDER 2026-08-22, SOLO TAMAÑO/TIPOGRAFIA (sin tocar datos): figsize fijado a 1.76in (=0.32in x
textwidth=5.5in del paper) para que \includegraphics[width=\linewidth] en el minipage de 0.32\textwidth
no reescale la figura -- a ese ancho fisico las leyendas largas originales no entraban, se acortaron
("Any-loss flag (binary)" -> "Any-loss (binary)", "Converted fraction (continuous)" -> "Converted
fraction") sin cambiar su significado. Ningun numero ni dato cambia.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    obs_root = SCRIPT_DIR.parent
    hist_dir = obs_root / "results" / "record_level_histogram"
    parser = argparse.ArgumentParser(
        description="Figura O1 (panel A) - sensibilidad al buffer, cohorte comun, binario vs continuo."
    )
    parser.add_argument("--input-csv", type=Path, default=hist_dir / "buffer_sensitivity_common_cohort.csv")
    parser.add_argument("--range-factor-csv", type=Path, default=hist_dir / "buffer_sensitivity_range_factor.csv")
    parser.add_argument("--output-figure", type=Path, default=obs_root / "figures" / "fig_buffer_sensitivity.pdf")
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

    print("Figura O1 (panel A) - sensibilidad al buffer, binario vs continuo")
    print("======================================================================")
    df = pd.read_csv(args.input_csv).sort_values("buffer_m")
    rf = pd.read_csv(args.range_factor_csv)
    binary_rf = float(rf.loc[rf["serie"].str.startswith("binario"), "range_factor"].iloc[0])
    continuous_rf = float(rf.loc[rf["serie"].str.startswith("continuo"), "range_factor"].iloc[0])
    print(f"Factor de rango: binario={binary_rf:.1f}x, continuo={continuous_rf:.2f}x")

    plt.rcParams.update({
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
        'lines.linewidth': 1.0, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'savefig.dpi': 300, 'pdf.fonttype': 42,
    })

    x = df["buffer_m"].to_numpy()
    color_bin, color_cont = "#c44e52", "#4c72b0"

    fig, ax1 = plt.subplots(figsize=(1.76, 1.95), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05)
    ax1.fill_between(x, df["binary_ci95_lo"], df["binary_ci95_hi"], color=color_bin, alpha=0.20)
    ax1.plot(x, df["binary_pct"], "o-", color=color_bin, linewidth=1.4, markersize=3.2,
              label="Any-loss (binary)")
    ax1.set_xlabel("Buffer radius (m)")
    ax1.set_ylabel("Any-loss flag (%)", color=color_bin)
    ax1.tick_params(axis="y", labelcolor=color_bin)
    ax1.set_xticks(x)
    ax1.set_ylim(0, max(df["binary_ci95_hi"]) * 1.12)
    ax1.annotate(f"{binary_rf:.1f}$\\times$", xy=(x[-1], df["binary_pct"].iloc[-1]),
                 xytext=(-6, 8), textcoords="offset points", color=color_bin, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.fill_between(x, df["continuous_ci95_lo"], df["continuous_ci95_hi"], color=color_cont, alpha=0.20)
    ax2.plot(x, df["continuous_pct"], "s-", color=color_cont, linewidth=1.4, markersize=3.2,
              label="Converted fraction")
    ax2.set_ylabel("Converted fraction (%)", color=color_cont)
    ax2.tick_params(axis="y", labelcolor=color_cont)
    ax2.set_ylim(0, max(df["continuous_ci95_hi"]) * 1.12)
    ax2.annotate(f"{continuous_rf:.2f}$\\times$", xy=(x[-1], df["continuous_pct"].iloc[-1]),
                 xytext=(-6, -12), textcoords="offset points", color=color_cont, fontweight="bold")

    ax1.grid(alpha=0.2, linewidth=0.5)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc="outside lower center", ncol=1,
               handlelength=1.3, handletextpad=0.4, columnspacing=1.0, labelspacing=0.25,
               frameon=False)

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(args.output_figure, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(args.output_figure.with_suffix(".png"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"Guardado (figura): {args.output_figure}")
    print(f"Guardado (figura, png): {args.output_figure.with_suffix('.png')}")


if __name__ == "__main__":
    main()
