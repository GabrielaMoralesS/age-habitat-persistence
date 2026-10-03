# CCAI OBSOLESCENCIA - R2 auxiliar (RENDER-ONLY): re-dibuja paper/figures/fig_survival_curves.pdf
# al tamaño físico exacto del panel B de fig:main (0.32\textwidth = 1.76in), leyendo UNICAMENTE
# paper/figures/fig_survival_curves_data.csv -- la malla 0-40 años con la curva KM fuera-de-fold
# (media + banda 95%) y las dos curvas Rayleigh que 22b_survival_curves_figure.py YA CALCULO Y
# GUARDO (100 curvas KM fuera-de-fold, bootstrap de bloque espacial x draw, ver el bloque POR QUE
# de ese script). NO CONSULTA EARTH ENGINE, NO ENTRENA NINGUN MODELO, NO RECALCULA NADA: no importa
# GroupKFold/kaplan_meier_estimator/sklearn/sksurv/geopandas, solo pandas+matplotlib sobre el CSV
# ya persistido.
"""
POR QUE
-------
22b_survival_curves_figure.py recalcula 100 curvas KM fuera-de-fold (20 draws x 5 folds espaciales)
solo para producir seis numeros por punto de malla que YA GUARDA en fig_survival_curves_data.csv
(km_out_of_fold_mean/ci95_lo/ci95_hi, rayleigh_sigma30, rayleigh_sigma_hat) -- recorrer ese calculo
de nuevo solo para cambiar tamaño de fuente y figsize es trabajo (y riesgo de divergencia numerica)
innecesario. Este script es la mitad "solo grafica" de 22b, separada para que regenerar la figura al
tamaño de impresion final no dependa de repetir el bootstrap.

TAMAÑO FISICO: panel B de fig:main se inserta a 0.32\\textwidth = 0.32*5.5in = 1.76in
(\\includegraphics[width=\\linewidth] dentro de un minipage de ese ancho) -- figsize se fija a ese
ancho exacto para que LaTeX no reescale la figura al compilar.

LEYENDA ACORTADA (mismo significado, sin tocar ningun numero): la banda de confianza de la curva
KM fuera-de-fold pasa de "Out-of-fold KM, 95% CI (spatial block x draw)" a "Out-of-fold KM (95% CI)"
(la mecanica del bootstrap de bloque espacial x draw ya esta en el pie de figura del paper, no hace
falta repetirla en la leyenda); "Rayleigh sigma=30 (literature)" pasa a "Rayleigh sigma=30" y
"Rayleigh sigma-hat=54.27 (estimated)" pasa a "Rayleigh sigma-hat=54.3" -- a 1.76in de ancho con
fuente de leyenda de 6pt las versiones largas no entraban dentro de los ejes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    obs_root = SCRIPT_DIR.parent
    fig_dir = obs_root / "figures"
    parser = argparse.ArgumentParser(
        description="Figura R2 (panel B, RENDER-ONLY) - curvas de supervivencia 0-40 años, desde "
        "fig_survival_curves_data.csv ya calculado por 22b_survival_curves_figure.py."
    )
    parser.add_argument("--input-csv", type=Path, default=fig_dir / "fig_survival_curves_data.csv")
    parser.add_argument("--output-figure", type=Path, default=fig_dir / "fig_survival_curves.pdf")
    parser.add_argument("--sigma-hat-label", type=float, default=54.3,
                         help="Valor mostrado en la leyenda (redondeo de 54.27, no recalculado).")
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

    print("Figura R2 (panel B, RENDER-ONLY) - curvas de supervivencia 0-40 años")
    print("=========================================================================")
    if not args.input_csv.exists():
        raise SystemExit(f"No se encontro {args.input_csv} -- correr 22b_survival_curves_figure.py primero.")
    df = pd.read_csv(args.input_csv).sort_values("t_years")
    print(f"Entrada (ya calculada): {args.input_csv} -- {len(df)} puntos de malla.")

    plt.rcParams.update({
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
        'lines.linewidth': 1.0, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'savefig.dpi': 300, 'pdf.fonttype': 42,
    })

    t = df["t_years"].to_numpy()
    fig, ax = plt.subplots(figsize=(1.76, 1.95), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05)
    ax.fill_between(t, df["km_out_of_fold_ci95_lo"], df["km_out_of_fold_ci95_hi"],
                     color="#4c72b0", alpha=0.22, label="Out-of-fold KM (95% CI)")
    ax.plot(t, df["km_out_of_fold_mean"], color="#4c72b0", linewidth=1.5, label="Out-of-fold KM (mean)")
    ax.plot(t, df["rayleigh_sigma30"], color="#c44e52", linestyle="--", linewidth=1.4,
            label=r"Rayleigh $\sigma=30$")
    ax.plot(t, df["rayleigh_sigma_hat"], color="#55a868", linestyle="-.", linewidth=1.4,
            label=rf"Rayleigh $\hat\sigma={args.sigma_hat_label:.1f}$")
    ax.set_xlim(0, float(t.max()))
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Record age (years)")
    ax.set_ylabel(r"Forest support $S(t)$")
    ax.grid(alpha=0.25, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=1,
               handlelength=1.3, handletextpad=0.4, columnspacing=1.0, labelspacing=0.25,
               frameon=False)

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(args.output_figure, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(args.output_figure.with_suffix(".png"), bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Guardado (figura): {args.output_figure}")
    print(f"Guardado (figura, png): {args.output_figure.with_suffix('.png')}")


if __name__ == "__main__":
    main()
