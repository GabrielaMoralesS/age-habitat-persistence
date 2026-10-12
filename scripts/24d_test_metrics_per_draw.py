# Table 1 metrics of the paper: AUROC/AUPRC on the frozen 2014->2019 test, computed WITHIN each
# positional draw and then averaged over draws (the definition stated in the Table 1 caption).
#
# Why this script exists: 24_landmark_forecast.py writes test_landmark_2014_metrics.csv with metrics
# computed once over the POOLED rows of all 20 draws. For AUROC the two aggregations agree to the third
# decimal; for AUPRC they do not (e.g. LightGBM 0.0371 pooled vs. 0.0390 mean over draws), because
# average precision is not linear in the rows. The paper reports the mean over draws, consistent with
# "metrics are computed within each draw and then averaged" in Methods. Paired bootstrap differences
# (test_landmark_2014_bootstrap_diffs.csv) are computed on pooled rows.
#
# Input : results/landmark/test_landmark_2014_predictions.csv (written by 24_landmark_forecast.py)
# Output: results/landmark/test_landmark_2014_metrics_per_draw.csv
from __future__ import annotations

import argparse
from pathlib import Path

METHODS = [
    "aleatorio",
    "mas_antiguo",
    "rayleigh_30",
    "rayleigh_estimado",
    "perdida_reciente",
    "landmark_logistic",
    "landmark_boosting",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    landmark_dir = root / "results" / "landmark"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path,
                        default=landmark_dir / "test_landmark_2014_predictions.csv")
    parser.add_argument("--output-csv", type=Path,
                        default=landmark_dir / "test_landmark_2014_metrics_per_draw.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    pred = pd.read_csv(args.predictions_csv)
    rows = []
    for method in METHODS:
        col = f"score_{method}"
        if col not in pred.columns:
            continue
        per_draw_auroc, per_draw_auprc = [], []
        for _, g in pred.groupby("draw_id"):
            if g["Z"].nunique() < 2:
                continue  # a draw without both classes has no defined AUROC/AUPRC
            per_draw_auroc.append(roc_auc_score(g["Z"], g[col]))
            per_draw_auprc.append(average_precision_score(g["Z"], g[col]))
        rows.append({
            "metodo": method,
            "auroc_mean_over_draws": float(np.mean(per_draw_auroc)),
            "auprc_mean_over_draws": float(np.mean(per_draw_auprc)),
            "auroc_pooled": float(roc_auc_score(pred["Z"], pred[col])),
            "auprc_pooled": float(average_precision_score(pred["Z"], pred[col])),
            "n_draws": len(per_draw_auroc),
        })
    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.round(4).to_string(index=False))
    print(f"\nWritten: {args.output_csv}")


if __name__ == "__main__":
    main()
