from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.figures import generate_all_figures

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"


def main() -> None:
    generate_all_figures(
        static_scores=pd.read_csv(TABLES / "static_efficiency_scores.csv"),
        rolling=pd.read_csv(TABLES / "rolling_efficiency_scores.csv"),
        persistence=pd.read_csv(TABLES / "rank_persistence.csv"),
        transition_matrix=pd.read_csv(TABLES / "transition_matrix.csv", index_col=0),
        mobility=pd.read_csv(TABLES / "mobility_summary.csv"),
        alpha_comparison=pd.read_csv(TABLES / "alpha_vs_ae_comparison.csv"),
        robustness=pd.read_csv(TABLES / "robustness_summary.csv"),
        residuals=pd.read_csv(TABLES / "static_efficiency_timeseries.csv"),
        output_dir=FIGURES,
    )
    print(f"Figures written to: {FIGURES}")


if __name__ == "__main__":
    main()
