from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis.alpha_baseline import compare_alpha_to_ae, estimate_alpha_baseline
from analysis.diagnostics import model_diagnostics
from analysis.figures import generate_all_figures
from analysis.mobility import portfolio_mobility_summary
from analysis.persistence import compute_persistence_metrics
from analysis.robustness import model_comparison, rolling_window_sensitivity
from analysis.rolling_windows import rolling_sfa
from analysis.static_sfa import estimate_static_sfa
from analysis.transitions import compute_transition_matrix
from sfa.loaders import build_factor_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
PORT_FILE = DATA_DIR / "25_size_bm_portfolios.csv"
FF_FILE = DATA_DIR / "ff3_factors.csv"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a reproducible pipeline run."""

    port_file: Path = PORT_FILE
    factor_file: Path = FF_FILE
    results_dir: Path = RESULTS_DIR
    factor_model: str = "ff3"
    sfa_model: str = "half_normal"
    rolling_window: int = 120
    rolling_step: int = 12
    min_obs: int = 120
    rolling_maxiter: int = 80
    static_maxiter: int = 500
    persistence_horizons: tuple[int, ...] = (1, 3, 6, 12)
    transition_horizon: int = 12
    robustness_windows: tuple[int, ...] = (60, 120, 180)
    robustness_step: int = 120

    @property
    def tables_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"


def _write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write a CSV file, creating the parent directory if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the research pipeline."""

    parser = argparse.ArgumentParser(
        description="Run the latent performance benchmarking research pipeline."
    )
    parser.add_argument("--portfolio-file", type=Path, default=PORT_FILE)
    parser.add_argument("--factor-file", type=Path, default=FF_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--factor-model",
        default="ff3",
        choices=["capm", "ff3", "ff5", "ff5_mom"],
    )
    parser.add_argument(
        "--sfa-model",
        default="half_normal",
        choices=["half_normal", "truncated_normal"],
    )
    parser.add_argument("--rolling-window", type=int, default=120)
    parser.add_argument("--rolling-step", type=int, default=12)
    parser.add_argument("--min-obs", type=int, default=120)
    parser.add_argument("--rolling-maxiter", type=int, default=80)
    parser.add_argument("--static-maxiter", type=int, default=500)
    parser.add_argument(
        "--persistence-horizons",
        type=int,
        nargs="+",
        default=[1, 3, 6, 12],
    )
    parser.add_argument(
        "--transition-horizon",
        type=int,
        default=12,
        help="Transition horizon in months.",
    )
    parser.add_argument(
        "--robustness-windows",
        type=int,
        nargs="+",
        default=[60, 120, 180],
    )
    parser.add_argument("--robustness-step", type=int, default=120)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Build a pipeline configuration from parsed CLI arguments."""

    return PipelineConfig(
        port_file=args.portfolio_file,
        factor_file=args.factor_file,
        results_dir=args.results_dir,
        factor_model=args.factor_model,
        sfa_model=args.sfa_model,
        rolling_window=args.rolling_window,
        rolling_step=args.rolling_step,
        min_obs=args.min_obs,
        rolling_maxiter=args.rolling_maxiter,
        static_maxiter=args.static_maxiter,
        persistence_horizons=tuple(args.persistence_horizons),
        transition_horizon=args.transition_horizon,
        robustness_windows=tuple(args.robustness_windows),
        robustness_step=args.robustness_step,
    )


def run_pipeline(config: PipelineConfig) -> dict:
    """Run the full analytical pipeline and return a compact summary."""

    started = time.perf_counter()
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading and aligning factor/portfolio data")

    dataset = build_factor_dataset(
        config.port_file,
        config.factor_file,
        factor_model=config.factor_model,
    )
    df = dataset.data
    _write_csv(df, config.tables_dir / "factor_model_dataset.csv")
    _write_csv(
        pd.DataFrame([dataset.summary()]),
        config.tables_dir / "factor_dataset_summary.csv",
    )

    LOGGER.info("Estimating static %s SFA", config.sfa_model)
    static_scores, static_timeseries = estimate_static_sfa(
        df,
        dataset.factor_cols,
        factor_model=dataset.factor_model,
        model_type=config.sfa_model,
        min_obs=config.min_obs,
        maxiter=config.static_maxiter,
    )
    _write_csv(static_scores, config.tables_dir / "static_efficiency_scores.csv")
    _write_csv(
        static_timeseries, config.tables_dir / "static_efficiency_timeseries.csv"
    )

    LOGGER.info("Estimating alpha baseline")
    alpha = estimate_alpha_baseline(
        df,
        dataset.factor_cols,
        factor_model=dataset.factor_model,
        min_obs=config.min_obs,
    )
    _write_csv(alpha, config.tables_dir / "alpha_baseline.csv")

    alpha_comparison = compare_alpha_to_ae(alpha, static_scores)
    _write_csv(alpha_comparison, config.tables_dir / "alpha_vs_ae_comparison.csv")

    LOGGER.info(
        "Estimating rolling SFA: window=%s, step=%s",
        config.rolling_window,
        config.rolling_step,
    )
    rolling = rolling_sfa(
        df,
        dataset.factor_cols,
        window=config.rolling_window,
        min_obs=config.min_obs,
        step=config.rolling_step,
        model_type=config.sfa_model,
        factor_model=dataset.factor_model,
        maxiter=config.rolling_maxiter,
    )
    _write_csv(rolling, config.tables_dir / "rolling_efficiency_scores.csv")

    LOGGER.info("Computing persistence, transitions, and mobility")
    persistence = compute_persistence_metrics(
        rolling,
        horizons_months=config.persistence_horizons,
    )
    _write_csv(persistence, config.tables_dir / "rank_persistence.csv")

    transition_matrix, transition_summary = compute_transition_matrix(
        rolling,
        horizon_months=config.transition_horizon,
    )
    _write_csv(
        transition_matrix, config.tables_dir / "transition_matrix.csv", index=True
    )
    _write_csv(transition_summary, config.tables_dir / "transition_summary.csv")

    mobility = portfolio_mobility_summary(rolling)
    _write_csv(mobility, config.tables_dir / "mobility_summary.csv")

    LOGGER.info("Running rolling-window sensitivity checks")
    robustness_summary, robustness_scores = rolling_window_sensitivity(
        df,
        dataset.factor_cols,
        factor_model=dataset.factor_model,
        model_type=config.sfa_model,
        windows=config.robustness_windows,
        step=config.robustness_step,
        maxiter=config.rolling_maxiter,
    )
    _write_csv(robustness_summary, config.tables_dir / "robustness_summary.csv")
    _write_csv(
        robustness_scores,
        config.tables_dir / "rolling_window_sensitivity_scores.csv",
    )

    LOGGER.info("Estimating truncated-normal static comparison model")
    truncated_scores, _ = estimate_static_sfa(
        df,
        dataset.factor_cols,
        factor_model=dataset.factor_model,
        model_type="truncated_normal",
        min_obs=config.min_obs,
        maxiter=config.static_maxiter,
    )
    comparison = model_comparison(static_scores, truncated_scores)
    _write_csv(comparison, config.tables_dir / "model_comparison.csv")

    elapsed = time.perf_counter() - started
    diagnostics = model_diagnostics(
        static_scores,
        static_timeseries,
        runtime_seconds=elapsed,
    )
    _write_csv(diagnostics, config.tables_dir / "model_diagnostics.csv")

    LOGGER.info("Generating figures")
    generate_all_figures(
        static_scores=static_scores,
        rolling=rolling,
        persistence=persistence,
        transition_matrix=transition_matrix,
        mobility=mobility,
        alpha_comparison=alpha_comparison,
        robustness=robustness_summary,
        residuals=static_timeseries,
        output_dir=config.figures_dir,
    )

    n_windows = int(rolling[["portfolio", "window_end"]].drop_duplicates().shape[0])
    summary = {
        "n_portfolios": dataset.n_portfolios,
        "sample_start": dataset.sample_start.date().isoformat(),
        "sample_end": dataset.sample_end.date().isoformat(),
        "factor_model": dataset.factor_model,
        "factor_cols": ", ".join(dataset.factor_cols),
        "sfa_model": config.sfa_model,
        "rolling_window": config.rolling_window,
        "rolling_step": config.rolling_step,
        "n_rolling_estimates": n_windows,
        "tables_dir": str(config.tables_dir),
        "figures_dir": str(config.figures_dir),
        "runtime_seconds": elapsed,
    }
    LOGGER.info("Pipeline complete in %.2f seconds", elapsed)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for the full research pipeline."""

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    summary = run_pipeline(config_from_args(args))

    print("\nLatent Performance Benchmarking pipeline complete")
    print("------------------------------------------------")
    print(f"Portfolios: {summary['n_portfolios']}")
    print(f"Sample period: {summary['sample_start']} to {summary['sample_end']}")
    print(f"Factor model: {summary['factor_model']} ({summary['factor_cols']})")
    print(f"SFA model: {summary['sfa_model']}")
    print(f"Rolling window: {summary['rolling_window']} months")
    print(f"Rolling step: {summary['rolling_step']} month(s)")
    print(f"Rolling portfolio-window estimates: {summary['n_rolling_estimates']}")
    print(f"Tables: {summary['tables_dir']}")
    print(f"Figures: {summary['figures_dir']}")
    print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
