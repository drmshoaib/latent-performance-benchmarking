from __future__ import annotations

import tempfile
import time
from pathlib import Path

from analysis.figures import static_ae_ranking
from analysis.rolling_windows import rolling_sfa
from analysis.static_sfa import estimate_static_sfa
from sfa.loaders import build_factor_dataset

ROOT = Path(__file__).resolve().parents[1]
PORT_FILE = ROOT / "data" / "25_size_bm_portfolios.csv"
FACTOR_FILE = ROOT / "data" / "ff3_factors.csv"


def _time_call(label: str, func):
    started = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - started
    return label, elapsed, result


def main() -> None:
    """Run a compact local benchmark for core analytical routines."""

    dataset = build_factor_dataset(PORT_FILE, FACTOR_FILE, factor_model="ff3")
    subset_names = sorted(dataset.data["portfolio"].unique())[:5]
    subset = dataset.data[dataset.data["portfolio"].isin(subset_names)].copy()

    rows = []

    label, elapsed, static_result = _time_call(
        "static_sfa_5_portfolios",
        lambda: estimate_static_sfa(
            subset,
            dataset.factor_cols,
            factor_model=dataset.factor_model,
            model_type="half_normal",
            min_obs=120,
            maxiter=120,
        ),
    )
    static_scores, _ = static_result
    rows.append((label, elapsed, len(static_scores)))

    label, elapsed, rolling = _time_call(
        "rolling_sfa_5_portfolios",
        lambda: rolling_sfa(
            subset,
            dataset.factor_cols,
            window=120,
            min_obs=120,
            step=24,
            factor_model=dataset.factor_model,
            maxiter=60,
        ),
    )
    rows.append((label, elapsed, len(rolling)))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        label, elapsed, _ = _time_call(
            "static_figure_generation",
            lambda: static_ae_ranking(static_scores, out),
        )
        rows.append((label, elapsed, int((out / "static_ae_ranking.png").exists())))

    print("Benchmark")
    print("---------")
    print(f"{'task':<30} {'seconds':>10} {'rows/files':>12}")
    for label, elapsed, count in rows:
        print(f"{label:<30} {elapsed:>10.2f} {count:>12}")


if __name__ == "__main__":
    main()
