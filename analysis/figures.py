from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def _portfolio_grid_position(name: str) -> tuple[int | None, int | None]:
    size = None
    bm = None
    if name.startswith("SMALL") or name.startswith("ME1"):
        size = 1
    elif name.startswith("ME2"):
        size = 2
    elif name.startswith("ME3"):
        size = 3
    elif name.startswith("ME4"):
        size = 4
    elif name.startswith("BIG") or name.startswith("ME5"):
        size = 5

    if "LoBM" in name or "BM1" in name:
        bm = 1
    elif "BM2" in name:
        bm = 2
    elif "BM3" in name:
        bm = 3
    elif "BM4" in name:
        bm = 4
    elif "HiBM" in name or "BM5" in name:
        bm = 5
    return size, bm


def static_ae_ranking(static_scores: pd.DataFrame, out: Path) -> None:
    """Plot static adjusted-efficiency scores by portfolio."""

    data = static_scores.sort_values("AE", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(data["portfolio"], data["AE"], color="#2f6f9f")
    ax.set_xlabel("Adjusted efficiency (AE)")
    ax.set_title("Static SFA Adjusted Efficiency Ranking")
    ax.set_xlim(
        max(0.0, data["AE"].min() - 0.002), min(1.001, data["AE"].max() + 0.001)
    )
    fig.tight_layout()
    fig.savefig(out / "static_ae_ranking.png")
    plt.close(fig)


def ae_heatmap(static_scores: pd.DataFrame, out: Path) -> None:
    """Plot static AE values on the 5x5 size/book-to-market grid."""

    records = []
    for _, row in static_scores.iterrows():
        size, bm = _portfolio_grid_position(str(row["portfolio"]))
        if size is not None and bm is not None:
            records.append({"size": size, "book_to_market": bm, "AE": row["AE"]})
    heat = pd.DataFrame(records).pivot(
        index="size", columns="book_to_market", values="AE"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        heat, annot=True, fmt=".4f", cmap="viridis", cbar_kws={"label": "AE"}, ax=ax
    )
    ax.set_xlabel("Book-to-market quintile")
    ax.set_ylabel("Size quintile")
    ax.set_title("Static AE Across Size x Book-to-Market Portfolios")
    fig.tight_layout()
    fig.savefig(out / "ae_heatmap_size_bm.png")
    plt.close(fig)


def rolling_ae_timeseries(rolling: pd.DataFrame, out: Path) -> None:
    """Plot cross-sectional rolling AE mean and dispersion."""

    df = rolling.dropna(subset=["AE"]).copy()
    df["window_end"] = pd.to_datetime(df["window_end"])
    summary = (
        df.groupby("window_end")["AE"]
        .agg(
            mean="mean", p10=lambda x: x.quantile(0.10), p90=lambda x: x.quantile(0.90)
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = summary["window_end"].to_numpy()
    ax.plot(x, summary["mean"], color="#1f4e79", label="Cross-sectional mean")
    ax.fill_between(
        x,
        summary["p10"].to_numpy(float),
        summary["p90"].to_numpy(float),
        color="#6aaed6",
        alpha=0.25,
        label="10th-90th percentile",
    )
    ax.set_xlabel("Window end")
    ax.set_ylabel("Adjusted efficiency (AE)")
    ax.set_title("Rolling SFA Adjusted Efficiency")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "rolling_ae_timeseries.png")
    plt.close(fig)


def rank_persistence_plot(persistence: pd.DataFrame, out: Path) -> None:
    """Plot rolling rank and AE persistence by horizon."""

    summary = (
        persistence.groupby("horizon_months")
        .agg(
            spearman=("spearman_rank_autocorrelation", "mean"),
            pearson=("pearson_ae_autocorrelation", "mean"),
            rank_change=("average_absolute_rank_change", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        summary["horizon_months"],
        summary["spearman"],
        marker="o",
        label="Spearman rank",
    )
    ax.plot(
        summary["horizon_months"], summary["pearson"], marker="s", label="Pearson AE"
    )
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Autocorrelation")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Rolling Persistence Diagnostics")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "rank_persistence.png")
    plt.close(fig)


def transition_heatmap(matrix: pd.DataFrame, out: Path) -> None:
    """Plot the quintile transition matrix as a heatmap."""

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Probability"},
        ax=ax,
    )
    ax.set_xlabel("To quintile")
    ax.set_ylabel("From quintile")
    ax.set_title("Quintile Transition Matrix")
    fig.tight_layout()
    fig.savefig(out / "transition_matrix_heatmap.png")
    plt.close(fig)


def mobility_plot(mobility: pd.DataFrame, out: Path) -> None:
    """Plot portfolios with the highest rolling rank volatility."""

    data = mobility.sort_values("rank_volatility", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(data["portfolio"], data["rank_volatility"], color="#7c4d79")
    ax.set_ylabel("Rank volatility")
    ax.set_title("Most Mobile Portfolios by Rolling Rank Volatility")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out / "mobility_summary.png")
    plt.close(fig)


def alpha_vs_ae_scatter(comparison: pd.DataFrame, out: Path) -> None:
    """Plot traditional alpha ranks against SFA AE ranks."""

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(comparison["alpha_rank"], comparison["AE_rank"], color="#2a9d8f", s=55)
    lim = [
        min(comparison["alpha_rank"].min(), comparison["AE_rank"].min()) - 1,
        max(comparison["alpha_rank"].max(), comparison["AE_rank"].max()) + 1,
    ]
    ax.plot(lim, lim, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Alpha rank")
    ax.set_ylabel("AE rank")
    ax.set_title("Alpha Rank vs SFA AE Rank")
    fig.tight_layout()
    fig.savefig(out / "alpha_vs_ae_rank_scatter.png")
    plt.close(fig)


def window_sensitivity_plot(robustness: pd.DataFrame, out: Path) -> None:
    """Plot rolling-window sensitivity correlations."""

    data = robustness.copy()
    data["comparison"] = (
        data["window_left"].astype(str) + " vs " + data["window_right"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(data))
    width = 0.35
    ax.bar(x - width / 2, data["rank_correlation"], width, label="Rank")
    ax.bar(x + width / 2, data["AE_correlation"], width, label="AE")
    ax.set_xticks(x)
    ax.set_xticklabels(data["comparison"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Correlation")
    ax.set_title("Rolling Window Sensitivity")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "window_sensitivity.png")
    plt.close(fig)


def residual_diagnostics_plot(residuals: pd.DataFrame, out: Path) -> None:
    """Plot a residual histogram for static SFA fits."""

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(
        residuals["residual"].dropna(), bins=50, kde=True, color="#4c78a8", ax=ax
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Residual")
    ax.set_title("Static SFA Residual Diagnostics")
    fig.tight_layout()
    fig.savefig(out / "residual_diagnostics.png")
    plt.close(fig)


def generate_all_figures(
    *,
    static_scores: pd.DataFrame,
    rolling: pd.DataFrame,
    persistence: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    mobility: pd.DataFrame,
    alpha_comparison: pd.DataFrame,
    robustness: pd.DataFrame,
    residuals: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate the standard set of GitHub-readable result figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _set_style()
    static_ae_ranking(static_scores, output_dir)
    ae_heatmap(static_scores, output_dir)
    rolling_ae_timeseries(rolling, output_dir)
    rank_persistence_plot(persistence, output_dir)
    transition_heatmap(transition_matrix, output_dir)
    mobility_plot(mobility, output_dir)
    alpha_vs_ae_scatter(alpha_comparison, output_dir)
    window_sensitivity_plot(robustness, output_dir)
    residual_diagnostics_plot(residuals, output_dir)
