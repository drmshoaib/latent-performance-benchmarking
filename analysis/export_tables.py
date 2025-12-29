from pathlib import Path
import pandas as pd

RESULTS = Path("results")
OUT = Path("latex_tables")
OUT.mkdir(exist_ok=True)


def df_to_latex(
    df: pd.DataFrame,
    filename: str,
    caption: str,
    label: str,
    float_format="%.3f",
):
    """
    Write a DataFrame to a standalone LaTeX table.
    """
    tex = df.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
        escape=False,
        column_format="l" + "c" * (len(df.columns) - 1),
    )

    path = OUT / filename
    path.write_text(tex)
    print(f"Wrote {path}")


# -------------------------------------------------
# 1) Static cross-sectional rankings
# -------------------------------------------------
df_rank = pd.read_csv(RESULTS / "ae_rankings.csv")

df_rank = df_rank[
    ["portfolio", "AE_H1_mean", "rank_AE_H1"]
].rename(
    columns={
        "portfolio": "Portfolio",
        "AE_H1_mean": "Mean AE",
        "rank_AE_H1": "Rank",
    }
)

df_to_latex(
    df_rank,
    "table_static_rankings.tex",
    caption="Static Cross-Sectional Performance Rankings",
    label="tab:static_rankings",
)


# -------------------------------------------------
# 2) Rank persistence summary
# -------------------------------------------------
df_persist = pd.read_csv(RESULTS / "rank_persistence.csv")

df_persist = df_persist.reset_index().rename(
    columns={
        "horizon": "Horizon",
        "mean": "Mean $\\rho$",
        "std": "Std",
        "min": "Min",
        "max": "Max",
    }
)

df_to_latex(
    df_persist,
    "table_rank_persistence.tex",
    caption="Rank Persistence Across Rolling Windows",
    label="tab:rank_persistence",
)


# -------------------------------------------------
# 3) Quintile transition matrix
# -------------------------------------------------
df_trans = pd.read_csv(RESULTS / "quintile_transitions_h1.csv", index_col=0)

df_trans = df_trans.reset_index().rename(columns={"From Quintile": "From"})

df_to_latex(
    df_trans,
    "table_quintile_transitions.tex",
    caption="Quintile Transition Probabilities",
    label="tab:quintile_transitions",
)


# -------------------------------------------------
# 4) Mobility metrics
# -------------------------------------------------
df_mob = pd.read_csv(RESULTS / "quintile_mobility_h1.csv")

df_mob = df_mob.rename(
    columns={
        "Quintile": "Quintile",
        "Stay": "Stay",
        "Improve": "Improve",
        "Deteriorate": "Deteriorate",
    }
)

df_to_latex(
    df_mob,
    "table_quintile_mobility.tex",
    caption="Quintile Mobility Metrics",
    label="tab:quintile_mobility",
)
