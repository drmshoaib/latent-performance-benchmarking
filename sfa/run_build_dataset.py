from pathlib import Path

from sfa.loaders import build_merged_dataset

PORT_FILE = Path("data/25_size_bm_portfolios.csv")
FF3_FILE = Path("data/ff3_factors.csv")
OUTFILE = Path("results/tables/factor_model_dataset.csv")


def main() -> None:
    """Build the cleaned factor-model dataset used by the analysis pipeline."""

    df = build_merged_dataset(PORT_FILE, FF3_FILE)
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTFILE, index=False)

    print("Merged SFA dataset written to:", OUTFILE.resolve())
    print("Rows:", len(df))
    print("Date range:", df["date"].min(), "to", df["date"].max())


if __name__ == "__main__":
    main()
