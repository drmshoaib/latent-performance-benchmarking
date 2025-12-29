from pathlib import Path
from sfa.loaders import build_merged_dataset

PORT_FILE = Path("data/25_size_bm_portfolios.csv")
FF3_FILE = Path("data/ff3_factors.csv")
OUTFILE = Path("results/merged_for_sfa.csv")

df = build_merged_dataset(PORT_FILE, FF3_FILE)
df.to_csv(OUTFILE, index=False)

print("Merged SFA dataset written to:", OUTFILE.resolve())
print("Rows:", len(df))
print("Date range:", df["date"].min(), "→", df["date"].max())
