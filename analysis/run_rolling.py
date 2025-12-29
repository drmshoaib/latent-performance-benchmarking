import pandas as pd
from analysis.rolling_windows import rolling_sfa

INFILE = "results/merged_for_sfa.csv"
OUTFILE = "results/ae_rolling.csv"

df = pd.read_csv(INFILE, parse_dates=["date"])

rolling = rolling_sfa(
    df,
    window=120,
    min_obs=120,
    step=12,   # yearly rolling
)

rolling.to_csv(OUTFILE, index=False)

print("Rolling-window results written to:", OUTFILE)
print("Rows:", len(rolling))
print("Window range:", rolling["window_end"].min(), "→", rolling["window_end"].max())
