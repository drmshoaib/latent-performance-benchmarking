import pandas as pd

from analysis.persistence import rank_persistence

INFILE = "results/ae_rolling.csv"
OUTFILE = "results/rank_persistence.csv"

df = pd.read_csv(INFILE, parse_dates=["window_end"])

# 1-year persistence (adjacent rolling windows)
p1 = rank_persistence(df, horizon=1)

# 2-year persistence
p2 = rank_persistence(df, horizon=2)

out = pd.concat([p1, p2], ignore_index=True)
out.to_csv(OUTFILE, index=False)

print("Rank persistence written to:", OUTFILE)
print(out.groupby("horizon")["spearman_rho"].describe())
