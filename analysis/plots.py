# -------------------------------
# 1. Rank persistence plot
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/rank_persistence.csv")

summary = (
    df.groupby("horizon")["spearman_rho"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(6, 4))
plt.plot(summary["horizon"], summary["spearman_rho"], marker="o")
plt.axhline(0, color="black", linewidth=0.6)
plt.xlabel("Horizon (months)")
plt.ylabel("Mean Spearman Rank Correlation")
plt.title("Rank Persistence of Latent Performance")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig_rank_persistence.png", dpi=300)
plt.close()


import seaborn as sns
import numpy as np

mat = pd.read_csv(
    "results/quintile_transitions_h1.csv",
    index_col=0
)

plt.figure(figsize=(6,5))
sns.heatmap(
    mat,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar_kws={"label": "Transition Probability"}
)
plt.xlabel("To Quintile")
plt.ylabel("From Quintile")
plt.title("Quintile Transition Matrix (H1)")
plt.tight_layout()
plt.savefig("results/fig_quintile_transitions.png", dpi=300)
plt.close()

mob = pd.read_csv("results/quintile_mobility_h1.csv")

mob.set_index("Quintile")[["Stay", "Improve", "Deteriorate"]].plot(
    kind="bar",
    stacked=True,
    figsize=(7,4)
)

plt.ylabel("Probability")
plt.title("Mobility of Latent Performance Quintiles")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("results/fig_mobility.png", dpi=300)
plt.close()
