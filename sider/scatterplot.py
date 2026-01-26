import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Load CSV
# -----------------------------
df = pd.read_csv("merged.csv")

# -----------------------------
# 2. Extract H2
# -----------------------------
h2 = df["H2(ppm)"]

# -----------------------------
# 3. Extract probability columns
# -----------------------------
prob_cols = [c for c in df.columns if c.startswith("Prob_")]
df_probs = df[prob_cols].copy()

def extract_mineral(col):
    return col.split("_")[2]

df_probs.columns = [extract_mineral(c) for c in df_probs.columns]

# -----------------------------
# 4. Merge duplicate minerals
# -----------------------------
df_minerals = (
    df_probs
    .T
    .groupby(level=0)
    .max()
    .T
)

# -----------------------------
# 5. Minerals of interest
# -----------------------------
minerals = [
    "Olivine",
    "Serpentine",
    "Cummingtonite",
    "Brucite"
]

# -----------------------------
# 6. Scatter + best-fit line
# -----------------------------
for mineral in minerals:
    if mineral not in df_minerals.columns:
        print(f"{mineral} not found")
        continue

    x = h2.values
    y = df_minerals[mineral].values

    # Remove NaNs
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]

    # Linear regression
    m, b = np.polyfit(x, y, 1)
    y_fit = m * x + b

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.6)
    plt.plot(x, y_fit)
    plt.xlabel("H₂ (ppm)")
    plt.ylabel(f"{mineral} Probability")
    plt.title(f"{mineral} Probability vs H₂ (Best Fit)")
    plt.tight_layout()
    plt.savefig(f"{mineral}.png")

    print(f"{mineral}: slope = {m:.4e}")