import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load CSV
# -----------------------------
df = pd.read_csv("merged.csv")

# -----------------------------
# 2. Select probability columns
# -----------------------------
prob_cols = [c for c in df.columns if c.startswith("Prob_")]
df_probs = df[prob_cols].copy()

# -----------------------------
# 3. Extract mineral names
# -----------------------------
def extract_mineral(col):
    return col.split("_")[2]

df_probs.columns = [extract_mineral(c) for c in df_probs.columns]

# -----------------------------
# 4. Merge duplicate minerals (max per row)
# -----------------------------
df_minerals = (
    df_probs
    .T
    .groupby(level=0)
    .max()
    .T
)

# -----------------------------
# 5. Collect top-5 minerals per row
# -----------------------------
top5_series = []

for _, row in df_minerals.iterrows():
    top5 = row.nlargest(5).index
    top5_series.extend(top5)

# -----------------------------
# 6. Count frequency of top-5 appearances
# -----------------------------
top5_counts = pd.Series(top5_series).value_counts()

# Optional: keep only the very top minerals
TOP_N = 20
top5_counts = top5_counts.head(TOP_N)

# -----------------------------
# 7. Plot
# -----------------------------
plt.figure(figsize=(10, 6))
top5_counts.plot(kind="bar")
plt.xlabel("Mineral")
plt.ylabel("Times appearing in top 5")
plt.title("Most Frequent Minerals in Top 5 Rankings")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
