import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(
    "results/calibration/prediction_results.csv"
)

# ==========================
# PLOT
# ==========================

plt.figure(figsize=(8, 8))

plt.scatter(
    df["distance"],
    df["distance_pred"],
    alpha=0.6,
    s=30,
    label="Predictions"
)

# Đường lý tưởng y = x

max_val = max(
    df["distance"].max(),
    df["distance_pred"].max()
)

plt.plot(
    [0, max_val],
    [0, max_val],
    "r--",
    linewidth=2,
    label="Ideal (y=x)"
)

plt.xlabel("Ground Truth Distance (m)")
plt.ylabel("Predicted Distance (m)")
plt.title("Predicted vs Ground Truth Distance")

plt.grid(True)
plt.legend()

plt.savefig(
    "results/plots/prediction_vs_groundtruth.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved: results/plots/prediction_vs_groundtruth.png"
)
