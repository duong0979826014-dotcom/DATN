import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results/calibration/prediction_results.csv"
)

summary = (
    df.groupby("distance")
    .agg({
        "distance_pred": "mean"
    })
    .reset_index()
)

plt.figure(figsize=(8,6))

plt.plot(
    summary["distance"],
    summary["distance"],
    "r--",
    linewidth=2,
    label="Ground Truth"
)

plt.plot(
    summary["distance"],
    summary["distance_pred"],
    "bo-",
    linewidth=2,
    label="Prediction"
)

for _, row in summary.iterrows():

    plt.text(
        row["distance"],
        row["distance_pred"],
        f"{row['distance_pred']:.1f}"
    )

plt.xlabel("Distance (m)")
plt.ylabel("Distance (m)")
plt.title("Mean Predicted Distance vs Ground Truth")

plt.grid(True)
plt.legend()

plt.savefig(
    "results/plots/mean_prediction_vs_gt.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved: results/plots/mean_prediction_vs_gt.png"
)
