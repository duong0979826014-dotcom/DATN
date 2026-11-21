import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(
    "results/calibration/calibration_summary.csv"
)

depth = df["mean"].values
distance = df["distance"].values

# ==========================
# MODELS
# ==========================

def inverse_model(x, a, b):
    return a / x + b

def power_model(x, a, b):
    return a * (x ** b)

def poly2_model(x, a, b, c):
    return a * x**2 + b * x + c

# ==========================
# FIT
# ==========================

params_inv, _ = curve_fit(
    inverse_model,
    depth,
    distance
)

params_power, _ = curve_fit(
    power_model,
    depth,
    distance,
    maxfev=10000
)

params_poly, _ = curve_fit(
    poly2_model,
    depth,
    distance
)

# ==========================
# PREDICTIONS
# ==========================

pred_inv = inverse_model(
    depth,
    *params_inv
)

pred_power = power_model(
    depth,
    *params_power
)

pred_poly = poly2_model(
    depth,
    *params_poly
)

# ==========================
# EVALUATE
# ==========================

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


results = []

for name, pred in [
    ("Inverse", pred_inv),
    ("Power", pred_power),
    ("Polynomial", pred_poly)
]:

    mae, rmse, r2 = evaluate(
        distance,
        pred
    )

    results.append([
        name,
        mae,
        rmse,
        r2
    ])

# ==========================
# PRINT RESULTS
# ==========================

print("\n===== MODEL COMPARISON =====\n")

for r in results:

    print(
        f"{r[0]:12s} | "
        f"MAE={r[1]:.4f} | "
        f"RMSE={r[2]:.4f} | "
        f"R2={r[3]:.4f}"
    )

print("\n===== EQUATIONS =====\n")

print(
    f"Inverse:\n"
    f"distance = "
    f"{params_inv[0]:.4f}/depth + "
    f"{params_inv[1]:.4f}\n"
)

print(
    f"Power:\n"
    f"distance = "
    f"{params_power[0]:.4f} * "
    f"depth^{params_power[1]:.4f}\n"
)

print(
    f"Polynomial:\n"
    f"distance = "
    f"{params_poly[0]:.8f}*depth² + "
    f"{params_poly[1]:.8f}*depth + "
    f"{params_poly[2]:.4f}\n"
)

# ==========================
# SAVE RESULTS
# ==========================

x_plot = np.linspace(
    depth.min(),
    depth.max(),
    300
)

plt.figure(figsize=(10,6))

plt.scatter(
    depth,
    distance,
    color="red",
    label="Data"
)

plt.plot(
    x_plot,
    inverse_model(
        x_plot,
        *params_inv
    ),
    label="Inverse"
)

plt.plot(
    x_plot,
    power_model(
        x_plot,
        *params_power
    ),
    label="Power"
)

plt.plot(
    x_plot,
    poly2_model(
        x_plot,
        *params_poly
    ),
    label="Polynomial"
)

plt.xlabel("Depth")
plt.ylabel("Distance (m)")
plt.title("Depth vs Distance Calibration")

plt.grid(True)
plt.legend()

plt.savefig(
    "results/plots/calibration_curve.png",
    dpi=300
)

print(
    "\nSaved: results/plots/calibration_curve.png"
)