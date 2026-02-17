from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from train import load_data, train_model


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outpath: str = "reports/figures/pred_vs_actual.png",
) -> None:
    """
    Save a simple scatter plot of predicted vs actual costs.
    Interviewer-friendly artifact: quick visual sanity check.
    """
    plt.figure()
    plt.scatter(y_true, y_pred, s=8)
    plt.xlabel("Actual allowed_amount ($)")
    plt.ylabel("Predicted allowed_amount ($)")
    plt.title("Predicted vs Actual (Synthetic Claims Cost Regression)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


if __name__ == "__main__":
    # Load dataset + train baseline model
    df = load_data()
    model, metrics = train_model(df)

    # Recreate a test split just for plotting (kept simple for day-1)
    X = df.drop(columns=["allowed_amount"])
    y = df["allowed_amount"].to_numpy()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model predicts in log-space (because train_model log-transforms y),
    # so we invert back to dollars here.
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)

    # Save plot artifact
    plot_pred_vs_actual(y_test, preds)

    print("Saved plot to reports/figures/pred_vs_actual.png")
    print("Metrics (test set):")
    for k, v in metrics.items():
        print(f"{k}: {v:,.4f}")
