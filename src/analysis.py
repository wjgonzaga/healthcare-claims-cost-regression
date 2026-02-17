
# This follow up deeper dive of data allows what exactly was causing R2 score to fall out.
# Why and where in the data set caused the massive overprediction and failed

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from train import load_data, train_model


def summarize_distribution(df: pd.DataFrame) -> None:
    """
    Print key statistics to understand cost skew and tail severity.
    """
    costs = df["allowed_amount"]

    print("\n=== Cost Distribution Summary ===")
    print(f"Count: {len(costs):,}")
    print(f"Median: ${costs.median():,.2f}")
    print(f"95th percentile: ${np.percentile(costs, 95):,.2f}")
    print(f"99th percentile: ${np.percentile(costs, 99):,.2f}")
    print(f"Max: ${costs.max():,.2f}")


def analyze_prediction_errors(df: pd.DataFrame) -> None:
    """
    Identify the largest absolute prediction errors.
    """
    model, _ = train_model(df)

    X = df.drop(columns=["allowed_amount"])
    y = df["allowed_amount"].to_numpy()

    preds_log = model.predict(X)
    preds = np.expm1(preds_log)

    errors = np.abs(y - preds)

    error_df = pd.DataFrame(
        {
            "actual_cost": y,
            "predicted_cost": preds,
            "absolute_error": errors,
        }
    ).sort_values("absolute_error", ascending=False)

    print("\n=== Top 10 Largest Prediction Errors ===")
    print(error_df.head(10))


def r2_without_top_outliers(df: pd.DataFrame, top_pct: float = 0.01) -> None:
    """
    Recompute R² after removing the highest-cost outliers.
    This is for diagnostic understanding, not model cheating.
    """
    threshold = np.percentile(df["allowed_amount"], 100 * (1 - top_pct))
    filtered_df = df[df["allowed_amount"] <= threshold]

    model, _ = train_model(filtered_df)

    X = filtered_df.drop(columns=["allowed_amount"])
    y = filtered_df["allowed_amount"].to_numpy()

    preds_log = model.predict(X)
    preds = np.expm1(preds_log)

    r2 = r2_score(y, preds)

    print(f"\n=== R² after removing top {int(top_pct*100)}% highest-cost cases ===")
    print(f"R²: {r2:.4f}")
    print(f"Rows remaining: {len(filtered_df):,}")


if __name__ == "__main__":
    df = load_data()

    summarize_distribution(df)
    analyze_prediction_errors(df)
    r2_without_top_outliers(df)
