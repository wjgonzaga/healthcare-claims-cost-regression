
# Bring in the external tools (libraries)


# Ues newer, safer Python typing rules
from __future__ import annotations

import numpy as np # For fast math need for modeling healthcare costs at scale
import pandas as pd # Raw claims data becomes model-ready analytics data

# Sklearn = standard ML library
from sklearn.compose import ColumnTransformer # allows different preprocessing for different column types
                                              # ensures demographics, plan info, and utilization metrics are all repared correctly for prediction 
from sklearn.linear_model import Ridge  #A linear regression model with regularization
                                        # Gives a reliable first estimate of expected healthcare cost without overreacting to noisy data.

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # How good is prediction
from sklearn.model_selection import train_test_split # Prevents fake optimism. Shows how the model would perform on new unseen patients. CREDIBILITY.
from sklearn.pipeline import Pipeline # This chains steps (preprocessing>model>predition) together
                                      # Ensures the model runs reliably the same way in production as testing
from sklearn.preprocessing import OneHotEncoder # Turns real-world categories into math the model can understand

# Reads the Data set created in data.py
def load_data(path: str = "data/processed/claims.csv") -> pd.DataFrame:
    return pd.read_csv(path)

# Demographics + utilizatino to extimate future allowed cost
def train_model(df: pd.DataFrame):
    target = "allowed_amount"
    X = df.drop(columns=[target])
    y = df[target].to_numpy()

    # log-transform: costs are right-skewed
    # Log-transform makes the problem more stable and easier for linear models
    # Here we are preventing the model from being dominated by rare, extremely expensive cases.
    y_log = np.log1p(y)

    cat_cols = ["sex", "region", "plan_type"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # converts region into numeric 0/1 columns.
    # The pipeline won't break when the business adds a new plan type or a new regional label
    # 'Futureproofing'
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # Ridge is the linear regression + regularization (prevents crazy coefficients, improves generalization)
    model = Ridge(alpha=1.0)
    pipe = Pipeline([("pre", pre), ("model", model)])

    # Simulating how model would perform on new members, not just memorizing history
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # Trains preprocessing + model together - No leakage
    pipe.fit(X_train, y_train)
    preds_log = pipe.predict(X_test)

    # invert back from log-transform for dollar metrics
    # Metrics are in real dollars, which finance/ops can understand
    y_true = np.expm1(y_test)
    preds = np.expm1(preds_log)

    mae = mean_absolute_error(y_true, preds)  
    mse = mean_squared_error(y_true, preds) # MAE - on average we're off by about $  --FINANCE
    rmse = np.sqrt(mse)  # RMSE - Big misses matter in high-cost cases --CARE MANAGEMENT
    r2 = r2_score(y_true, preds) # r2 - How much of cost variability can we explain --DATA SCIENCE

    return pipe, {"MAE": mae, "RMSE": rmse, "R2": r2}

# MAIN BLOCK. Prints metrics
if __name__ == "__main__":
    df = load_data()
    _, metrics = train_model(df)
    print("Metrics (test set):")
    for k, v in metrics.items():
        print(f"{k}: {v:,.4f}")
