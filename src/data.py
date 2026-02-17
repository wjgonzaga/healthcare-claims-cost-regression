from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_claims(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    """
    Publishable synthetic healthcare claims-like dataset (no PHI).
    Target is right-skewed allowed_amount with realistic utilization drivers.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 90, size=n)
    sex = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])
    region = rng.choice(["West", "Midwest", "South", "Northeast"], size=n, p=[0.23, 0.22, 0.35, 0.20])
    plan_type = rng.choice(["HMO", "PPO", "HDHP"], size=n, p=[0.35, 0.45, 0.20])

    chronic_conditions = rng.poisson(lam=1.4, size=n).clip(0, 8)
    er_visits = rng.poisson(lam=0.25 + 0.10 * (chronic_conditions > 1), size=n).clip(0, 10)
    inpatient_admits = rng.poisson(lam=0.05 + 0.03 * (chronic_conditions > 2), size=n).clip(0, 5)
    outpatient_visits = rng.poisson(lam=2.0 + 0.35 * chronic_conditions, size=n).clip(0, 40)
    rx_fills = rng.poisson(lam=3.0 + 0.8 * chronic_conditions, size=n).clip(0, 60)

    base = 200 + (age - 18) * 8
    chronic_cost = chronic_conditions * 450
    utilization = (
        er_visits * 1200
        + inpatient_admits * 9000
        + outpatient_visits * 180
        + rx_fills * 55
    )

    region_factor = np.select(
        [region == "West", region == "Northeast", region == "Midwest", region == "South"],
        [1.10, 1.08, 0.98, 0.95],
        default=1.0,
    )

    plan_factor = np.select(
        [plan_type == "PPO", plan_type == "HDHP", plan_type == "HMO"],
        [1.05, 0.97, 0.99],
        default=1.0,
    )

    sex_factor = np.where(sex == "F", 1.02, 1.00)

    # multiplicative log-normal noise → right-skewed cost
    noise = rng.lognormal(mean=0.0, sigma=0.35, size=n)

    allowed_amount = (
        (base + chronic_cost + utilization)
        * region_factor
        * plan_factor
        * sex_factor
        * noise
    )

    allowed_amount = np.maximum(allowed_amount, 0).round(2)

    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "region": region,
            "plan_type": plan_type,
            "chronic_conditions": chronic_conditions,
            "er_visits": er_visits,
            "inpatient_admits": inpatient_admits,
            "outpatient_visits": outpatient_visits,
            "rx_fills": rx_fills,
            "allowed_amount": allowed_amount,
        }
    )


def save_processed(df: pd.DataFrame, path: str = "data/processed/claims.csv") -> None:
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = make_synthetic_claims()
    save_processed(df)
    print(f"Saved {len(df):,} rows to data/processed/claims.csv")
