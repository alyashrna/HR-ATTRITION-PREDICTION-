"""Feature engineering shared by training and the app."""
import pandas as pd

CATEGORICAL = ["gender", "marital_status", "education", "department", "overtime", "work_mode"]
NUMERIC = ["age", "job_level", "monthly_income", "years_at_company", "years_since_last_promotion",
           "total_working_years", "num_companies_worked", "manager_changes",
           "distance_from_home_km", "job_satisfaction", "environment_satisfaction",
           "work_life_balance", "performance_rating", "training_hours_last_year",
           "last_salary_hike_pct",
           # engineered
           "pay_vs_peer_median", "is_new_joiner", "promotion_stagnation", "job_hopper_index"]


def add_features(df: pd.DataFrame, peer_medians: pd.Series | None = None):
    """
    Add engineered features. `peer_medians` = median income per (department, job_level),
    learned on the training set so the test set / app never leak information.
    Returns (df_with_features, peer_medians).
    """
    df = df.copy()
    if peer_medians is None:
        peer_medians = df.groupby(["department", "job_level"])["monthly_income"].median()
    key = pd.MultiIndex.from_frame(df[["department", "job_level"]])
    df["pay_vs_peer_median"] = (df["monthly_income"].to_numpy() / peer_medians.reindex(key).to_numpy()).round(3)
    df["is_new_joiner"] = (df["years_at_company"] <= 1).astype(int)
    df["promotion_stagnation"] = (df["years_since_last_promotion"] / (df["years_at_company"] + 1)).round(3)
    df["job_hopper_index"] = (df["num_companies_worked"] / (df["total_working_years"] + 1)).round(3)
    return df, peer_medians
