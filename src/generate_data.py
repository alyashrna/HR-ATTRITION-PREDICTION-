"""
Generate a realistic synthetic HR dataset for a mid-sized Malaysian company.

Attrition is simulated from a logistic model with realistic drivers
(overtime, low satisfaction, stagnant promotion, low pay for the level,
long commute, early tenure) plus random noise, so the relationships are
learnable but not perfect.

Output: data/employees.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
rng = np.random.default_rng(SEED)
N = 4_000

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

DEPTS = {"Sales": 0.24, "Customer Service": 0.18, "Operations": 0.20, "IT": 0.16,
         "Finance": 0.10, "Marketing": 0.07, "Human Resources": 0.05}
# base monthly salary (RM) for job level 1..5
LEVEL_PAY = {1: 3_200, 2: 4_800, 3: 7_200, 4: 11_000, 5: 17_000}
DEPT_PAY_MULT = {"Sales": 1.0, "Customer Service": 0.85, "Operations": 0.95, "IT": 1.2,
                 "Finance": 1.1, "Marketing": 1.0, "Human Resources": 0.95}


def pick(options: dict, n: int):
    k = list(options)
    p = np.array(list(options.values()), dtype=float)
    return rng.choice(k, n, p=p / p.sum())


def main() -> None:
    dept = pick(DEPTS, N)
    age = np.clip(rng.normal(33, 8, N).round(), 21, 60).astype(int)
    # job level loosely follows age
    level = np.clip(((age - 21) / 8 + rng.normal(1, 0.9, N)).round(), 1, 5).astype(int)
    total_years = np.clip(age - 22 - rng.integers(0, 3, N), 0, None)
    years_at_company = np.minimum(total_years, rng.exponential(4.5, N).round()).astype(int)
    years_since_promo = np.minimum(years_at_company, rng.exponential(2.2, N).round()).astype(int)
    num_companies = np.clip(rng.poisson(total_years / 5), 0, 9)

    base = np.array([LEVEL_PAY[l] for l in level]) * np.array([DEPT_PAY_MULT[d] for d in dept])
    pay_ratio = rng.lognormal(0, 0.16, N)            # >1 = paid above band
    income = (base * pay_ratio).round(-1)

    overtime = rng.random(N) < np.where(np.isin(dept, ["Sales", "Operations", "IT"]), 0.34, 0.22)
    distance = np.clip(rng.gamma(2.0, 8, N).round(1), 1, 80)
    work_mode = rng.choice(["Onsite", "Hybrid", "Remote"], N, p=[0.52, 0.38, 0.10])
    job_sat = rng.choice([1, 2, 3, 4], N, p=[0.18, 0.20, 0.32, 0.30])
    env_sat = rng.choice([1, 2, 3, 4], N, p=[0.17, 0.21, 0.31, 0.31])
    wlb = rng.choice([1, 2, 3, 4], N, p=[0.08, 0.22, 0.52, 0.18])
    perf = rng.choice([2, 3, 4, 5], N, p=[0.06, 0.52, 0.32, 0.10])
    training = np.clip(rng.normal(24, 10, N).round(), 0, 60).astype(int)
    hike = np.clip(rng.normal(4.5, 2, N) + (perf - 3) * 1.2, 0, 15).round(1)
    manager_changes = np.clip(rng.poisson(years_at_company / 3), 0, 6)
    marital = rng.choice(["Single", "Married", "Divorced"], N, p=[0.42, 0.50, 0.08])
    education = rng.choice(["Diploma", "Bachelor's", "Master's", "PhD"], N, p=[0.22, 0.60, 0.16, 0.02])
    gender = rng.choice(["Female", "Male"], N, p=[0.48, 0.52])

    # ---- attrition logit: each term is a plausible real-world driver
    z = (-3.05
         + 1.25 * overtime
         + 0.55 * (job_sat == 1) + 0.25 * (job_sat == 2)
         + 0.40 * (env_sat == 1)
         + 0.60 * (wlb == 1) + 0.15 * (wlb == 2)
         - 1.40 * np.log(pay_ratio) * 3           # underpaid vs band -> leave
         + 0.16 * np.clip(years_since_promo - 2, 0, None)
         + 0.018 * np.clip(distance - 15, 0, None)
         + 0.55 * (years_at_company <= 1)
         - 0.08 * np.clip(years_at_company - 2, 0, 10)
         + 0.45 * (marital == "Single")
         + 0.12 * num_companies
         - 0.25 * (level - 1)
         - 0.10 * (hike - 4.5)
         + 0.30 * (work_mode == "Onsite") - 0.20 * (work_mode == "Hybrid")
         + 0.25 * (dept == "Sales") + 0.20 * (dept == "Customer Service")
         - 0.02 * (training - 24)
         + 0.20 * np.clip(manager_changes - 1, 0, None)
         + rng.normal(0, 0.55, N))                   # unobserved factors
    attrition = rng.random(N) < 1 / (1 + np.exp(-z))

    df = pd.DataFrame({
        "employee_id": [f"E{i:05d}" for i in range(1, N + 1)],
        "age": age, "gender": gender, "marital_status": marital, "education": education,
        "department": dept, "job_level": level, "monthly_income": income,
        "years_at_company": years_at_company, "years_since_last_promotion": years_since_promo,
        "total_working_years": total_years, "num_companies_worked": num_companies,
        "manager_changes": manager_changes,
        "overtime": np.where(overtime, "Yes", "No"), "work_mode": work_mode,
        "distance_from_home_km": distance,
        "job_satisfaction": job_sat, "environment_satisfaction": env_sat,
        "work_life_balance": wlb, "performance_rating": perf,
        "training_hours_last_year": training, "last_salary_hike_pct": hike,
        "attrition": np.where(attrition, "Yes", "No"),
    })
    df.to_csv(OUT / "employees.csv", index=False)
    print(f"{len(df):,} employees, attrition rate = {attrition.mean():.1%} -> {OUT/'employees.csv'}")


if __name__ == "__main__":
    main()
