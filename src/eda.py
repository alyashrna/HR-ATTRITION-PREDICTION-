"""
Exploratory analysis: which factors are associated with employees leaving?
Exports charts to images/ and a summary table to reports/eda_attrition_rates.csv
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from style import BLUE, MUTED, ORANGE, apply_style

ROOT = Path(__file__).resolve().parents[1]
IMG, REPORTS = ROOT / "images", ROOT / "reports"
IMG.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True)["left"].agg(["mean", "size"]).reset_index()
    g.columns = ["value", "attrition_rate", "employees"]
    g.insert(0, "factor", col)
    return g


def main() -> None:
    apply_style()
    df = pd.read_csv(ROOT / "data" / "employees.csv")
    df["left"] = df["attrition"].eq("Yes")
    overall = df["left"].mean()
    print(f"Overall attrition rate: {overall:.1%}")

    df["tenure_band"] = pd.cut(df["years_at_company"], [-1, 1, 3, 5, 10, 50],
                               labels=["0-1 yr", "2-3 yrs", "4-5 yrs", "6-10 yrs", "10+ yrs"])
    peer = df.groupby(["department", "job_level"])["monthly_income"].transform("median")
    df["pay_band"] = pd.cut(df["monthly_income"] / peer, [0, 0.9, 1.1, 10],
                            labels=["Below peers (<90%)", "In line", "Above peers (>110%)"])
    df["commute"] = pd.cut(df["distance_from_home_km"], [0, 10, 25, 200],
                           labels=["<10 km", "10-25 km", ">25 km"])
    sat_labels = {1: "1 Low", 2: "2", 3: "3", 4: "4 High"}
    df["job_satisfaction_lbl"] = df["job_satisfaction"].map(sat_labels)

    panels = [("overtime", "Overtime"), ("pay_band", "Pay vs peers (same dept & level)"),
              ("job_satisfaction_lbl", "Job satisfaction"), ("tenure_band", "Tenure"),
              ("work_mode", "Work mode"), ("department", "Department")]
    tables = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title) in zip(axes.flat, panels):
        t = rate_by(df, col)
        tables.append(t)
        if col == "department":
            t = t.sort_values("attrition_rate")
        colors = [ORANGE if r == t["attrition_rate"].max() else BLUE for r in t["attrition_rate"]]
        ax.barh(t["value"].astype(str), t["attrition_rate"] * 100, color=colors, height=0.6)
        ax.axvline(overall * 100, color=MUTED, ls="--", lw=1)
        for y, v in enumerate(t["attrition_rate"] * 100):
            ax.text(v, y, f" {v:.0f}%", va="center", fontsize=9, color=MUTED)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlim(0, t["attrition_rate"].max() * 100 * 1.25)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Attrition rate by factor  (dashed line = company average {overall:.0%}, "
                 f"orange = highest)", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMG / "attrition_by_factor.png", dpi=150)
    plt.close(fig)

    # income distribution leavers vs stayers
    fig, ax = plt.subplots(figsize=(9, 4))
    for left, color, label in [(False, BLUE, "Stayed"), (True, ORANGE, "Left")]:
        ax.hist(df.loc[df["left"] == left, "monthly_income"] / 1000, bins=40, range=(0, 30),
                color=color, alpha=0.55, density=True, label=label)
    ax.set_xlabel("Monthly income (RM '000)")
    ax.set_yticks([])
    ax.legend(frameon=False)
    ax.set_title("Leavers are concentrated in lower income bands", loc="left")
    fig.tight_layout()
    fig.savefig(IMG / "income_distribution.png", dpi=150)
    plt.close(fig)

    out = pd.concat(tables)
    out["attrition_rate"] = (out["attrition_rate"] * 100).round(1)
    out.to_csv(REPORTS / "eda_attrition_rates.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
