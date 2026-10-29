"""
Train and evaluate attrition models.

- 3 models compared with 5-fold stratified cross-validation (ROC-AUC)
- best model evaluated on a held-out 20% test set
- decision threshold chosen to maximise business savings, not accuracy
- permutation importance to explain the drivers
- saves the model to models/attrition_model.joblib
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC, add_features
from style import BLUE, CATEGORICAL as PALETTE, MUTED, ORANGE, apply_style

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
IMG, REPORTS, MODELS = ROOT / "images", ROOT / "reports", ROOT / "models"
for d in (IMG, REPORTS, MODELS):
    d.mkdir(exist_ok=True)

# Business assumptions (documented in README)
REPLACEMENT_COST_MONTHS = 6     # cost to replace a leaver ≈ 6 months' salary
INTERVENTION_COST = 3_000       # RM per employee flagged (1:1s, retention bonus, training)
INTERVENTION_SUCCESS = 0.30     # share of true leavers retained by the intervention


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
    ])


MODELS_TO_TRY = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                            class_weight="balanced", n_jobs=-1, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                        max_leaf_nodes=15, l2_regularization=1.0,
                                                        class_weight="balanced", random_state=SEED),
}


def savings_at(threshold, proba, y, income):
    """Net RM saved by intervening on everyone above the threshold."""
    flagged = proba >= threshold
    tp = flagged & (y == 1)
    saved = (income[tp] * REPLACEMENT_COST_MONTHS * INTERVENTION_SUCCESS).sum()
    return saved - flagged.sum() * INTERVENTION_COST


def main() -> None:
    apply_style()
    raw = pd.read_csv(ROOT / "data" / "employees.csv")
    y = raw["attrition"].eq("Yes").astype(int).to_numpy()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        raw, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, peer = add_features(X_train_raw)
    X_test, _ = add_features(X_test_raw, peer)

    # ---------------------------------------------------- model comparison
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_rows = []
    print("5-fold CV ROC-AUC")
    for name, model in MODELS_TO_TRY.items():
        pipe = Pipeline([("prep", preprocessor()), ("model", model)])
        s = cross_val_score(pipe, X_train[NUMERIC + CATEGORICAL], y_train, cv=cv, scoring="roc_auc")
        cv_rows.append({"model": name, "cv_roc_auc_mean": s.mean().round(3), "cv_roc_auc_std": s.std().round(3)})
        print(f"  {name:<20} {s.mean():.3f} ± {s.std():.3f}")
    cv_df = pd.DataFrame(cv_rows).sort_values("cv_roc_auc_mean", ascending=False)
    cv_df.to_csv(REPORTS / "model_comparison.csv", index=False)

    best_name = cv_df.iloc[0]["model"]
    best = Pipeline([("prep", preprocessor()), ("model", MODELS_TO_TRY[best_name])])
    best.fit(X_train[NUMERIC + CATEGORICAL], y_train)
    proba = best.predict_proba(X_test[NUMERIC + CATEGORICAL])[:, 1]

    # all models on test set (for ROC chart)
    test_curves = {}
    for name, model in MODELS_TO_TRY.items():
        p = Pipeline([("prep", preprocessor()), ("model", model)]).fit(
            X_train[NUMERIC + CATEGORICAL], y_train).predict_proba(X_test[NUMERIC + CATEGORICAL])[:, 1]
        test_curves[name] = p

    # ------------------------------------------------ threshold by business value
    income = X_test["monthly_income"].to_numpy()
    thresholds = np.round(np.arange(0.20, 0.81, 0.01), 2)
    savings = np.array([savings_at(t, proba, y_test, income) for t in thresholds])
    best_t = float(thresholds[savings.argmax()])
    pred = (proba >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        "best_model": best_name,
        "test_roc_auc": round(roc_auc_score(y_test, proba), 3),
        "test_pr_auc": round(average_precision_score(y_test, proba), 3),
        "baseline_pr_auc (attrition rate)": round(y_test.mean(), 3),
        "chosen_threshold": best_t,
        "precision": round(tp / (tp + fp), 3), "recall": round(tp / (tp + fn), 3),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "employees_flagged": int(tp + fp),
        "est_net_savings_test_set_RM": int(savings.max()),
        # test set = 20% of staff -> scale to the whole workforce
        "est_net_savings_whole_workforce_RM": int(savings.max() / 0.2),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, pred, target_names=["Stayed", "Left"]))

    # --------------------------------------------------------------- charts
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for (name, p), c in zip(test_curves.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(y_test, p):.2f})")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("ROC curve (test set)", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[1]
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax.plot(rec, prec, color=BLUE, lw=2)
    ax.axhline(y_test.mean(), color=MUTED, ls="--", lw=1)
    ax.text(0.02, y_test.mean() + 0.02, "random guess", color=MUTED, fontsize=9)
    ax.scatter([metrics["recall"]], [metrics["precision"]], color=ORANGE, s=60, zorder=3,
               edgecolor="white", linewidth=2)
    ax.annotate(f"chosen threshold {best_t}", (metrics["recall"], metrics["precision"]),
                xytext=(10, 10), textcoords="offset points", fontsize=9, color=MUTED)
    ax.set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
    ax.set_title(f"Precision–recall: {best_name}", loc="left")

    ax = axes[2]
    ax.plot(thresholds, savings / 1000, color=BLUE, lw=2)
    ax.axvline(best_t, color=ORANGE, lw=1.5)
    ax.annotate(f"max RM{savings.max()/1000:.0f}K\n@ threshold {best_t}",
                (best_t, savings.max() / 1000), xytext=(-150, -60), textcoords="offset points",
                fontsize=9, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(xlabel="Probability threshold", ylabel="Net savings (RM '000)")
    ax.set_title("Picking the threshold by business value", loc="left")
    fig.tight_layout()
    fig.savefig(IMG / "model_evaluation.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------- feature importance
    pi = permutation_importance(best, X_test[NUMERIC + CATEGORICAL], y_test, scoring="roc_auc",
                                n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = (pd.DataFrame({"feature": NUMERIC + CATEGORICAL, "importance": pi.importances_mean,
                         "std": pi.importances_std})
             .sort_values("importance", ascending=False))
    imp.round(4).to_csv(REPORTS / "feature_importance.csv", index=False)
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["feature"].str.replace("_", " "), top["importance"], xerr=top["std"],
            color=BLUE, height=0.6, error_kw=dict(ecolor=MUTED, lw=1))
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title("What drives attrition? (permutation importance)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(IMG / "feature_importance.png", dpi=150)
    plt.close(fig)

    # ----------------------------------- odds ratios (interpretable model only)
    if best_name == "Logistic Regression":
        names = best.named_steps["prep"].get_feature_names_out()
        coefs = best.named_steps["model"].coef_[0]
        odds = (pd.DataFrame({"feature": names, "coef": coefs, "odds_ratio": np.exp(coefs)})
                  .assign(feature=lambda d: d["feature"].str.replace(r"^(num|cat)__", "", regex=True))
                  .sort_values("coef", ascending=False))
        odds.round(3).to_csv(REPORTS / "logistic_odds_ratios.csv", index=False)

    # --------------------------------------- refit on all data & score everyone
    full, peer_all = add_features(raw)
    final = Pipeline([("prep", preprocessor()), ("model", MODELS_TO_TRY[best_name])])
    final.fit(full[NUMERIC + CATEGORICAL], y)
    joblib.dump({"model": final, "peer_medians": peer_all, "threshold": best_t,
                 "model_name": best_name}, MODELS / "attrition_model.joblib")

    current = full[full["attrition"] == "No"].copy()
    current["attrition_risk"] = final.predict_proba(current[NUMERIC + CATEGORICAL])[:, 1].round(3)
    current["risk_band"] = pd.cut(current["attrition_risk"], [0, 0.3, best_t, 1],
                                  labels=["Low", "Medium", "High"], include_lowest=True)
    cols = ["employee_id", "department", "job_level", "monthly_income", "overtime",
            "job_satisfaction", "years_since_last_promotion", "pay_vs_peer_median",
            "attrition_risk", "risk_band"]
    current.sort_values("attrition_risk", ascending=False)[cols].to_csv(
        REPORTS / "current_employee_risk.csv", index=False)
    print("Risk bands (current employees):", current["risk_band"].value_counts().to_dict())
    print(f"Saved model -> {MODELS / 'attrition_model.joblib'}")


if __name__ == "__main__":
    main()
