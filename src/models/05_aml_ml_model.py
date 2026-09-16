"""
======================================================================
US FINANCIAL CRIME AI PROJECT
PHASE 5 - AML MACHINE LEARNING MODEL
======================================================================

Purpose:
    Train and compare machine-learning models for suspicious
    transaction detection.

Input:
    data/processed/aml_ml_features.csv

Models:
    1. Logistic Regression baseline
    2. XGBoost
    3. CatBoost

Evaluation:
    Precision
    Recall
    F1
    ROC-AUC
    PR-AUC
    Confusion Matrix

Important:
    This is a synthetic portfolio/demo project.
    Model outputs are analytical and are NOT SAR filing decisions.
======================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


warnings.filterwarnings("ignore")


# ======================================================================
# 1. PROJECT PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "aml_ml_features.csv"
)

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 2. HELPER FUNCTION
# ======================================================================

def print_section(title):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


# ======================================================================
# 3. LOAD DATA
# ======================================================================

print_section("PHASE 5 - AML MACHINE LEARNING")

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n\n"
        "Please run Phase 4 first."
    )

print(f"Loading:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(
    f"\nDataset shape: "
    f"{df.shape}"
)


# ======================================================================
# 4. BASIC CLEANING
# ======================================================================

print_section("4. DATA PREPARATION")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = df.dropna(
    subset=["target"]
).copy()

df["target"] = (
    df["target"]
    .astype(int)
)


print(
    f"Transactions available: "
    f"{len(df):,}"
)

print(
    f"Suspicious transactions: "
    f"{df['target'].sum():,}"
)

print(
    f"Normal transactions: "
    f"{(df['target'] == 0).sum():,}"
)

print(
    f"Suspicious rate: "
    f"{df['target'].mean() * 100:.2f}%"
)


# ======================================================================
# 5. SORT BY TIME
# ======================================================================

print_section("5. TEMPORAL ORDERING")

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ======================================================================
# 6. IDENTIFY LEAKAGE / NON-MODEL COLUMNS
# ======================================================================

print_section("6. LEAKAGE CONTROL")

"""
These columns should NOT be directly used as model features.

known_suspicious:
    This is the synthetic ground-truth label.

target:
    Same target label.

scenario:
    Directly identifies the synthetic suspicious scenario.

transaction_id/customer_id:
    Identifiers rather than behavioral ML features.

Other text/category fields:
    We will keep the first model focused on engineered numerical
    behavioral features.
"""

excluded_columns = [
    "transaction_id",
    "customer_id",
    "account_id",
    "sender_account",
    "receiver_account",
    "timestamp",

    "scenario",

    "transaction_type",
    "channel",
    "origin_country",
    "destination_country",

    "known_suspicious",
    "target",
]


candidate_features = [
    column
    for column in df.columns
    if column not in excluded_columns
]


# ======================================================================
# 7. KEEP NUMERIC FEATURES ONLY
# ======================================================================

numeric_features = []

for column in candidate_features:

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):

        numeric_features.append(column)


print(
    f"\nCandidate numerical features: "
    f"{len(numeric_features)}"
)


# ======================================================================
# 8. REMOVE POTENTIAL DIRECT LABEL PROXY
# ======================================================================

"""
The synthetic dataset contains rule-engine outputs.

We will NOT allow direct rule outputs that may be too closely tied
to the injected suspicious label to dominate the first model.

However, behavioral features such as transaction velocity,
structuring counts, cash ratios, etc. remain useful.

The following direct alert/ground-truth style columns are excluded
from the first clean ML experiment.
"""

direct_rule_columns = [
    "rule_alert",
    "rule_risk_score",
    "red_flag_count",

    "customer_average_rule_score",
    "customer_max_rule_score",
    "high_risk_rule_indicator",

    "customer_average_red_flags",
    "customer_max_red_flags",
]


model_features = [
    column
    for column in numeric_features
    if column not in direct_rule_columns
]


print(
    f"Final ML features: "
    f"{len(model_features)}"
)

print("\nFeatures used:")

for number, feature in enumerate(
    model_features,
    start=1
):

    print(
        f"{number:02d}. {feature}"
    )


# ======================================================================
# 9. HANDLE MISSING / INFINITE VALUES
# ======================================================================

print_section("9. FEATURE CLEANING")

X = (
    df[model_features]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)

y = df["target"]


print(
    f"X shape: {X.shape}"
)

print(
    f"y shape: {y.shape}"
)


# ======================================================================
# 10. TRAIN / TEST SPLIT
# ======================================================================

print_section("10. TRAIN TEST SPLIT")

"""
We use a time-based split rather than randomly mixing future
transactions into training data.

First 80%:
    Training

Last 20%:
    Testing

This better represents how a transaction-monitoring model would
operate in practice.
"""

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()


print(
    f"Training records: "
    f"{len(X_train):,}"
)

print(
    f"Testing records: "
    f"{len(X_test):,}"
)

print(
    f"Training suspicious rate: "
    f"{y_train.mean() * 100:.2f}%"
)

print(
    f"Testing suspicious rate: "
    f"{y_test.mean() * 100:.2f}%"
)


# ======================================================================
# 11. HANDLE CLASS IMBALANCE
# ======================================================================

print_section("11. CLASS IMBALANCE")

positive_count = int(
    y_train.sum()
)

negative_count = int(
    len(y_train) - positive_count
)

if positive_count > 0:

    scale_pos_weight = (
        negative_count
        / positive_count
    )

else:

    scale_pos_weight = 1.0


print(
    f"Normal training transactions: "
    f"{negative_count:,}"
)

print(
    f"Suspicious training transactions: "
    f"{positive_count:,}"
)

print(
    f"Calculated scale_pos_weight: "
    f"{scale_pos_weight:.2f}"
)


# ======================================================================
# 12. MODEL 1 - LOGISTIC REGRESSION
# ======================================================================

print_section("12. LOGISTIC REGRESSION BASELINE")

logistic_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


print(
    "Training Logistic Regression..."
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_probability = (
    logistic_model
    .predict_proba(X_test)[:, 1]
)


# ======================================================================
# 13. MODEL 2 - XGBOOST
# ======================================================================

print_section("13. XGBOOST")

xgb_model = XGBClassifier(

    n_estimators=400,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.85,

    colsample_bytree=0.85,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1
)


print(
    "Training XGBoost..."
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_probability = (
    xgb_model
    .predict_proba(X_test)[:, 1]
)


# ======================================================================
# 14. MODEL 3 - CATBOOST
# ======================================================================

print_section("14. CATBOOST")

catboost_model = CatBoostClassifier(

    iterations=400,

    depth=7,

    learning_rate=0.05,

    loss_function="Logloss",

    eval_metric="AUC",

    verbose=False,

    random_seed=42,

    thread_count=-1,

    auto_class_weights="Balanced"
)


print(
    "Training CatBoost..."
)

catboost_model.fit(
    X_train,
    y_train
)

catboost_probability = (
    catboost_model
    .predict_proba(X_test)[:, 1]
)


# ======================================================================
# 15. EVALUATION FUNCTION
# ======================================================================

def evaluate_model(
    model_name,
    probabilities,
    threshold=0.50
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = safe_rate(
        fp,
        fp + tn
    )

    false_negative_rate = safe_rate(
        fn,
        fn + tp
    )

    print("\n")
    print("-" * 70)

    print(
        f"MODEL: {model_name}"
    )

    print("-" * 70)

    print(
        f"Threshold          : {threshold:.2f}"
    )

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Precision          : {precision:.4f}"
    )

    print(
        f"Recall             : {recall:.4f}"
    )

    print(
        f"F1 Score           : {f1:.4f}"
    )

    print(
        f"ROC-AUC            : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC             : {pr_auc:.4f}"
    )

    print(
        f"True Negatives     : {tn:,}"
    )

    print(
        f"False Positives    : {fp:,}"
    )

    print(
        f"False Negatives    : {fn:,}"
    )

    print(
        f"True Positives     : {tp:,}"
    )

    print(
        f"False Positive Rate: {false_positive_rate:.4f}"
    )

    print(
        f"False Negative Rate: {false_negative_rate:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def safe_rate(
    numerator,
    denominator
):

    if denominator == 0:

        return 0.0

    return numerator / denominator


# ======================================================================
# 16. EVALUATE MODELS
# ======================================================================

print_section("16. MODEL EVALUATION")

results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        logistic_probability
    )
)

results.append(
    evaluate_model(
        "XGBoost",
        xgb_probability
    )
)

results.append(
    evaluate_model(
        "CatBoost",
        catboost_probability
    )
)


results_df = pd.DataFrame(
    results
)


# ======================================================================
# 17. MODEL COMPARISON
# ======================================================================

print_section("17. MODEL COMPARISON")

comparison_columns = [
    "model",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "false_positive_rate",
    "false_negative_rate",
]

print(
    results_df[
        comparison_columns
    ].to_string(
        index=False
    )
)


# ======================================================================
# 18. SELECT BEST MODEL
# ======================================================================

print_section("18. MODEL SELECTION")

"""
For this first AML experiment we select based on PR-AUC.

Why PR-AUC?

AML datasets can be highly imbalanced.
PR-AUC gives more useful information about performance on
the positive/suspicious class than accuracy alone.
"""

best_index = (
    results_df["pr_auc"]
    .idxmax()
)

best_model_name = (
    results_df
    .loc[best_index, "model"]
)

best_pr_auc = (
    results_df
    .loc[best_index, "pr_auc"]
)


print(
    f"Best model: "
    f"{best_model_name}"
)

print(
    f"Best PR-AUC: "
    f"{best_pr_auc:.4f}"
)


# ======================================================================
# 19. SELECT MODEL OBJECT
# ======================================================================

if best_model_name == "Logistic Regression":

    best_model = logistic_model

    best_probability = logistic_probability

elif best_model_name == "XGBoost":

    best_model = xgb_model

    best_probability = xgb_probability

else:

    best_model = catboost_model

    best_probability = catboost_probability


# ======================================================================
# 20. SAVE MODEL
# ======================================================================

print_section("20. SAVING BEST MODEL")

import joblib


if best_model_name == "Logistic Regression":

    model_file = (
        MODEL_DIR
        / "aml_logistic_regression.joblib"
    )

    joblib.dump(
        best_model,
        model_file
    )

elif best_model_name == "XGBoost":

    model_file = (
        MODEL_DIR
        / "aml_xgboost.json"
    )

    best_model.save_model(
        model_file
    )

else:

    model_file = (
        MODEL_DIR
        / "aml_catboost.cbm"
    )

    best_model.save_model(
        model_file
    )


print(
    f"Best model saved to:\n"
    f"{model_file}"
)


# ======================================================================
# 21. SAVE TEST PREDICTIONS
# ======================================================================

print_section("21. SAVING TEST PREDICTIONS")

test_predictions = df.iloc[
    split_index:
].copy()

test_predictions[
    "ml_probability"
] = best_probability

test_predictions[
    "ml_prediction"
] = (
    best_probability >= 0.50
).astype(int)


test_predictions[
    "model_risk_level"
] = pd.cut(

    test_predictions[
        "ml_probability"
    ],

    bins=[
        -0.001,
        0.25,
        0.50,
        0.75,
        1.001
    ],

    labels=[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]
)


prediction_file = (
    REPORT_DIR
    / "aml_test_predictions.csv"
)

test_predictions.to_csv(
    prediction_file,
    index=False
)

print(
    f"Test predictions saved to:\n"
    f"{prediction_file}"
)


# ======================================================================
# 22. CONFUSION MATRIX
# ======================================================================

print_section("22. CONFUSION MATRIX")

best_predictions = (
    best_probability >= 0.50
).astype(int)

cm = confusion_matrix(
    y_test,
    best_predictions
)

print(
    "\nConfusion Matrix:"
)

print(cm)


plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm
)

plt.title(
    f"AML Model Confusion Matrix - {best_model_name}"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    [0, 1],
    ["Normal", "Suspicious"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Suspicious"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

confusion_file = (
    REPORT_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    confusion_file,
    dpi=150
)

plt.close()


# ======================================================================
# 23. PRECISION-RECALL CURVE
# ======================================================================

print_section("23. PRECISION RECALL CURVE")

precision_values, recall_values, thresholds = (
    precision_recall_curve(
        y_test,
        best_probability
    )
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_values,
    precision_values
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    f"Precision-Recall Curve - {best_model_name}"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

pr_curve_file = (
    REPORT_DIR
    / "precision_recall_curve.png"
)

plt.savefig(
    pr_curve_file,
    dpi=150
)

plt.close()


# ======================================================================
# 24. XGBOOST / CATBOOST FEATURE IMPORTANCE
# ======================================================================

print_section("24. FEATURE IMPORTANCE")

if best_model_name == "XGBoost":

    importance_values = (
        best_model.feature_importances_
    )

elif best_model_name == "CatBoost":

    importance_values = (
        best_model.get_feature_importance()
    )

else:

    importance_values = (
        np.abs(
            best_model
            .named_steps["model"]
            .coef_[0]
        )
    )


feature_importance = pd.DataFrame({

    "feature": model_features,

    "importance": importance_values

})


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


feature_importance_file = (
    REPORT_DIR
    / "feature_importance.csv"
)

feature_importance.to_csv(
    feature_importance_file,
    index=False
)


print(
    "\nTop 20 features:"
)

print(
    feature_importance
    .head(20)
    .to_string(index=False)
)


# ======================================================================
# 25. FEATURE IMPORTANCE CHART
# ======================================================================

top_features = (
    feature_importance
    .head(20)
    .sort_values(
        "importance"
    )
)


plt.figure(
    figsize=(10, 8)
)

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    f"Top AML Model Features - {best_model_name}"
)

plt.tight_layout()

importance_chart_file = (
    REPORT_DIR
    / "feature_importance.png"
)

plt.savefig(
    importance_chart_file,
    dpi=150
)

plt.close()


# ======================================================================
# 26. SAVE MODEL COMPARISON
# ======================================================================

comparison_file = (
    REPORT_DIR
    / "model_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)


# ======================================================================
# 27. MODEL SUMMARY
# ======================================================================

summary_file = (
    REPORT_DIR
    / "model_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "US FINANCIAL CRIME AI PROJECT\n"
    )

    file.write(
        "PHASE 5 - AML MACHINE LEARNING MODEL\n\n"
    )

    file.write(
        f"Best model: {best_model_name}\n"
    )

    file.write(
        f"PR-AUC: {best_pr_auc:.6f}\n"
    )

    best_row = results_df.loc[
        best_index
    ]

    file.write(
        f"Precision: {best_row['precision']:.6f}\n"
    )

    file.write(
        f"Recall: {best_row['recall']:.6f}\n"
    )

    file.write(
        f"F1: {best_row['f1']:.6f}\n"
    )

    file.write(
        f"ROC-AUC: {best_row['roc_auc']:.6f}\n"
    )

    file.write(
        f"False Positive Rate: "
        f"{best_row['false_positive_rate']:.6f}\n"
    )

    file.write(
        f"False Negative Rate: "
        f"{best_row['false_negative_rate']:.6f}\n"
    )


# ======================================================================
# 28. FINAL OUTPUT
# ======================================================================

print_section("PHASE 5 COMPLETED SUCCESSFULLY")

print(
    f"\nBest model: "
    f"{best_model_name}"
)

print(
    f"PR-AUC: "
    f"{best_pr_auc:.4f}"
)

print("\nGenerated files:")

print(
    f"1. {model_file}"
)

print(
    f"2. {prediction_file}"
)

print(
    f"3. {comparison_file}"
)

print(
    f"4. {feature_importance_file}"
)

print(
    f"5. {confusion_file}"
)

print(
    f"6. {pr_curve_file}"
)

print(
    f"7. {importance_chart_file}"
)

print(
    f"8. {summary_file}"
)

print("\n")
print("=" * 80)

print(
    "NEXT PHASE:"
)

print(
    "PHASE 6 - SHAP EXPLAINABILITY"
)

print(
    "\nWe will explain WHY the AML model considers "
    "a transaction suspicious."
)

print(
    "\nThis will make the project much more realistic "
    "for an AML / Financial Crime Analyst portfolio."
)

print("=" * 80)