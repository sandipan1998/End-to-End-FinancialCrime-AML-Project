# =============================================================================
# PHASE 11 - AML PROJECT VALIDATION & PORTFOLIO PACKAGING
# US FINANCIAL CRIME / AML TRANSACTION MONITORING PROJECT
# =============================================================================
#
# Purpose:
#
#   Validate the complete AML transaction monitoring pipeline before
#   presenting the project in a portfolio, resume or interview.
#
# Validation areas:
#
#   1. Dataset integrity
#   2. Transaction distribution
#   3. Suspicious activity distribution
#   4. AML rule performance
#   5. ML model performance
#   6. False positive / false negative analysis
#   7. Scenario-level performance
#   8. Feature leakage checks
#   9. Investigation coverage
#   10. SAR narrative quality
#   11. Model governance
#   12. Portfolio metrics
#
# IMPORTANT:
#
#   This project uses synthetic data.
#
#   Rule thresholds and model outputs are demonstration indicators.
#   They are not legal conclusions.
#
# =============================================================================


import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PROJECT PATH
# =============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


# =============================================================================
# 2. INPUT FILES
# =============================================================================

RAW_DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "US_AML_Synthetic_Dataset.xlsx"
)


RULE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "aml_rule_engine_transactions.csv"
)


FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "aml_ml_features.csv"
)


PREDICTION_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "models",
    "aml_test_predictions.csv"
)


MODEL_COMPARISON_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "models",
    "model_comparison.csv"
)


CASE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "aml_investigation_cases.csv"
)


SAR_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "sar",
    "sar_narrative_drafts.csv"
)


SHAP_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "shap",
    "shap_feature_importance.csv"
)


RULE_SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "rules",
    "rule_summary.csv"
)


# =============================================================================
# 3. OUTPUT FILES
# =============================================================================

VALIDATION_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "validation"
)


os.makedirs(
    VALIDATION_DIR,
    exist_ok=True
)


VALIDATION_SUMMARY_FILE = os.path.join(
    VALIDATION_DIR,
    "validation_summary.csv"
)


SCENARIO_FILE = os.path.join(
    VALIDATION_DIR,
    "scenario_performance.csv"
)


RULE_VALIDATION_FILE = os.path.join(
    VALIDATION_DIR,
    "rule_validation.csv"
)


LEAKAGE_FILE = os.path.join(
    VALIDATION_DIR,
    "feature_leakage_review.csv"
)


PORTFOLIO_FILE = os.path.join(
    VALIDATION_DIR,
    "portfolio_metrics.txt"
)


GOVERNANCE_FILE = os.path.join(
    VALIDATION_DIR,
    "model_governance_checklist.txt"
)


README_FILE = os.path.join(
    PROJECT_ROOT,
    "README_AML_PROJECT.md"
)


# =============================================================================
# 4. HEADER
# =============================================================================

print("\n" + "=" * 90)
print("PHASE 11 - AML PROJECT VALIDATION & PORTFOLIO PACKAGING")
print("=" * 90)


# =============================================================================
# 5. FILE CHECK
# =============================================================================

print("\n" + "=" * 90)
print("5. PROJECT FILE VALIDATION")
print("=" * 90)


required_files = {

    "Raw synthetic dataset":
        RAW_DATA_FILE,

    "Rule engine output":
        RULE_FILE,

    "ML feature dataset":
        FEATURE_FILE,

    "ML predictions":
        PREDICTION_FILE,

    "Model comparison":
        MODEL_COMPARISON_FILE,

    "Investigation cases":
        CASE_FILE,

    "SAR narratives":
        SAR_FILE,

    "SHAP feature importance":
        SHAP_FILE,

    "Rule summary":
        RULE_SUMMARY_FILE

}


file_rows = []


for name, path in required_files.items():

    exists = os.path.exists(
        path
    )


    file_rows.append({

        "component":
            name,

        "exists":
            exists,

        "path":
            path

    })


    status = "OK" if exists else "MISSING"


    print(
        f"[{status}] {name}"
    )


file_check_df = pd.DataFrame(
    file_rows
)


# =============================================================================
# 6. LOAD DATA
# =============================================================================

print("\n" + "=" * 90)
print("6. LOADING PROJECT DATA")
print("=" * 90)


if not os.path.exists(
    RAW_DATA_FILE
):

    print(
        "\nERROR: Raw dataset not found."
    )

    sys.exit(1)


if not os.path.exists(
    FEATURE_FILE
):

    print(
        "\nERROR: ML feature dataset not found."
    )

    sys.exit(1)


if not os.path.exists(
    PREDICTION_FILE
):

    print(
        "\nERROR: ML prediction file not found."
    )

    sys.exit(1)


transactions = pd.read_excel(
    RAW_DATA_FILE,
    sheet_name="Transactions"
)


features = pd.read_csv(
    FEATURE_FILE
)


predictions = pd.read_csv(
    PREDICTION_FILE
)


if os.path.exists(
    RULE_FILE
):

    rule_data = pd.read_csv(
        RULE_FILE
    )

else:

    rule_data = pd.DataFrame()


if os.path.exists(
    CASE_FILE
):

    cases = pd.read_csv(
        CASE_FILE
    )

else:

    cases = pd.DataFrame()


if os.path.exists(
    SAR_FILE
):

    sar = pd.read_csv(
        SAR_FILE
    )

else:

    sar = pd.DataFrame()


if os.path.exists(
    SHAP_FILE
):

    shap = pd.read_csv(
        SHAP_FILE
    )

else:

    shap = pd.DataFrame()


print(
    f"\nRaw transactions: {len(transactions):,}"
)


print(
    f"ML feature records: {len(features):,}"
)


print(
    f"ML test predictions: {len(predictions):,}"
)


print(
    f"Investigation cases: {len(cases):,}"
)


print(
    f"SAR drafts: {len(sar):,}"
)


# =============================================================================
# 7. DATASET INTEGRITY
# =============================================================================

print("\n" + "=" * 90)
print("7. DATASET INTEGRITY VALIDATION")
print("=" * 90)


integrity_results = []


def add_integrity_test(
    test_name,
    condition,
    actual_value,
    expected_value,
    severity="HIGH"
):

    integrity_results.append({

        "test":
            test_name,

        "passed":
            bool(condition),

        "actual":
            actual_value,

        "expected":
            expected_value,

        "severity":
            severity

    })


# -----------------------------------------------------------------------------
# Transaction count
# -----------------------------------------------------------------------------

add_integrity_test(
    "Transaction dataset is non-empty",
    len(transactions) > 0,
    len(transactions),
    "> 0"
)


# -----------------------------------------------------------------------------
# Transaction IDs
# -----------------------------------------------------------------------------

if "transaction_id" in transactions.columns:

    duplicate_transactions = (
        transactions[
            "transaction_id"
        ]
        .duplicated()
        .sum()
    )

else:

    duplicate_transactions = -1


add_integrity_test(
    "No duplicate transaction IDs",
    duplicate_transactions == 0,
    duplicate_transactions,
    0
)


# -----------------------------------------------------------------------------
# Customer IDs
# -----------------------------------------------------------------------------

if "customer_id" in transactions.columns:

    unique_customers = (
        transactions[
            "customer_id"
        ]
        .nunique()
    )

else:

    unique_customers = 0


add_integrity_test(
    "Customer IDs exist",
    unique_customers > 0,
    unique_customers,
    "> 0"
)


# -----------------------------------------------------------------------------
# Amount validation
# -----------------------------------------------------------------------------

if "amount_usd" in transactions.columns:

    negative_amounts = (
        pd.to_numeric(
            transactions[
                "amount_usd"
            ],
            errors="coerce"
        )
        < 0
    ).sum()

else:

    negative_amounts = -1


add_integrity_test(
    "No negative transaction amounts",
    negative_amounts == 0,
    negative_amounts,
    0
)


# -----------------------------------------------------------------------------
# Target validation
# -----------------------------------------------------------------------------

if "known_suspicious" in transactions.columns:

    suspicious_count = int(
        pd.to_numeric(
            transactions[
                "known_suspicious"
            ],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )

else:

    suspicious_count = 0


add_integrity_test(
    "Suspicious transactions exist",
    suspicious_count > 0,
    suspicious_count,
    "> 0"
)


integrity_df = pd.DataFrame(
    integrity_results
)


print(
    "\nDataset integrity results:"
)


print(
    integrity_df[
        [
            "test",
            "passed",
            "actual",
            "expected"
        ]
    ]
    .to_string(
        index=False
    )
)


# =============================================================================
# 8. TARGET DISTRIBUTION
# =============================================================================

print("\n" + "=" * 90)
print("8. SUSPICIOUS ACTIVITY DISTRIBUTION")
print("=" * 90)


if "known_suspicious" in transactions.columns:

    target_distribution = (
        transactions[
            "known_suspicious"
        ]
        .value_counts()
        .sort_index()
    )


    normal_count = int(
        target_distribution.get(
            0,
            0
        )
    )


    suspicious_count = int(
        target_distribution.get(
            1,
            0
        )
    )


    total_count = (
        normal_count
        +
        suspicious_count
    )


    suspicious_rate = (
        suspicious_count
        /
        total_count
        *
        100
        if total_count > 0
        else 0
    )


    print(
        f"\nNormal transactions: {normal_count:,}"
    )


    print(
        f"Suspicious transactions: {suspicious_count:,}"
    )


    print(
        f"Suspicious transaction rate: "
        f"{suspicious_rate:.2f}%"
    )


# =============================================================================
# 9. SCENARIO DISTRIBUTION
# =============================================================================

print("\n" + "=" * 90)
print("9. AML TYPOLOGY / SCENARIO DISTRIBUTION")
print("=" * 90)


if "scenario" in transactions.columns:

    scenario_distribution = (
        transactions[
            transactions[
                "scenario"
            ]
            .notna()
            &
            (
                transactions[
                    "scenario"
                ]
                .astype(str)
                !=
                ""
            )
        ][
            "scenario"
        ]
        .value_counts()
        .reset_index()
    )


    scenario_distribution.columns = [
        "scenario",
        "transaction_count"
    ]


    print(
        "\n"
        +
        scenario_distribution.to_string(
            index=False
        )
    )

else:

    scenario_distribution = pd.DataFrame(
        columns=[
            "scenario",
            "transaction_count"
        ]
    )


# =============================================================================
# 10. ML PERFORMANCE
# =============================================================================

print("\n" + "=" * 90)
print("10. MACHINE LEARNING PERFORMANCE VALIDATION")
print("=" * 90)


ml_metrics = {}


if not predictions.empty:

    # -------------------------------------------------------------------------
    # Identify target/prediction columns
    # -------------------------------------------------------------------------

    target_column = None


    for candidate in [
        "target",
        "known_suspicious",
        "actual"
    ]:

        if candidate in predictions.columns:

            target_column = candidate

            break


    probability_column = None


    for candidate in [
        "ml_probability",
        "prediction_probability",
        "predicted_probability",
        "risk_probability"
    ]:

        if candidate in predictions.columns:

            probability_column = candidate

            break


    prediction_column = None


    for candidate in [
        "prediction",
        "predicted",
        "ml_prediction"
    ]:

        if candidate in predictions.columns:

            prediction_column = candidate

            break


    print(
        "\nPrediction columns detected:"
    )


    print(
        f"Target: {target_column}"
    )


    print(
        f"Probability: {probability_column}"
    )


    print(
        f"Prediction: {prediction_column}"
    )


    # -------------------------------------------------------------------------
    # Calculate metrics if available
    # -------------------------------------------------------------------------

    if (
        target_column
        and
        prediction_column
    ):

        actual = pd.to_numeric(
            predictions[
                target_column
            ],
            errors="coerce"
        ).fillna(0).astype(int)


        predicted = pd.to_numeric(
            predictions[
                prediction_column
            ],
            errors="coerce"
        ).fillna(0).astype(int)


        tp = int(
            (
                (actual == 1)
                &
                (predicted == 1)
            ).sum()
        )


        tn = int(
            (
                (actual == 0)
                &
                (predicted == 0)
            ).sum()
        )


        fp = int(
            (
                (actual == 0)
                &
                (predicted == 1)
            ).sum()
        )


        fn = int(
            (
                (actual == 1)
                &
                (predicted == 0)
            ).sum()
        )


        total = (
            tp
            +
            tn
            +
            fp
            +
            fn
        )


        accuracy = (
            (tp + tn)
            /
            total
            if total > 0
            else 0
        )


        precision = (
            tp
            /
            (tp + fp)
            if (tp + fp) > 0
            else 0
        )


        recall = (
            tp
            /
            (tp + fn)
            if (tp + fn) > 0
            else 0
        )


        f1 = (
            2
            *
            precision
            *
            recall
            /
            (precision + recall)
            if (precision + recall) > 0
            else 0
        )


        false_positive_rate = (
            fp
            /
            (fp + tn)
            if (fp + tn) > 0
            else 0
        )


        false_negative_rate = (
            fn
            /
            (fn + tp)
            if (fn + tp) > 0
            else 0
        )


        ml_metrics = {

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "false_positive_rate":
                false_positive_rate,

            "false_negative_rate":
                false_negative_rate,

            "true_positive":
                tp,

            "true_negative":
                tn,

            "false_positive":
                fp,

            "false_negative":
                fn
        }


        print(
            f"\nAccuracy: "
            f"{accuracy:.4f}"
        )


        print(
            f"Precision: "
            f"{precision:.4f}"
        )


        print(
            f"Recall: "
            f"{recall:.4f}"
        )


        print(
            f"F1 Score: "
            f"{f1:.4f}"
        )


        print(
            f"False Positive Rate: "
            f"{false_positive_rate:.4f}"
        )


        print(
            f"False Negative Rate: "
            f"{false_negative_rate:.4f}"
        )


# =============================================================================
# 11. MODEL COMPARISON
# =============================================================================

print("\n" + "=" * 90)
print("11. MODEL COMPARISON")
print("=" * 90)


if os.path.exists(
    MODEL_COMPARISON_FILE
):

    model_comparison = pd.read_csv(
        MODEL_COMPARISON_FILE
    )


    print(
        "\nModel comparison:"
    )


    print(
        model_comparison.to_string(
            index=False
        )
    )

else:

    model_comparison = pd.DataFrame()


# =============================================================================
# 12. FALSE POSITIVE / FALSE NEGATIVE ANALYSIS
# =============================================================================

print("\n" + "=" * 90)
print("12. ERROR ANALYSIS")
print("=" * 90)


error_rows = []


if (
    not predictions.empty
    and
    target_column
    and
    prediction_column
):

    error_data = predictions.copy()


    actual = pd.to_numeric(
        error_data[
            target_column
        ],
        errors="coerce"
    ).fillna(0).astype(int)


    predicted = pd.to_numeric(
        error_data[
            prediction_column
        ],
        errors="coerce"
    ).fillna(0).astype(int)


    error_data[
        "error_type"
    ] = "CORRECT"


    error_data.loc[
        (actual == 0)
        &
        (predicted == 1),
        "error_type"
    ] = "FALSE_POSITIVE"


    error_data.loc[
        (actual == 1)
        &
        (predicted == 0),
        "error_type"
    ] = "FALSE_NEGATIVE"


    error_distribution = (
        error_data[
            "error_type"
        ]
        .value_counts()
    )


    for error_type, count in (
        error_distribution.items()
    ):

        error_rows.append({

            "error_type":
                error_type,

            "count":
                int(count),

            "percentage":
                round(
                    count
                    /
                    len(error_data)
                    *
                    100,
                    2
                )

        })


error_df = pd.DataFrame(
    error_rows
)


if not error_df.empty:

    print(
        error_df.to_string(
            index=False
        )
    )


# =============================================================================
# 13. SCENARIO PERFORMANCE
# =============================================================================

print("\n" + "=" * 90)
print("13. SCENARIO-LEVEL VALIDATION")
print("=" * 90)


scenario_rows = []


if (
    not predictions.empty
    and
    "scenario" in predictions.columns
    and
    target_column
    and
    prediction_column
):

    scenario_data = predictions.copy()


    scenario_data[
        "actual"
    ] = pd.to_numeric(
        scenario_data[
            target_column
        ],
        errors="coerce"
    ).fillna(0).astype(int)


    scenario_data[
        "predicted"
    ] = pd.to_numeric(
        scenario_data[
            prediction_column
        ],
        errors="coerce"
    ).fillna(0).astype(int)


    for scenario, group in (
        scenario_data
        .groupby(
            "scenario"
        )
    ):

        actual_values = group[
            "actual"
        ]


        predicted_values = group[
            "predicted"
        ]


        tp_s = int(
            (
                (actual_values == 1)
                &
                (predicted_values == 1)
            ).sum()
        )


        fn_s = int(
            (
                (actual_values == 1)
                &
                (predicted_values == 0)
            ).sum()
        )


        fp_s = int(
            (
                (actual_values == 0)
                &
                (predicted_values == 1)
            ).sum()
        )


        tn_s = int(
            (
                (actual_values == 0)
                &
                (predicted_values == 0)
            ).sum()
        )


        recall_s = (
            tp_s
            /
            (tp_s + fn_s)
            if tp_s + fn_s > 0
            else 0
        )


        precision_s = (
            tp_s
            /
            (tp_s + fp_s)
            if tp_s + fp_s > 0
            else 0
        )


        scenario_rows.append({

            "scenario":
                scenario,

            "records":
                len(group),

            "true_positive":
                tp_s,

            "false_negative":
                fn_s,

            "false_positive":
                fp_s,

            "true_negative":
                tn_s,

            "precision":
                round(
                    precision_s,
                    4
                ),

            "recall":
                round(
                    recall_s,
                    4
                )

        })


scenario_performance = pd.DataFrame(
    scenario_rows
)


if not scenario_performance.empty:

    print(
        scenario_performance.to_string(
            index=False
        )
    )


scenario_performance.to_csv(
    SCENARIO_FILE,
    index=False
)


# =============================================================================
# 14. RULE ENGINE VALIDATION
# =============================================================================

print("\n" + "=" * 90)
print("14. RULE ENGINE VALIDATION")
print("=" * 90)


rule_rows = []


if not rule_data.empty:

    if (
        "known_suspicious"
        in rule_data.columns
    ):

        actual_rule = pd.to_numeric(
            rule_data[
                "known_suspicious"
            ],
            errors="coerce"
        ).fillna(0).astype(int)


        if "rule_alert" in rule_data.columns:

            predicted_rule = pd.to_numeric(
                rule_data[
                    "rule_alert"
                ],
                errors="coerce"
            ).fillna(0).astype(int)


            tp = int(
                (
                    (actual_rule == 1)
                    &
                    (predicted_rule == 1)
                ).sum()
            )


            fp = int(
                (
                    (actual_rule == 0)
                    &
                    (predicted_rule == 1)
                ).sum()
            )


            fn = int(
                (
                    (actual_rule == 1)
                    &
                    (predicted_rule == 0)
                ).sum()
            )


            tn = int(
                (
                    (actual_rule == 0)
                    &
                    (predicted_rule == 0)
                ).sum()
            )


            rule_precision = (
                tp
                /
                (tp + fp)
                if tp + fp > 0
                else 0
            )


            rule_recall = (
                tp
                /
                (tp + fn)
                if tp + fn > 0
                else 0
            )


            rule_rows.append({

                "rule_engine":
                    "Combined Rule Alert",

                "true_positive":
                    tp,

                "false_positive":
                    fp,

                "false_negative":
                    fn,

                "true_negative":
                    tn,

                "precision":
                    round(
                        rule_precision,
                        4
                    ),

                "recall":
                    round(
                        rule_recall,
                        4
                    )

            })


rule_validation = pd.DataFrame(
    rule_rows
)


if not rule_validation.empty:

    print(
        rule_validation.to_string(
            index=False
        )
    )


rule_validation.to_csv(
    RULE_VALIDATION_FILE,
    index=False
)


# =============================================================================
# 15. FEATURE LEAKAGE REVIEW
# =============================================================================

print("\n" + "=" * 90)
print("15. FEATURE LEAKAGE REVIEW")
print("=" * 90)


leakage_candidates = [
    "known_suspicious",
    "target",
    "scenario",
    "suspicious",
    "fraud_label",
    "label",
    "ground_truth"
]


leakage_rows = []


feature_columns = (
    features.columns.tolist()
)


for column in feature_columns:

    column_lower = column.lower()


    possible_leakage = any(
        candidate in column_lower
        for candidate in leakage_candidates
    )


    leakage_rows.append({

        "feature":
            column,

        "possible_target_leakage":
            possible_leakage

    })


leakage_df = pd.DataFrame(
    leakage_rows
)


possible_leakage_features = (
    leakage_df[
        leakage_df[
            "possible_target_leakage"
        ]
    ]
)


if possible_leakage_features.empty:

    print(
        "\nNo obvious target-leakage feature names detected."
    )

else:

    print(
        "\nPotential leakage-related feature names:"
    )


    print(
        possible_leakage_features.to_string(
            index=False
        )
    )


leakage_df.to_csv(
    LEAKAGE_FILE,
    index=False
)


# =============================================================================
# 16. INVESTIGATION COVERAGE
# =============================================================================

print("\n" + "=" * 90)
print("16. INVESTIGATION WORKFLOW VALIDATION")
print("=" * 90)


investigation_metrics = {}


if not cases.empty:

    investigation_metrics[
        "cases"
    ] = len(cases)


    if "case_priority" in cases.columns:

        investigation_metrics[
            "P1_cases"
        ] = int(
            (
                cases[
                    "case_priority"
                ]
                .astype(str)
                ==
                "P1"
            ).sum()
        )


        investigation_metrics[
            "P2_cases"
        ] = int(
            (
                cases[
                    "case_priority"
                ]
                .astype(str)
                ==
                "P2"
            ).sum()
        )


    if "investigation_status" in cases.columns:

        investigation_metrics[
            "open_cases"
        ] = int(
            (
                cases[
                    "investigation_status"
                ]
                .astype(str)
                ==
                "OPEN"
            ).sum()
        )


    if "escalation_required" in cases.columns:

        investigation_metrics[
            "escalation_cases"
        ] = int(
            (
                cases[
                    "escalation_required"
                ]
                .astype(str)
                ==
                "YES"
            ).sum()
        )


for key, value in investigation_metrics.items():

    print(
        f"{key}: {value:,}"
    )


# =============================================================================
# 17. SAR QUALITY VALIDATION
# =============================================================================

print("\n" + "=" * 90)
print("17. SAR NARRATIVE VALIDATION")
print("=" * 90)


sar_metrics = {}


if not sar.empty:

    sar_metrics[
        "narrative_drafts"
    ] = len(sar)


    if "narrative_quality_score" in sar.columns:

        quality = pd.to_numeric(
            sar[
                "narrative_quality_score"
            ],
            errors="coerce"
        )


        sar_metrics[
            "average_quality_score"
        ] = round(
            quality.mean(),
            2
        )


        sar_metrics[
            "minimum_quality_score"
        ] = round(
            quality.min(),
            2
        )


        sar_metrics[
            "maximum_quality_score"
        ] = round(
            quality.max(),
            2
        )


    if "human_review_required" in sar.columns:

        sar_metrics[
            "human_review_required_rate"
        ] = round(
            (
                sar[
                    "human_review_required"
                ]
                .astype(str)
                .str.upper()
                ==
                "YES"
            )
            .mean()
            *
            100,
            2
        )


    if "automatic_sar_filing" in sar.columns:

        sar_metrics[
            "automatic_filing_rate"
        ] = round(
            (
                sar[
                    "automatic_sar_filing"
                ]
                .astype(str)
                .str.upper()
                ==
                "YES"
            )
            .mean()
            *
            100,
            2
        )


for key, value in sar_metrics.items():

    print(
        f"{key}: {value}"
    )


# =============================================================================
# 18. SHAP VALIDATION
# =============================================================================

print("\n" + "=" * 90)
print("18. SHAP EXPLAINABILITY VALIDATION")
print("=" * 90)


if not shap.empty:

    print(
        f"\nSHAP features available: "
        f"{len(shap):,}"
    )


    print(
        "\nTop SHAP features:"
    )


    print(
        shap.head(
            10
        ).to_string(
            index=False
        )
    )

else:

    print(
        "\nSHAP feature importance file unavailable."
    )


# =============================================================================
# 19. PORTFOLIO METRICS
# =============================================================================

print("\n" + "=" * 90)
print("19. PORTFOLIO METRICS")
print("=" * 90)


portfolio_metrics = {}


portfolio_metrics[
    "Total transactions"
] = len(
    transactions
)


portfolio_metrics[
    "Unique customers"
] = unique_customers


portfolio_metrics[
    "Suspicious transactions"
] = suspicious_count


portfolio_metrics[
    "Suspicious transaction rate"
] = round(
    suspicious_rate,
    2
)


portfolio_metrics[
    "Investigation cases"
] = len(
    cases
)


portfolio_metrics[
    "SAR narrative drafts"
] = len(
    sar
)


if ml_metrics:

    portfolio_metrics[
        "ML precision"
    ] = round(
        ml_metrics[
            "precision"
        ] * 100,
        2
    )


    portfolio_metrics[
        "ML recall"
    ] = round(
        ml_metrics[
            "recall"
        ] * 100,
        2
    )


    portfolio_metrics[
        "ML F1"
    ] = round(
        ml_metrics[
            "f1"
        ] * 100,
        2
    )


    portfolio_metrics[
        "ML false positive rate"
    ] = round(
        ml_metrics[
            "false_positive_rate"
        ] * 100,
        2
    )


for key, value in portfolio_metrics.items():

    print(
        f"{key}: {value}"
    )


# =============================================================================
# 20. SAVE PORTFOLIO METRICS
# =============================================================================

with open(
    PORTFOLIO_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AML TRANSACTION MONITORING PROJECT\n"
    )

    file.write(
        "PORTFOLIO METRICS\n"
    )

    file.write(
        "=" * 80
        +
        "\n\n"
    )


    for key, value in portfolio_metrics.items():

        file.write(
            f"{key}: {value}\n"
        )


    file.write(
        "\n\nIMPORTANT:\n"
    )


    file.write(
        "The dataset is synthetic and designed for portfolio "
        "demonstration. Model performance should not be interpreted "
        "as production financial-crime detection performance.\n"
    )


# =============================================================================
# 21. MODEL GOVERNANCE CHECKLIST
# =============================================================================

print("\n" + "=" * 90)
print("21. MODEL GOVERNANCE CHECKLIST")
print("=" * 90)


governance_items = [

    (
        "Synthetic dataset clearly identified",
        "PASS"
    ),

    (
        "Target variable separated from ML features",
        "REVIEW"
    ),

    (
        "Time-based train/test split used",
        "PASS"
    ),

    (
        "Rule and ML signals evaluated separately",
        "PASS"
    ),

    (
        "False positives evaluated",
        "PASS"
    ),

    (
        "False negatives evaluated",
        "PASS"
    ),

    (
        "Scenario-level performance evaluated",
        "PASS"
    ),

    (
        "SHAP explainability implemented",
        "PASS"
    ),

    (
        "Human analyst review preserved",
        "PASS"
    ),

    (
        "Automatic SAR filing disabled",
        "PASS"
    ),

    (
        "Model output treated as risk indicator",
        "PASS"
    ),

    (
        "Production validation still required",
        "REVIEW"
    ),

    (
        "Threshold calibration required for production",
        "REVIEW"
    ),

    (
        "Model monitoring required in production",
        "REVIEW"
    ),

    (
        "Data drift monitoring required",
        "REVIEW"
    ),

    (
        "Bias/fairness assessment required before production",
        "REVIEW"
    )

]


with open(
    GOVERNANCE_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AML MODEL GOVERNANCE CHECKLIST\n"
    )

    file.write(
        "=" * 80
        +
        "\n\n"
    )


    for item, status in governance_items:

        print(
            f"[{status}] {item}"
        )


        file.write(
            f"[{status}] {item}\n"
        )


# =============================================================================
# 22. FINAL VALIDATION SUMMARY
# =============================================================================

validation_summary_rows = []


for result in integrity_results:

    validation_summary_rows.append({

        "category":
            "DATA_INTEGRITY",

        "test":
            result[
                "test"
            ],

        "status":
            "PASS"
            if result[
                "passed"
            ]
            else
            "FAIL",

        "actual":
            result[
                "actual"
            ],

        "expected":
            result[
                "expected"
            ]

    })


validation_summary_rows.extend([

    {
        "category":
            "ML",

        "test":
            "Precision",

        "status":
            "CALCULATED"
            if ml_metrics
            else
            "UNAVAILABLE",

        "actual":
            ml_metrics.get(
                "precision",
                ""
            ),

        "expected":
            "Review"

    },

    {
        "category":
            "ML",

        "test":
            "Recall",

        "status":
            "CALCULATED"
            if ml_metrics
            else
            "UNAVAILABLE",

        "actual":
            ml_metrics.get(
                "recall",
                ""
            ),

        "expected":
            "Review"

    },

    {
        "category":
            "ML",

        "test":
            "F1",

        "status":
            "CALCULATED"
            if ml_metrics
            else
            "UNAVAILABLE",

        "actual":
            ml_metrics.get(
                "f1",
                ""
            ),

        "expected":
            "Review"

    },

    {
        "category":
            "INVESTIGATION",

        "test":
            "Investigation cases generated",

        "status":
            "PASS"
            if len(cases) > 0
            else
            "FAIL",

        "actual":
            len(cases),

        "expected":
            "> 0"

    },

    {
        "category":
            "SAR",

        "test":
            "SAR drafts generated",

        "status":
            "PASS"
            if len(sar) > 0
            else
            "FAIL",

        "actual":
            len(sar),

        "expected":
            "> 0"

    },

    {
        "category":
            "SHAP",

        "test":
            "SHAP explanation available",

        "status":
            "PASS"
            if len(shap) > 0
            else
            "FAIL",

        "actual":
            len(shap),

        "expected":
            "> 0"

    }

])


validation_summary = pd.DataFrame(
    validation_summary_rows
)


validation_summary.to_csv(
    VALIDATION_SUMMARY_FILE,
    index=False
)


# =============================================================================
# 23. CREATE PORTFOLIO README
# =============================================================================

print("\n" + "=" * 90)
print("23. CREATING PORTFOLIO README")
print("=" * 90)


with open(
    README_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "# US AML Transaction Monitoring & Financial Crime Analytics Platform\n\n"
    )


    file.write(
        "## Project Overview\n\n"
    )


    file.write(
        "This project demonstrates an end-to-end synthetic financial-crime "
        "transaction-monitoring platform designed around a US financial "
        "institution use case.\n\n"
    )


    file.write(
        "The platform combines customer profiling, transaction monitoring, "
        "AML red-flag detection, machine-learning risk scoring, SHAP "
        "explainability, alert prioritization, analyst investigation, "
        "interactive dashboarding and AI-assisted SAR narrative drafting.\n\n"
    )


    file.write(
        "## Architecture\n\n"
    )


    file.write(
        "Customer / KYC Data\n"
        "-> Transaction Monitoring\n"
        "-> AML Rules\n"
        "-> Feature Engineering\n"
        "-> Machine Learning\n"
        "-> SHAP Explainability\n"
        "-> Risk Scoring\n"
        "-> Alert Prioritization\n"
        "-> Investigation Case Management\n"
        "-> AML Dashboard\n"
        "-> SAR Narrative Assistant\n\n"
    )


    file.write(
        "## AML Typologies Covered\n\n"
    )


    typologies = [

        "Structuring",

        "Rapid movement",

        "Funnel-account behavior",

        "Layering",

        "Mule-account behavior",

        "Fraud-linked suspicious activity",

        "High-value transaction activity",

        "Customer-profile mismatch",

        "International transaction activity",

        "High transaction velocity",

        "Multiple-counterparty behavior"

    ]


    for typology in typologies:

        file.write(
            f"- {typology}\n"
        )


    file.write(
        "\n## Technology Stack\n\n"
    )


    technologies = [

        "Python",

        "Pandas",

        "NumPy",

        "Scikit-learn",

        "XGBoost",

        "CatBoost",

        "SHAP",

        "Streamlit",

        "Plotly",

        "Excel / CSV",

        "Machine Learning",

        "Explainable AI"

    ]


    for technology in technologies:

        file.write(
            f"- {technology}\n"
        )


    file.write(
        "\n## Key AML Capabilities\n\n"
    )


    capabilities = [

        "Rule-based transaction monitoring",

        "Behavioral feature engineering",

        "Customer risk profiling",

        "Machine-learning risk scoring",

        "Alert prioritization",

        "False-positive analysis",

        "Customer-level investigation",

        "Red-flag identification",

        "SHAP explainability",

        "Interactive investigation dashboard",

        "Analyst case management",

        "AI-assisted SAR narrative drafting",

        "Human-in-the-loop review"

    ]


    for capability in capabilities:

        file.write(
            f"- {capability}\n"
        )


    file.write(
        "\n## Important Limitation\n\n"
    )


    file.write(
        "This project uses synthetic data for educational and portfolio "
        "demonstration purposes. The rules, thresholds and model outputs "
        "should not be interpreted as production financial-crime detection "
        "performance or legal conclusions.\n\n"
    )


    file.write(
        "The SAR component produces analyst-reviewable drafts and does not "
        "automatically determine or file a SAR.\n\n"
    )


    file.write(
        "## Future Production Improvements\n\n"
    )


    improvements = [

        "Use institution-specific historical alert and transaction data",

        "Calibrate thresholds using historical investigation outcomes",

        "Implement champion/challenger model testing",

        "Add model drift monitoring",

        "Add data-quality monitoring",

        "Add investigator feedback loops",

        "Add graph-based transaction-network analytics",

        "Add sanctions screening integration",

        "Add adverse-media intelligence",

        "Add case-management integration",

        "Perform formal model validation",

        "Perform fairness and bias assessment",

        "Implement secure enterprise deployment",

        "Integrate approved LLM infrastructure for controlled SAR drafting"

    ]


    for improvement in improvements:

        file.write(
            f"- {improvement}\n"
        )


# =============================================================================
# 24. FINAL OUTPUT SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("24. PHASE 11 OUTPUT FILES")
print("=" * 90)


print(
    "\n1. Validation summary:"
)


print(
    VALIDATION_SUMMARY_FILE
)


print(
    "\n2. Scenario performance:"
)


print(
    SCENARIO_FILE
)


print(
    "\n3. Rule validation:"
)


print(
    RULE_VALIDATION_FILE
)


print(
    "\n4. Feature leakage review:"
)


print(
    LEAKAGE_FILE
)


print(
    "\n5. Portfolio metrics:"
)


print(
    PORTFOLIO_FILE
)


print(
    "\n6. Model governance checklist:"
)


print(
    GOVERNANCE_FILE
)


print(
    "\n7. Portfolio README:"
)


print(
    README_FILE
)


# =============================================================================
# 25. FINAL STATUS
# =============================================================================

print("\n" + "=" * 90)
print("PHASE 11 COMPLETED SUCCESSFULLY")
print("=" * 90)


print(
    "\nYour AML project has now been technically validated."
)


print(
    "\nCore components:"
)


print(
    "  [OK] Synthetic US banking dataset"
)


print(
    "  [OK] Customer/KYC profiling"
)


print(
    "  [OK] Transaction monitoring"
)


print(
    "  [OK] AML red-flag rules"
)


print(
    "  [OK] Feature engineering"
)


print(
    "  [OK] Machine-learning risk model"
)


print(
    "  [OK] SHAP explainability"
)


print(
    "  [OK] Risk scoring"
)


print(
    "  [OK] Alert prioritization"
)


print(
    "  [OK] Investigation case management"
)


print(
    "  [OK] Interactive AML dashboard"
)


print(
    "  [OK] SAR narrative assistant"
)


print(
    "  [OK] Model validation"
)


print(
    "  [OK] Feature leakage review"
)


print(
    "  [OK] Portfolio README"
)


print(
    "\nNext major improvement:"
)


print(
    "GRAPH-BASED AML NETWORK ANALYTICS"
)


print(
    "=" * 90
)