# =============================================================================
# PHASE 7 - AML RISK SCORING & ALERT PRIORITIZATION
# US FINANCIAL CRIME / AML TRANSACTION MONITORING PROJECT
# =============================================================================
#
# Purpose:
#   Combine:
#       1. Rule-engine risk score
#       2. ML risk probability
#       3. Customer risk
#       4. PEP / sanctions / adverse media indicators
#       5. SHAP model explanations
#
#   into an analyst-friendly AML alert score and investigation queue.
#
# IMPORTANT:
#   This score is a risk-prioritization mechanism.
#   It is NOT proof of money laundering or fraud.
#   It does NOT automatically determine whether a SAR should be filed.
#
# =============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "aml_ml_features.csv"
)


RULE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "aml_rule_engine_transactions.csv"
)


SHAP_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "shap",
    "transaction_shap_explanations.csv"
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)


REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "alerts"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# =============================================================================
# 2. FILE PATHS
# =============================================================================

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "aml_alerts_scored.csv"
)


CUSTOMER_ALERT_FILE = os.path.join(
    OUTPUT_DIR,
    "customer_alert_risk_summary.csv"
)


ALERT_SUMMARY_FILE = os.path.join(
    REPORT_DIR,
    "alert_summary.csv"
)


PRIORITY_FILE = os.path.join(
    REPORT_DIR,
    "top_priority_alerts.csv"
)


INVESTIGATION_QUEUE_FILE = os.path.join(
    REPORT_DIR,
    "analyst_investigation_queue.csv"
)


DRIVER_SUMMARY_FILE = os.path.join(
    REPORT_DIR,
    "risk_driver_summary.csv"
)


CONFIG_FILE = os.path.join(
    REPORT_DIR,
    "risk_scoring_configuration.txt"
)


# =============================================================================
# 3. HEADER
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 7 - AML RISK SCORING & ALERT PRIORITIZATION")
print("=" * 80)


# =============================================================================
# 4. LOAD MAIN DATASET
# =============================================================================

print("\n" + "=" * 80)
print("4. LOADING ML FEATURE DATASET")
print("=" * 80)


if not os.path.exists(INPUT_FILE):

    print("\nERROR: Input dataset not found.")

    print(INPUT_FILE)

    sys.exit(1)


df = pd.read_csv(
    INPUT_FILE
)


print(
    f"\nDataset shape: {df.shape}"
)


# =============================================================================
# 5. LOAD RULE ENGINE DATA
# =============================================================================

print("\n" + "=" * 80)
print("5. LOADING RULE ENGINE OUTPUT")
print("=" * 80)


if os.path.exists(RULE_FILE):

    rule_df = pd.read_csv(
        RULE_FILE
    )

    print(
        f"Rule dataset shape: {rule_df.shape}"
    )

else:

    print(
        "\nWARNING: Rule engine file not found."
    )

    print(
        "The script will attempt to use rule columns "
        "already present in the ML dataset."
    )

    rule_df = None


# =============================================================================
# 6. LOAD SHAP EXPLANATIONS
# =============================================================================

print("\n" + "=" * 80)
print("6. LOADING SHAP EXPLANATIONS")
print("=" * 80)


if os.path.exists(SHAP_FILE):

    shap_df = pd.read_csv(
        SHAP_FILE
    )

    print(
        f"SHAP dataset shape: {shap_df.shape}"
    )

else:

    print(
        "\nWARNING: SHAP transaction explanation file not found."
    )

    print(
        "The scoring process will continue without SHAP drivers."
    )

    shap_df = None


# =============================================================================
# 7. BASIC CLEANING
# =============================================================================

print("\n" + "=" * 80)
print("7. CLEANING INPUT DATA")
print("=" * 80)


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


if "timestamp" in df.columns:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )


# =============================================================================
# 8. MERGE RULE ENGINE OUTPUT IF NECESSARY
# =============================================================================

print("\n" + "=" * 80)
print("8. PREPARING RULE RISK SCORE")
print("=" * 80)


# -------------------------------------------------------------------------
# First preference:
# use rule_risk_score already available in the ML dataset.
# -------------------------------------------------------------------------

if "rule_risk_score" in df.columns:

    print(
        "\nRule risk score found in ML dataset."
    )


# -------------------------------------------------------------------------
# Otherwise merge from rule-engine dataset.
# -------------------------------------------------------------------------

elif rule_df is not None and "rule_risk_score" in rule_df.columns:

    print(
        "\nMerging rule risk score from rule engine output."
    )


    if (
        "transaction_id" in df.columns
        and
        "transaction_id" in rule_df.columns
    ):

        rule_columns = [
            "transaction_id",
            "rule_risk_score"
        ]


        if "rule_alert" in rule_df.columns:

            rule_columns.append(
                "rule_alert"
            )


        if "risk_level" in rule_df.columns:

            rule_columns.append(
                "risk_level"
            )


        df = df.merge(
            rule_df[
                rule_columns
            ],
            on="transaction_id",
            how="left"
        )


    else:

        print(
            "\nWARNING: transaction_id unavailable."
        )

else:

    print(
        "\nWARNING: No rule risk score found."
    )

    df["rule_risk_score"] = 0


# Ensure numeric
df["rule_risk_score"] = pd.to_numeric(
    df["rule_risk_score"],
    errors="coerce"
).fillna(0)


df["rule_risk_score"] = (
    df["rule_risk_score"]
    .clip(0, 100)
)


# =============================================================================
# 9. PREPARE ML RISK SCORE
# =============================================================================

print("\n" + "=" * 80)
print("9. PREPARING ML RISK SCORE")
print("=" * 80)


# -------------------------------------------------------------------------
# Prefer ML probability if available.
# -------------------------------------------------------------------------

if "ml_risk_probability" in df.columns:

    print(
        "\nML probability already exists in dataset."
    )


elif "ml_probability" in df.columns:

    df["ml_risk_probability"] = pd.to_numeric(
        df["ml_probability"],
        errors="coerce"
    ).fillna(0)


else:

    # ---------------------------------------------------------------------
    # Phase 5 predictions may exist in a separate file.
    # ---------------------------------------------------------------------

    prediction_file = os.path.join(
        PROJECT_ROOT,
        "reports",
        "models",
        "aml_test_predictions.csv"
    )


    if os.path.exists(prediction_file):

        print(
            "\nLoading ML predictions from:"
        )

        print(
            prediction_file
        )


        predictions = pd.read_csv(
            prediction_file
        )


        # -------------------------------------------------------------
        # Determine probability column
        # -------------------------------------------------------------

        probability_column = None


        for column in [
            "ml_probability",
            "probability",
            "predicted_probability",
            "risk_probability",
            "prediction_probability"
        ]:

            if column in predictions.columns:

                probability_column = column

                break


        if probability_column is not None:

            if (
                "transaction_id" in df.columns
                and
                "transaction_id" in predictions.columns
            ):

                prediction_subset = predictions[
                    [
                        "transaction_id",
                        probability_column
                    ]
                ].copy()


                prediction_subset = (
                    prediction_subset
                    .drop_duplicates(
                        subset=[
                            "transaction_id"
                        ]
                    )
                )


                prediction_subset = (
                    prediction_subset
                    .rename(
                        columns={
                            probability_column:
                                "ml_risk_probability"
                        }
                    )
                )


                df = df.merge(
                    prediction_subset,
                    on="transaction_id",
                    how="left"
                )


        else:

            print(
                "\nWARNING: No ML probability column found."
            )

            df["ml_risk_probability"] = 0


    else:

        print(
            "\nWARNING: ML prediction file not found."
        )

        df["ml_risk_probability"] = 0


df["ml_risk_probability"] = pd.to_numeric(
    df["ml_risk_probability"],
    errors="coerce"
).fillna(0)


# -------------------------------------------------------------------------
# Safety:
# If probability is accidentally stored as percentage, convert it.
# -------------------------------------------------------------------------

if df["ml_risk_probability"].max() > 1:

    df["ml_risk_probability"] = (
        df["ml_risk_probability"] / 100
    )


df["ml_risk_probability"] = (
    df["ml_risk_probability"]
    .clip(0, 1)
)


df["ml_risk_score"] = (
    df["ml_risk_probability"] * 100
)


print(
    "\nML risk score statistics:"
)

print(
    df["ml_risk_score"].describe()
)


# =============================================================================
# 10. CUSTOMER RISK COMPONENT
# =============================================================================

print("\n" + "=" * 80)
print("10. CALCULATING CUSTOMER RISK COMPONENT")
print("=" * 80)


# -------------------------------------------------------------------------
# Existing numeric customer risk
# -------------------------------------------------------------------------

if "customer_risk_numeric" in df.columns:

    customer_risk_numeric = pd.to_numeric(
        df["customer_risk_numeric"],
        errors="coerce"
    ).fillna(0)


# -------------------------------------------------------------------------
# Convert customer risk text if available
# -------------------------------------------------------------------------

elif "risk_rating" in df.columns:

    risk_map = {
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2,
        "CRITICAL": 3
    }


    customer_risk_numeric = (
        df["risk_rating"]
        .astype(str)
        .str.upper()
        .map(risk_map)
        .fillna(0)
    )


# -------------------------------------------------------------------------
# Fallback
# -------------------------------------------------------------------------

else:

    customer_risk_numeric = pd.Series(
        0,
        index=df.index
    )


# Convert 0-3 into 0-100
df["customer_risk_score"] = (
    customer_risk_numeric / 3
) * 100


df["customer_risk_score"] = (
    df["customer_risk_score"]
    .clip(0, 100)
)


print(
    "\nCustomer risk score distribution:"
)

print(
    df["customer_risk_score"].describe()
)


# =============================================================================
# 11. PEP / SANCTIONS / ADVERSE MEDIA COMPONENT
# =============================================================================

print("\n" + "=" * 80)
print("11. CALCULATING CUSTOMER DUE-DILIGENCE RISK")
print("=" * 80)


def get_binary_column(
    dataframe,
    column_name
):

    if column_name not in dataframe.columns:

        return pd.Series(
            0,
            index=dataframe.index
        )


    return pd.to_numeric(
        dataframe[column_name],
        errors="coerce"
    ).fillna(0).clip(0, 1)


df["pep_indicator"] = get_binary_column(
    df,
    "pep_flag"
)


df["sanctions_indicator"] = get_binary_column(
    df,
    "sanctions_flag"
)


df["adverse_media_indicator"] = get_binary_column(
    df,
    "adverse_media_flag"
)


# -------------------------------------------------------------------------
# IMPORTANT:
# These are risk-prioritization components for this synthetic project.
# They should not be interpreted as legal conclusions.
# -------------------------------------------------------------------------

df["cdd_risk_component"] = (

    df["pep_indicator"] * 25

    +

    df["sanctions_indicator"] * 60

    +

    df["adverse_media_indicator"] * 25

)


df["cdd_risk_component"] = (
    df["cdd_risk_component"]
    .clip(0, 100)
)


# =============================================================================
# 12. FINAL AML RISK SCORE
# =============================================================================

print("\n" + "=" * 80)
print("12. CALCULATING FINAL AML RISK SCORE")
print("=" * 80)


# =============================================================================
# CONFIGURATION
# =============================================================================
#
# ML model        = 40%
# Rule engine     = 35%
# Customer risk   = 15%
# CDD indicators  = 10%
#
# Total           = 100%
#
# This is a project-specific configurable scoring framework.
# It is NOT a regulatory formula.
# =============================================================================


ML_WEIGHT = 0.40
RULE_WEIGHT = 0.35
CUSTOMER_WEIGHT = 0.15
CDD_WEIGHT = 0.10


WEIGHT_TOTAL = (
    ML_WEIGHT
    +
    RULE_WEIGHT
    +
    CUSTOMER_WEIGHT
    +
    CDD_WEIGHT
)


if abs(WEIGHT_TOTAL - 1.0) > 0.0001:

    print(
        "\nERROR: Risk scoring weights do not equal 100%."
    )

    sys.exit(1)


df["final_aml_risk_score"] = (

    df["ml_risk_score"]
    * ML_WEIGHT

    +

    df["rule_risk_score"]
    * RULE_WEIGHT

    +

    df["customer_risk_score"]
    * CUSTOMER_WEIGHT

    +

    df["cdd_risk_component"]
    * CDD_WEIGHT

)


df["final_aml_risk_score"] = (
    df["final_aml_risk_score"]
    .clip(0, 100)
)


# =============================================================================
# 13. AML RISK LEVEL
# =============================================================================

print("\n" + "=" * 80)
print("13. ASSIGNING AML RISK LEVEL")
print("=" * 80)


def assign_risk_level(
    score
):

    if score >= 75:

        return "CRITICAL"

    elif score >= 50:

        return "HIGH"

    elif score >= 25:

        return "MEDIUM"

    else:

        return "LOW"


df["final_aml_risk_level"] = (
    df["final_aml_risk_score"]
    .apply(
        assign_risk_level
    )
)


print(
    "\nFinal AML risk level distribution:"
)

print(
    df[
        "final_aml_risk_level"
    ].value_counts()
)


# =============================================================================
# 14. ALERT GENERATION
# =============================================================================

print("\n" + "=" * 80)
print("14. GENERATING AML ALERTS")
print("=" * 80)


# Alert threshold
ALERT_THRESHOLD = 25


df["aml_alert"] = (
    df["final_aml_risk_score"]
    >= ALERT_THRESHOLD
)


df["aml_alert"] = (
    df["aml_alert"]
    .astype(int)
)


print(
    f"\nAlert threshold: {ALERT_THRESHOLD}"
)


print(
    "\nAlert count:"
)

print(
    df["aml_alert"].value_counts()
)


# =============================================================================
# 15. ALERT PRIORITY
# =============================================================================

print("\n" + "=" * 80)
print("15. ASSIGNING ANALYST PRIORITY")
print("=" * 80)


def assign_priority(
    row
):

    score = row[
        "final_aml_risk_score"
    ]


    sanctions = row[
        "sanctions_indicator"
    ]


    pep = row[
        "pep_indicator"
    ]


    # -------------------------------------------------------------
    # P1 - Urgent
    # -------------------------------------------------------------

    if (
        score >= 75
        or
        sanctions == 1
    ):

        return "P1"


    # -------------------------------------------------------------
    # P2 - High
    # -------------------------------------------------------------

    elif score >= 50:

        return "P2"


    # -------------------------------------------------------------
    # P3 - Standard
    # -------------------------------------------------------------

    elif score >= 25:

        return "P3"


    # -------------------------------------------------------------
    # P4 - Low / Monitoring
    # -------------------------------------------------------------

    else:

        return "P4"


df["alert_priority"] = (
    df.apply(
        assign_priority,
        axis=1
    )
)


print(
    "\nAlert priority distribution:"
)

print(
    df[
        "alert_priority"
    ].value_counts()
)


# =============================================================================
# 16. MODEL / RULE AGREEMENT
# =============================================================================

print("\n" + "=" * 80)
print("16. CALCULATING MODEL / RULE AGREEMENT")
print("=" * 80)


def calculate_agreement(
    row
):

    ml_high = (
        row[
            "ml_risk_score"
        ] >= 50
    )


    rule_high = (
        row[
            "rule_risk_score"
        ] >= 50
    )


    if ml_high and rule_high:

        return "STRONG_AGREEMENT"


    elif ml_high and not rule_high:

        return "ML_HIGH_RULE_LOW"


    elif rule_high and not ml_high:

        return "RULE_HIGH_ML_LOW"


    else:

        return "LOW_RISK_AGREEMENT"


df["model_rule_agreement"] = (
    df.apply(
        calculate_agreement,
        axis=1
    )
)


# =============================================================================
# 17. DETERMINE PRIMARY RISK DRIVER
# =============================================================================

print("\n" + "=" * 80)
print("17. IDENTIFYING PRIMARY RISK DRIVER")
print("=" * 80)


def determine_primary_driver(
    row
):

    components = {

        "ML_MODEL": row[
            "ml_risk_score"
        ],

        "RULE_ENGINE": row[
            "rule_risk_score"
        ],

        "CUSTOMER_RISK": row[
            "customer_risk_score"
        ],

        "CDD_INDICATORS": row[
            "cdd_risk_component"
        ]

    }


    return max(
        components,
        key=components.get
    )


df["primary_risk_driver"] = (
    df.apply(
        determine_primary_driver,
        axis=1
    )
)


# =============================================================================
# 18. SHAP RISK DRIVERS
# =============================================================================

print("\n" + "=" * 80)
print("18. ADDING SHAP RISK DRIVERS")
print("=" * 80)


if shap_df is not None:

    required_shap_columns = [
        "transaction_id",
        "top_positive_shap_features",
        "top_positive_shap_values",
        "top_negative_shap_features",
        "top_negative_shap_values"
    ]


    available_shap_columns = [
        column
        for column in required_shap_columns
        if column in shap_df.columns
    ]


    if (
        "transaction_id" in available_shap_columns
        and
        len(available_shap_columns) > 1
    ):

        shap_subset = (
            shap_df[
                available_shap_columns
            ]
            .drop_duplicates(
                subset=[
                    "transaction_id"
                ]
            )
        )


        df = df.merge(
            shap_subset,
            on="transaction_id",
            how="left"
        )


        print(
            "\nSHAP drivers merged successfully."
        )


    else:

        print(
            "\nWARNING: Required SHAP columns unavailable."
        )

else:

    print(
        "\nSHAP data unavailable."
    )


# Fill missing SHAP fields
for column in [
    "top_positive_shap_features",
    "top_positive_shap_values",
    "top_negative_shap_features",
    "top_negative_shap_values"
]:

    if column not in df.columns:

        df[column] = ""


    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
    )


# =============================================================================
# 19. AML ALERT REASON
# =============================================================================

print("\n" + "=" * 80)
print("19. GENERATING ALERT REASONS")
print("=" * 80)


def generate_alert_reason(
    row
):

    reasons = []


    if row[
        "rule_risk_score"
    ] >= 50:

        reasons.append(
            "High rule-engine risk"
        )


    if row[
        "ml_risk_score"
    ] >= 75:

        reasons.append(
            "High ML suspiciousness score"
        )


    elif row[
        "ml_risk_score"
    ] >= 50:

        reasons.append(
            "Elevated ML suspiciousness score"
        )


    if row[
        "pep_indicator"
    ] == 1:

        reasons.append(
            "PEP indicator"
        )


    if row[
        "sanctions_indicator"
    ] == 1:

        reasons.append(
            "Sanctions indicator"
        )


    if row[
        "adverse_media_indicator"
    ] == 1:

        reasons.append(
            "Adverse media indicator"
        )


    if row[
        "customer_risk_score"
    ] >= 66.67:

        reasons.append(
            "High customer risk"
        )


    positive_shap = row[
        "top_positive_shap_features"
    ]


    if positive_shap:

        reasons.append(
            "Behavioral model drivers: "
            + positive_shap
        )


    if len(reasons) == 0:

        reasons.append(
            "Low-level monitoring signal"
        )


    return "; ".join(
        reasons
    )


df["alert_reason"] = (
    df.apply(
        generate_alert_reason,
        axis=1
    )
)


# =============================================================================
# 20. INVESTIGATION RECOMMENDATION
# =============================================================================

print("\n" + "=" * 80)
print("20. GENERATING INVESTIGATION RECOMMENDATION")
print("=" * 80)


def investigation_recommendation(
    row
):

    score = row[
        "final_aml_risk_score"
    ]


    sanctions = row[
        "sanctions_indicator"
    ]


    priority = row[
        "alert_priority"
    ]


    agreement = row[
        "model_rule_agreement"
    ]


    if sanctions == 1:

        return (
            "URGENT REVIEW - "
            "Review sanctions-related information and "
            "customer/transaction context before disposition."
        )


    if priority == "P1":

        return (
            "URGENT INVESTIGATION - "
            "Review transaction activity, customer profile, "
            "counterparties and relevant KYC/CDD information."
        )


    if (
        priority == "P2"
        and
        agreement == "STRONG_AGREEMENT"
    ):

        return (
            "HIGH PRIORITY INVESTIGATION - "
            "Rule and ML signals agree; perform detailed "
            "transaction and customer activity review."
        )


    if priority == "P2":

        return (
            "HIGH PRIORITY REVIEW - "
            "Investigate transaction behavior and "
            "customer risk factors."
        )


    if priority == "P3":

        return (
            "STANDARD INVESTIGATION - "
            "Review relevant transactions, risk indicators "
            "and customer profile."
        )


    return (
        "MONITOR - "
        "No immediate high-priority investigation indicated."
    )


df["investigation_recommendation"] = (
    df.apply(
        investigation_recommendation,
        axis=1
    )
)


# =============================================================================
# 21. ALERT TYPE
# =============================================================================

print("\n" + "=" * 80)
print("21. CLASSIFYING ALERT TYPE")
print("=" * 80)


def determine_alert_type(
    row
):

    # -------------------------------------------------------------
    # Use known scenario where available.
    # This is useful for synthetic-project validation.
    # -------------------------------------------------------------

    scenario = str(
        row.get(
            "scenario",
            ""
        )
    ).upper()


    if scenario == "STRUCTURING":

        return "STRUCTURING"


    if scenario == "RAPID_MOVEMENT":

        return "RAPID_MOVEMENT"


    if scenario == "FUNNEL_ACCOUNT":

        return "FUNNEL_ACCOUNT"


    if scenario == "LAYERING":

        return "LAYERING"


    if scenario == "MULE":

        return "MULE_BEHAVIOR"


    if scenario == "FRAUD_LINKED":

        return "FRAUD_LINKED"


    # -------------------------------------------------------------
    # Otherwise derive from risk indicators.
    # -------------------------------------------------------------

    if (
        "structuring" in row[
            "alert_reason"
        ].lower()
    ):

        return "STRUCTURING_RELATED"


    if (
        "rapid" in row[
            "alert_reason"
        ].lower()
    ):

        return "RAPID_MOVEMENT_RELATED"


    return "GENERAL_TRANSACTION_MONITORING"


df["alert_type"] = (
    df.apply(
        determine_alert_type,
        axis=1
    )
)


# =============================================================================
# 22. CUSTOMER-LEVEL ALERT AGGREGATION
# =============================================================================

print("\n" + "=" * 80)
print("22. CREATING CUSTOMER-LEVEL ALERT SUMMARY")
print("=" * 80)


if "customer_id" in df.columns:

    customer_summary = (
        df.groupby(
            "customer_id"
        )
        .agg(

            transaction_count=(
                "transaction_id",
                "count"
            ),

            alert_count=(
                "aml_alert",
                "sum"
            ),

            maximum_aml_risk_score=(
                "final_aml_risk_score",
                "max"
            ),

            average_aml_risk_score=(
                "final_aml_risk_score",
                "mean"
            ),

            maximum_ml_score=(
                "ml_risk_score",
                "max"
            ),

            maximum_rule_score=(
                "rule_risk_score",
                "max"
            ),

            pep_indicator=(
                "pep_indicator",
                "max"
            ),

            sanctions_indicator=(
                "sanctions_indicator",
                "max"
            ),

            adverse_media_indicator=(
                "adverse_media_indicator",
                "max"
            )

        )
        .reset_index()
    )


    customer_summary[
        "customer_alert_rate"
    ] = (
        customer_summary[
            "alert_count"
        ]
        /
        customer_summary[
            "transaction_count"
        ]
    )


    customer_summary[
        "customer_risk_level"
    ] = (
        customer_summary[
            "maximum_aml_risk_score"
        ]
        .apply(
            assign_risk_level
        )
    )


    customer_summary = (
        customer_summary
        .sort_values(
            [
                "maximum_aml_risk_score",
                "alert_count"
            ],
            ascending=False
        )
    )


    customer_summary.to_csv(
        CUSTOMER_ALERT_FILE,
        index=False
    )


    print(
        f"\nCustomer summary saved:"
    )

    print(
        CUSTOMER_ALERT_FILE
    )


# =============================================================================
# 23. CREATE ANALYST ALERT QUEUE
# =============================================================================

print("\n" + "=" * 80)
print("23. CREATING ANALYST INVESTIGATION QUEUE")
print("=" * 80)


# Only alerts
alert_queue = (
    df[
        df["aml_alert"] == 1
    ]
    .copy()
)


# Sort by:
# 1. Priority
# 2. Final risk score
# 3. ML score

priority_order = {
    "P1": 1,
    "P2": 2,
    "P3": 3,
    "P4": 4
}


alert_queue[
    "priority_rank"
] = (
    alert_queue[
        "alert_priority"
    ]
    .map(
        priority_order
    )
    .fillna(99)
)


alert_queue = (
    alert_queue
    .sort_values(
        [
            "priority_rank",
            "final_aml_risk_score",
            "ml_risk_score"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
)


# =============================================================================
# 24. INVESTIGATION QUEUE COLUMNS
# =============================================================================

queue_columns = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount_usd",
    "transaction_type",
    "channel",
    "origin_country",
    "destination_country",
    "is_international",
    "is_cash",
    "scenario",
    "known_suspicious",
    "rule_risk_score",
    "ml_risk_probability",
    "ml_risk_score",
    "customer_risk_score",
    "cdd_risk_component",
    "final_aml_risk_score",
    "final_aml_risk_level",
    "aml_alert",
    "alert_priority",
    "alert_type",
    "model_rule_agreement",
    "primary_risk_driver",
    "alert_reason",
    "top_positive_shap_features",
    "top_positive_shap_values",
    "top_negative_shap_features",
    "top_negative_shap_values",
    "investigation_recommendation"
]


available_queue_columns = [
    column
    for column in queue_columns
    if column in alert_queue.columns
]


investigation_queue = (
    alert_queue[
        available_queue_columns
    ]
    .copy()
)


investigation_queue.to_csv(
    INVESTIGATION_QUEUE_FILE,
    index=False
)


print(
    "\nInvestigation queue saved:"
)

print(
    INVESTIGATION_QUEUE_FILE
)


# =============================================================================
# 25. TOP PRIORITY ALERTS
# =============================================================================

print("\n" + "=" * 80)
print("25. CREATING TOP PRIORITY ALERT REPORT")
print("=" * 80)


top_priority_alerts = (
    investigation_queue
    .head(100)
    .copy()
)


top_priority_alerts.to_csv(
    PRIORITY_FILE,
    index=False
)


print(
    "\nTop priority alerts saved:"
)

print(
    PRIORITY_FILE
)


# =============================================================================
# 26. ALERT SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("26. CREATING ALERT SUMMARY")
print("=" * 80)


summary_rows = []


summary_rows.append({

    "metric":
        "Total transactions",

    "value":
        len(df)

})


summary_rows.append({

    "metric":
        "Total AML alerts",

    "value":
        int(
            df["aml_alert"].sum()
        )

})


summary_rows.append({

    "metric":
        "Alert rate (%)",

    "value":
        round(
            df["aml_alert"].mean() * 100,
            2
        )

})


summary_rows.append({

    "metric":
        "P1 alerts",

    "value":
        int(
            (
                df["alert_priority"] == "P1"
            ).sum()
        )

})


summary_rows.append({

    "metric":
        "P2 alerts",

    "value":
        int(
            (
                df["alert_priority"] == "P2"
            ).sum()
        )

})


summary_rows.append({

    "metric":
        "P3 alerts",

    "value":
        int(
            (
                df["alert_priority"] == "P3"
            ).sum()
        )

})


summary_rows.append({

    "metric":
        "P4 alerts",

    "value":
        int(
            (
                df["alert_priority"] == "P4"
            ).sum()
        )

})


summary_rows.append({

    "metric":
        "Critical risk",

    "value":
        int(
            (
                df[
                    "final_aml_risk_level"
                ]
                ==
                "CRITICAL"
            ).sum()
        )

})


summary_rows.append({

    "metric":
        "High risk",

    "value":
        int(
            (
                df[
                    "final_aml_risk_level"
                ]
                ==
                "HIGH"
            ).sum()
        )

})


summary_rows.append({

    "metric":
        "Medium risk",

    "value":
        int(
            (
                df[
                    "final_aml_risk_level"
                ]
                ==
                "MEDIUM"
            ).sum()
        )

})


summary_rows.append({

    "metric":
        "Low risk",

    "value":
        int(
            (
                df[
                    "final_aml_risk_level"
                ]
                ==
                "LOW"
            ).sum()
        )

})


summary_rows.append({

    "metric":
        "Strong ML / Rule agreement",

    "value":
        int(
            (
                df[
                    "model_rule_agreement"
                ]
                ==
                "STRONG_AGREEMENT"
            ).sum()
        )

})


summary_rows.append({

    "metric":
        "ML high / Rule low",

    "value":
        int(
            (
                df[
                    "model_rule_agreement"
                ]
                ==
                "ML_HIGH_RULE_LOW"
            ).sum()
        )

})


summary_rows.append({

    "metric":
        "Rule high / ML low",

    "value":
        int(
            (
                df[
                    "model_rule_agreement"
                ]
                ==
                "RULE_HIGH_ML_LOW"
            ).sum()
        )

})


alert_summary = pd.DataFrame(
    summary_rows
)


alert_summary.to_csv(
    ALERT_SUMMARY_FILE,
    index=False
)


print(
    "\nAlert summary saved:"
)

print(
    ALERT_SUMMARY_FILE
)


# =============================================================================
# 27. RISK DRIVER SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("27. CREATING RISK DRIVER SUMMARY")
print("=" * 80)


driver_summary = (
    df.groupby(
        "primary_risk_driver"
    )
    .agg(

        transaction_count=(
            "transaction_id",
            "count"
        ),

        alert_count=(
            "aml_alert",
            "sum"
        ),

        average_final_risk_score=(
            "final_aml_risk_score",
            "mean"
        ),

        maximum_final_risk_score=(
            "final_aml_risk_score",
            "max"
        )

    )
    .reset_index()
)


driver_summary[
    "alert_rate"
] = (
    driver_summary[
        "alert_count"
    ]
    /
    driver_summary[
        "transaction_count"
    ]
)


driver_summary.to_csv(
    DRIVER_SUMMARY_FILE,
    index=False
)


print(
    "\nRisk driver summary saved:"
)

print(
    DRIVER_SUMMARY_FILE
)


# =============================================================================
# 28. SAVE CONFIGURATION
# =============================================================================

print("\n" + "=" * 80)
print("28. SAVING RISK SCORING CONFIGURATION")
print("=" * 80)


with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AML RISK SCORING CONFIGURATION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


    file.write(
        "IMPORTANT:\n"
    )

    file.write(
        "This is a project-specific risk prioritization framework.\n"
    )

    file.write(
        "It is not a regulatory formula or legal determination.\n"
    )

    file.write(
        "A high score does not prove money laundering or fraud.\n"
    )

    file.write(
        "SAR decisions require appropriate human investigation "
        "and institutional procedures.\n\n"
    )


    file.write(
        "SCORING WEIGHTS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )


    file.write(
        f"ML Risk Score: {ML_WEIGHT:.0%}\n"
    )

    file.write(
        f"Rule Engine Score: {RULE_WEIGHT:.0%}\n"
    )

    file.write(
        f"Customer Risk Score: {CUSTOMER_WEIGHT:.0%}\n"
    )

    file.write(
        f"CDD Indicator Component: {CDD_WEIGHT:.0%}\n\n"
    )


    file.write(
        "RISK LEVELS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        "CRITICAL: 75-100\n"
    )

    file.write(
        "HIGH: 50-74.99\n"
    )

    file.write(
        "MEDIUM: 25-49.99\n"
    )

    file.write(
        "LOW: 0-24.99\n\n"
    )


    file.write(
        "ALERT THRESHOLD\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"AML alert threshold: {ALERT_THRESHOLD}\n\n"
    )


    file.write(
        "PRIORITY LEVELS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        "P1: Urgent\n"
    )

    file.write(
        "P2: High\n"
    )

    file.write(
        "P3: Standard\n"
    )

    file.write(
        "P4: Low / Monitoring\n"
    )


print(
    "\nConfiguration saved:"
)

print(
    CONFIG_FILE
)


# =============================================================================
# 29. FINAL DISPLAY
# =============================================================================

print("\n" + "=" * 80)
print("29. FINAL AML ALERT SUMMARY")
print("=" * 80)


print(
    f"\nTotal transactions: "
    f"{len(df):,}"
)


print(
    f"Total AML alerts: "
    f"{int(df['aml_alert'].sum()):,}"
)


print(
    f"Alert rate: "
    f"{df['aml_alert'].mean() * 100:.2f}%"
)


print(
    "\nRisk levels:"
)

print(
    df[
        "final_aml_risk_level"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nAlert priorities:"
)

print(
    df[
        "alert_priority"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nModel / Rule agreement:"
)

print(
    df[
        "model_rule_agreement"
    ]
    .value_counts()
    .to_string()
)


# =============================================================================
# 30. TOP 10 ALERTS
# =============================================================================

print("\n" + "=" * 80)
print("30. TOP 10 AML ALERTS")
print("=" * 80)


display_columns = [
    "transaction_id",
    "customer_id",
    "amount_usd",
    "final_aml_risk_score",
    "final_aml_risk_level",
    "alert_priority",
    "alert_type",
    "primary_risk_driver",
    "model_rule_agreement"
]


display_columns = [
    column
    for column in display_columns
    if column in investigation_queue.columns
]


if len(investigation_queue) > 0:

    print(
        investigation_queue[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

else:

    print(
        "\nNo AML alerts generated."
    )


# =============================================================================
# 31. OUTPUT FILES
# =============================================================================

print("\n" + "=" * 80)
print("31. PHASE 7 OUTPUT FILES")
print("=" * 80)


print(
    "\n1. Transaction-level AML alerts:"
)

print(
    OUTPUT_FILE
)


# Save final scored dataset
df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n2. Customer alert summary:"
)

print(
    CUSTOMER_ALERT_FILE
)


print(
    "\n3. Alert summary:"
)

print(
    ALERT_SUMMARY_FILE
)


print(
    "\n4. Top priority alerts:"
)

print(
    PRIORITY_FILE
)


print(
    "\n5. Analyst investigation queue:"
)

print(
    INVESTIGATION_QUEUE_FILE
)


print(
    "\n6. Risk driver summary:"
)

print(
    DRIVER_SUMMARY_FILE
)


print(
    "\n7. Risk scoring configuration:"
)

print(
    CONFIG_FILE
)


# =============================================================================
# 32. COMPLETION
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 7 COMPLETED SUCCESSFULLY")
print("=" * 80)


print(
    "\nYour AML Transaction Monitoring project now has:"
)

print(
    "  [OK] Rule-based risk scoring"
)

print(
    "  [OK] ML-based risk scoring"
)

print(
    "  [OK] Customer risk scoring"
)

print(
    "  [OK] PEP / sanctions / adverse-media indicators"
)

print(
    "  [OK] SHAP risk drivers"
)

print(
    "  [OK] Combined AML risk score"
)

print(
    "  [OK] AML risk levels"
)

print(
    "  [OK] P1-P4 alert prioritization"
)

print(
    "  [OK] Model / rule agreement analysis"
)

print(
    "  [OK] Analyst investigation queue"
)

print(
    "  [OK] Customer-level alert aggregation"
)


print(
    "\nNext phase:"
)

print(
    "PHASE 8 - AML ALERT INVESTIGATION & CASE MANAGEMENT"
)

print(
    "=" * 80
)