# =============================================================================
# PHASE 8 - AML ALERT INVESTIGATION & CASE MANAGEMENT
# US FINANCIAL CRIME / AML TRANSACTION MONITORING PROJECT
# =============================================================================
#
# Purpose:
#   Simulate an AML/FCC analyst investigation workflow.
#
# The system:
#   1. Loads prioritized AML alerts
#   2. Builds customer profiles
#   3. Reviews transaction history
#   4. Identifies behavioral red flags
#   5. Reviews ML / Rule / SHAP signals
#   6. Creates investigation cases
#   7. Generates analyst investigation recommendations
#   8. Supports case disposition
#
# IMPORTANT:
#   This system does NOT determine criminal activity.
#   This system does NOT automatically file a SAR.
#   Final disposition and SAR decisions require human investigation,
#   appropriate evidence, policies and institutional procedures.
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


TRANSACTION_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "aml_alerts_scored.csv"
)


CUSTOMER_ALERT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "customer_alert_risk_summary.csv"
)


INVESTIGATION_QUEUE_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "alerts",
    "analyst_investigation_queue.csv"
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)


REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "investigation"
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
# 2. OUTPUT FILES
# =============================================================================

CASE_FILE = os.path.join(
    OUTPUT_DIR,
    "aml_investigation_cases.csv"
)


CUSTOMER_PROFILE_FILE = os.path.join(
    REPORT_DIR,
    "customer_investigation_profiles.csv"
)


TRANSACTION_REVIEW_FILE = os.path.join(
    REPORT_DIR,
    "transaction_review_summary.csv"
)


RED_FLAG_FILE = os.path.join(
    REPORT_DIR,
    "investigation_red_flags.csv"
)


CASE_SUMMARY_FILE = os.path.join(
    REPORT_DIR,
    "case_summary.csv"
)


ANALYST_GUIDE_FILE = os.path.join(
    REPORT_DIR,
    "analyst_investigation_guide.txt"
)


# =============================================================================
# 3. HEADER
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 8 - AML ALERT INVESTIGATION & CASE MANAGEMENT")
print("=" * 80)


# =============================================================================
# 4. LOAD ALERT DATA
# =============================================================================

print("\n" + "=" * 80)
print("4. LOADING AML ALERT DATA")
print("=" * 80)


if not os.path.exists(TRANSACTION_FILE):

    print(
        "\nERROR: AML scored transaction file not found."
    )

    print(
        TRANSACTION_FILE
    )

    print(
        "\nPlease run Phase 7 first."
    )

    sys.exit(1)


df = pd.read_csv(
    TRANSACTION_FILE
)


print(
    f"\nTransaction dataset shape: {df.shape}"
)


# =============================================================================
# 5. CLEAN DATA
# =============================================================================

print("\n" + "=" * 80)
print("5. CLEANING DATA")
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
# 6. VERIFY REQUIRED COLUMNS
# =============================================================================

print("\n" + "=" * 80)
print("6. VERIFYING REQUIRED FIELDS")
print("=" * 80)


required_columns = [
    "transaction_id",
    "customer_id",
    "amount_usd",
    "final_aml_risk_score",
    "final_aml_risk_level",
    "alert_priority",
    "alert_type",
    "alert_reason",
    "investigation_recommendation"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if len(missing_columns) > 0:

    print(
        "\nERROR: Required columns missing:"
    )

    for column in missing_columns:

        print(
            f"  - {column}"
        )

    print(
        "\nPlease run Phase 7 again."
    )

    sys.exit(1)


print(
    "\nAll required investigation fields are available."
)


# =============================================================================
# 7. SELECT ALERTS FOR INVESTIGATION
# =============================================================================

print("\n" + "=" * 80)
print("7. SELECTING ALERTS FOR INVESTIGATION")
print("=" * 80)


# Only actual alerts
alerts = (
    df[
        df["aml_alert"] == 1
    ]
    .copy()
)


print(
    f"\nTotal transactions: {len(df):,}"
)


print(
    f"Total AML alerts: {len(alerts):,}"
)


# =============================================================================
# 8. CUSTOMER PROFILE FUNCTION
# =============================================================================

print("\n" + "=" * 80)
print("8. BUILDING CUSTOMER INVESTIGATION PROFILES")
print("=" * 80)


def safe_first(
    series,
    default=""
):

    if series is None:

        return default


    series = series.dropna()


    if len(series) == 0:

        return default


    return series.iloc[0]


def numeric_sum(
    series
):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0).sum()


def numeric_mean(
    series
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()


    if len(values) == 0:

        return 0.0


    return values.mean()


def numeric_max(
    series
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()


    if len(values) == 0:

        return 0.0


    return values.max()


def numeric_min(
    series
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()


    if len(values) == 0:

        return 0.0


    return values.min()


# =============================================================================
# 9. BUILD CUSTOMER PROFILES
# =============================================================================

customer_profiles = []


for customer_id, customer_transactions in df.groupby(
    "customer_id"
):

    customer_transactions = (
        customer_transactions
        .sort_values(
            "timestamp"
        )
    )


    total_transactions = len(
        customer_transactions
    )


    total_volume = numeric_sum(
        customer_transactions[
            "amount_usd"
        ]
    )


    average_transaction = numeric_mean(
        customer_transactions[
            "amount_usd"
        ]
    )


    maximum_transaction = numeric_max(
        customer_transactions[
            "amount_usd"
        ]
    )


    minimum_transaction = numeric_min(
        customer_transactions[
            "amount_usd"
        ]
    )


    alert_count = int(
        numeric_sum(
            customer_transactions[
                "aml_alert"
            ]
        )
    )


    suspicious_count = int(
        numeric_sum(
            customer_transactions.get(
                "known_suspicious",
                pd.Series(
                    0,
                    index=customer_transactions.index
                )
            )
        )
    )


    maximum_risk_score = numeric_max(
        customer_transactions[
            "final_aml_risk_score"
        ]
    )


    average_risk_score = numeric_mean(
        customer_transactions[
            "final_aml_risk_score"
        ]
    )


    # -------------------------------------------------------------
    # Customer profile information
    # -------------------------------------------------------------

    profile_fields = {}


    for field in [
        "customer_type",
        "state",
        "occupation",
        "annual_income_usd",
        "customer_since",
        "risk_rating",
        "pep_flag",
        "sanctions_flag",
        "adverse_media_flag",
        "expected_monthly_volume_usd"
    ]:

        if field in customer_transactions.columns:

            profile_fields[field] = safe_first(
                customer_transactions[field]
            )

        else:

            profile_fields[field] = ""


    # -------------------------------------------------------------
    # Geographic activity
    # -------------------------------------------------------------

    if "origin_country" in customer_transactions.columns:

        origin_country_count = (
            customer_transactions[
                "origin_country"
            ]
            .nunique()
        )

    else:

        origin_country_count = 0


    if "destination_country" in customer_transactions.columns:

        destination_country_count = (
            customer_transactions[
                "destination_country"
            ]
            .nunique()
        )

    else:

        destination_country_count = 0


    # -------------------------------------------------------------
    # Counterparties
    # -------------------------------------------------------------

    if "receiver_account" in customer_transactions.columns:

        unique_counterparties = (
            customer_transactions[
                "receiver_account"
            ]
            .nunique()
        )

    else:

        unique_counterparties = 0


    # -------------------------------------------------------------
    # Transaction behavior
    # -------------------------------------------------------------

    cash_count = int(
        numeric_sum(
            customer_transactions.get(
                "is_cash",
                pd.Series(
                    0,
                    index=customer_transactions.index
                )
            )
        )
    )


    international_count = int(
        numeric_sum(
            customer_transactions.get(
                "is_international",
                pd.Series(
                    0,
                    index=customer_transactions.index
                )
            )
        )
    )


    profile = {

        "customer_id":
            customer_id,

        "customer_type":
            profile_fields.get(
                "customer_type",
                ""
            ),

        "state":
            profile_fields.get(
                "state",
                ""
            ),

        "occupation":
            profile_fields.get(
                "occupation",
                ""
            ),

        "annual_income_usd":
            profile_fields.get(
                "annual_income_usd",
                0
            ),

        "customer_since":
            profile_fields.get(
                "customer_since",
                ""
            ),

        "risk_rating":
            profile_fields.get(
                "risk_rating",
                ""
            ),

        "pep_flag":
            profile_fields.get(
                "pep_flag",
                0
            ),

        "sanctions_flag":
            profile_fields.get(
                "sanctions_flag",
                0
            ),

        "adverse_media_flag":
            profile_fields.get(
                "adverse_media_flag",
                0
            ),

        "expected_monthly_volume_usd":
            profile_fields.get(
                "expected_monthly_volume_usd",
                0
            ),

        "total_transactions":
            total_transactions,

        "total_transaction_volume_usd":
            total_volume,

        "average_transaction_usd":
            average_transaction,

        "maximum_transaction_usd":
            maximum_transaction,

        "minimum_transaction_usd":
            minimum_transaction,

        "cash_transaction_count":
            cash_count,

        "international_transaction_count":
            international_count,

        "unique_counterparties":
            unique_counterparties,

        "origin_country_count":
            origin_country_count,

        "destination_country_count":
            destination_country_count,

        "alert_count":
            alert_count,

        "known_suspicious_count":
            suspicious_count,

        "maximum_risk_score":
            maximum_risk_score,

        "average_risk_score":
            average_risk_score

    }


    customer_profiles.append(
        profile
    )


customer_profile_df = pd.DataFrame(
    customer_profiles
)


customer_profile_df.to_csv(
    CUSTOMER_PROFILE_FILE,
    index=False
)


print(
    f"\nCustomer profiles created: "
    f"{len(customer_profile_df):,}"
)


print(
    "\nSaved:"
)

print(
    CUSTOMER_PROFILE_FILE
)


# =============================================================================
# 10. TRANSACTION REVIEW SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("10. BUILDING TRANSACTION REVIEW SUMMARY")
print("=" * 80)


transaction_review = (
    df.groupby(
        "customer_id"
    )
    .agg(

        transaction_count=(
            "transaction_id",
            "count"
        ),

        total_volume_usd=(
            "amount_usd",
            "sum"
        ),

        average_transaction_usd=(
            "amount_usd",
            "mean"
        ),

        maximum_transaction_usd=(
            "amount_usd",
            "max"
        ),

        alert_count=(
            "aml_alert",
            "sum"
        ),

        maximum_risk_score=(
            "final_aml_risk_score",
            "max"
        ),

        average_risk_score=(
            "final_aml_risk_score",
            "mean"
        )

    )
    .reset_index()
)


transaction_review[
    "alert_rate"
] = (
    transaction_review[
        "alert_count"
    ]
    /
    transaction_review[
        "transaction_count"
    ]
)


transaction_review.to_csv(
    TRANSACTION_REVIEW_FILE,
    index=False
)


print(
    "\nSaved:"
)

print(
    TRANSACTION_REVIEW_FILE
)


# =============================================================================
# 11. RED FLAG DETECTION
# =============================================================================

print("\n" + "=" * 80)
print("11. IDENTIFYING INVESTIGATION RED FLAGS")
print("=" * 80)


def add_red_flag(
    flags,
    customer_transactions,
    condition,
    code,
    description
):

    if condition:

        flags.append({

            "red_flag_code":
                code,

            "red_flag_description":
                description

        })


red_flag_rows = []


for customer_id, customer_transactions in df.groupby(
    "customer_id"
):

    customer_transactions = (
        customer_transactions
        .sort_values(
            "timestamp"
        )
    )


    flags = []


    # -------------------------------------------------------------------------
    # 1. Structuring
    # -------------------------------------------------------------------------

    if "amount_usd" in customer_transactions.columns:

        amounts = pd.to_numeric(
            customer_transactions[
                "amount_usd"
            ],
            errors="coerce"
        ).fillna(0)


        structuring_count = int(
            (
                (amounts >= 8000)
                &
                (amounts < 10000)
            ).sum()
        )

    else:

        structuring_count = 0


    add_red_flag(
        flags,
        customer_transactions,
        structuring_count >= 3,
        "RF_STRUCTURING",
        "Multiple transactions observed in the synthetic "
        "structuring threshold range."
    )


    # -------------------------------------------------------------------------
    # 2. High transaction velocity
    # -------------------------------------------------------------------------

    if "customer_24h_transaction_count" in customer_transactions.columns:

        max_24h_count = numeric_max(
            customer_transactions[
                "customer_24h_transaction_count"
            ]
        )

    else:

        max_24h_count = 0


    add_red_flag(
        flags,
        customer_transactions,
        max_24h_count >= 10,
        "RF_HIGH_VELOCITY",
        "High transaction activity observed within a "
        "24-hour period."
    )


    # -------------------------------------------------------------------------
    # 3. Rapid movement
    # -------------------------------------------------------------------------

    if "rapid_movement_indicator" in customer_transactions.columns:

        rapid_count = int(
            numeric_sum(
                customer_transactions[
                    "rapid_movement_indicator"
                ]
            )
        )

    else:

        rapid_count = 0


    add_red_flag(
        flags,
        customer_transactions,
        rapid_count > 0,
        "RF_RAPID_MOVEMENT",
        "Transactions show rapid movement behavior."
    )


    # -------------------------------------------------------------------------
    # 4. Multiple counterparties
    # -------------------------------------------------------------------------

    if "receiver_account" in customer_transactions.columns:

        counterparty_count = (
            customer_transactions[
                "receiver_account"
            ]
            .nunique()
        )

    else:

        counterparty_count = 0


    add_red_flag(
        flags,
        customer_transactions,
        counterparty_count >= 8,
        "RF_MULTIPLE_COUNTERPARTIES",
        "Customer interacted with multiple counterparties."
    )


    # -------------------------------------------------------------------------
    # 5. International activity
    # -------------------------------------------------------------------------

    international_count = int(
        numeric_sum(
            customer_transactions.get(
                "is_international",
                pd.Series(
                    0,
                    index=customer_transactions.index
                )
            )
        )
    )


    add_red_flag(
        flags,
        customer_transactions,
        international_count > 0,
        "RF_INTERNATIONAL_ACTIVITY",
        "International transaction activity observed."
    )


    # -------------------------------------------------------------------------
    # 6. Cash activity
    # -------------------------------------------------------------------------

    cash_count = int(
        numeric_sum(
            customer_transactions.get(
                "is_cash",
                pd.Series(
                    0,
                    index=customer_transactions.index
                )
            )
        )
    )


    add_red_flag(
        flags,
        customer_transactions,
        cash_count > 0,
        "RF_CASH_ACTIVITY",
        "Cash transaction activity observed."
    )


    # -------------------------------------------------------------------------
    # 7. Profile mismatch
    # -------------------------------------------------------------------------

    if "amount_vs_expected" in customer_transactions.columns:

        max_expected_ratio = numeric_max(
            customer_transactions[
                "amount_vs_expected"
            ]
        )

    else:

        max_expected_ratio = 0


    add_red_flag(
        flags,
        customer_transactions,
        max_expected_ratio > 3,
        "RF_PROFILE_MISMATCH",
        "Transaction activity exceeds expected customer "
        "volume assumptions."
    )


    # -------------------------------------------------------------------------
    # 8. PEP
    # -------------------------------------------------------------------------

    pep_flag = int(
        numeric_max(
            customer_transactions.get(
                "pep_flag",
                pd.Series(
                    0,
                    index=customer_transactions.index
                )
            )
        )
    )


    add_red_flag(
        flags,
        customer_transactions,
        pep_flag == 1,
        "RF_PEP",
        "Customer is marked with a PEP indicator in the "
        "synthetic dataset."
    )


    # -------------------------------------------------------------------------
    # 9. Sanctions
    # -------------------------------------------------------------------------

    sanctions_flag = int(
        numeric_max(
            customer_transactions.get(
                "sanctions_flag",
                pd.Series(
                    0,
                    index=customer_transactions.index
                )
            )
        )
    )


    add_red_flag(
        flags,
        customer_transactions,
        sanctions_flag == 1,
        "RF_SANCTIONS",
        "Customer has a sanctions indicator in the "
        "synthetic dataset."
    )


    # -------------------------------------------------------------------------
    # 10. Adverse media
    # -------------------------------------------------------------------------

    adverse_media_flag = int(
        numeric_max(
            customer_transactions.get(
                "adverse_media_flag",
                pd.Series(
                    0,
                    index=customer_transactions.index
                )
            )
        )
    )


    add_red_flag(
        flags,
        customer_transactions,
        adverse_media_flag == 1,
        "RF_ADVERSE_MEDIA",
        "Customer has an adverse-media indicator in the "
        "synthetic dataset."
    )


    # -------------------------------------------------------------------------
    # 11. Funnel behavior
    # -------------------------------------------------------------------------

    origin_country_count = (
        customer_transactions[
            "origin_country"
        ]
        .nunique()
        if "origin_country" in customer_transactions.columns
        else 0
    )


    add_red_flag(
        flags,
        customer_transactions,
        (
            origin_country_count >= 3
            and
            len(customer_transactions) >= 10
        ),
        "RF_FUNNEL_BEHAVIOR",
        "Activity involves multiple origin countries and "
        "elevated transaction frequency."
    )


    # -------------------------------------------------------------------------
    # 12. Mule behavior
    # -------------------------------------------------------------------------

    mule_condition = (
        len(customer_transactions) >= 15
        and
        (
            pd.to_numeric(
                customer_transactions[
                    "amount_usd"
                ],
                errors="coerce"
            )
            .fillna(0)
            .max()
            >= 5000
        )
    )


    add_red_flag(
        flags,
        customer_transactions,
        mule_condition,
        "RF_MULE_BEHAVIOR",
        "Transaction activity is consistent with the "
        "synthetic mule-behavior scenario."
    )


    # -------------------------------------------------------------------------
    # Save customer red flags
    # -------------------------------------------------------------------------

    for flag in flags:

        red_flag_rows.append({

            "customer_id":
                customer_id,

            "red_flag_code":
                flag[
                    "red_flag_code"
                ],

            "red_flag_description":
                flag[
                    "red_flag_description"
                ]

        })


red_flag_df = pd.DataFrame(
    red_flag_rows
)


if len(red_flag_df) > 0:

    red_flag_summary = (
        red_flag_df
        .groupby(
            [
                "customer_id",
                "red_flag_code"
            ]
        )
        .agg(

            red_flag_description=(
                "red_flag_description",
                "first"
            )

        )
        .reset_index()
    )

else:

    red_flag_summary = pd.DataFrame(
        columns=[
            "customer_id",
            "red_flag_code",
            "red_flag_description"
        ]
    )


red_flag_summary.to_csv(
    RED_FLAG_FILE,
    index=False
)


print(
    f"\nCustomers with identified red flags: "
    f"{red_flag_summary['customer_id'].nunique():,}"
)


print(
    "\nSaved:"
)

print(
    RED_FLAG_FILE
)


# =============================================================================
# 12. CASE GENERATION
# =============================================================================

print("\n" + "=" * 80)
print("12. GENERATING AML INVESTIGATION CASES")
print("=" * 80)


# -------------------------------------------------------------------------
# We investigate each unique customer with an AML alert.
# This avoids creating a completely separate case for every transaction.
# -------------------------------------------------------------------------

alert_customers = (
    alerts[
        "customer_id"
    ]
    .dropna()
    .astype(str)
    .unique()
)


case_rows = []


for customer_id in alert_customers:

    customer_alerts = (
        alerts[
            alerts[
                "customer_id"
            ]
            .astype(str)
            ==
            str(customer_id)
        ]
        .copy()
    )


    customer_all_transactions = (
        df[
            df[
                "customer_id"
            ]
            .astype(str)
            ==
            str(customer_id)
        ]
        .copy()
    )


    # -------------------------------------------------------------------------
    # Highest risk alert
    # -------------------------------------------------------------------------

    highest_risk_row = (
        customer_alerts
        .sort_values(
            [
                "final_aml_risk_score",
                "ml_risk_score",
                "rule_risk_score"
            ],
            ascending=False
        )
        .iloc[0]
    )


    highest_risk_score = float(
        highest_risk_row[
            "final_aml_risk_score"
        ]
    )


    # -------------------------------------------------------------------------
    # Alert count
    # -------------------------------------------------------------------------

    alert_count = len(
        customer_alerts
    )


    # -------------------------------------------------------------------------
    # Total transaction volume
    # -------------------------------------------------------------------------

    total_volume = numeric_sum(
        customer_all_transactions[
            "amount_usd"
        ]
    )


    # -------------------------------------------------------------------------
    # Alert transaction volume
    # -------------------------------------------------------------------------

    alert_volume = numeric_sum(
        customer_alerts[
            "amount_usd"
        ]
    )


    # -------------------------------------------------------------------------
    # Customer profile
    # -------------------------------------------------------------------------

    profile_match = (
        customer_profile_df[
            customer_profile_df[
                "customer_id"
            ]
            .astype(str)
            ==
            str(customer_id)
        ]
    )


    if len(profile_match) > 0:

        profile = profile_match.iloc[0]

    else:

        profile = pd.Series()


    # -------------------------------------------------------------------------
    # Red flags
    # -------------------------------------------------------------------------

    customer_red_flags = (
        red_flag_summary[
            red_flag_summary[
                "customer_id"
            ]
            .astype(str)
            ==
            str(customer_id)
        ]
    )


    red_flag_codes = (
        customer_red_flags[
            "red_flag_code"
        ]
        .astype(str)
        .tolist()
    )


    red_flag_descriptions = (
        customer_red_flags[
            "red_flag_description"
        ]
        .astype(str)
        .tolist()
    )


    # -------------------------------------------------------------------------
    # SHAP drivers
    # -------------------------------------------------------------------------

    shap_positive = (
        customer_alerts[
            "top_positive_shap_features"
        ]
        .dropna()
        .astype(str)
        .tolist()
        if
        "top_positive_shap_features"
        in customer_alerts.columns
        else []
    )


    shap_positive = [
        value
        for value in shap_positive
        if value.strip()
    ]


    # Keep unique drivers
    shap_positive_unique = []


    for value in shap_positive:

        for driver in value.split("|"):

            driver = driver.strip()

            if (
                driver
                and
                driver not in shap_positive_unique
            ):

                shap_positive_unique.append(
                    driver
                )


    # -------------------------------------------------------------------------
    # Determine case priority
    # -------------------------------------------------------------------------

    priorities = (
        customer_alerts[
            "alert_priority"
        ]
        .astype(str)
        .tolist()
    )


    if "P1" in priorities:

        case_priority = "P1"

    elif "P2" in priorities:

        case_priority = "P2"

    elif "P3" in priorities:

        case_priority = "P3"

    else:

        case_priority = "P4"


    # -------------------------------------------------------------------------
    # Determine case risk level
    # -------------------------------------------------------------------------

    if highest_risk_score >= 75:

        case_risk_level = "CRITICAL"

    elif highest_risk_score >= 50:

        case_risk_level = "HIGH"

    elif highest_risk_score >= 25:

        case_risk_level = "MEDIUM"

    else:

        case_risk_level = "LOW"


    # -------------------------------------------------------------------------
    # Determine preliminary disposition recommendation
    # -------------------------------------------------------------------------

    if case_priority == "P1":

        preliminary_disposition = (
            "ESCALATE_FOR_DETAILED_REVIEW"
        )

    elif case_priority == "P2":

        preliminary_disposition = (
            "INVESTIGATE"
        )

    elif case_priority == "P3":

        preliminary_disposition = (
            "STANDARD_REVIEW"
        )

    else:

        preliminary_disposition = (
            "MONITOR"
        )


    # -------------------------------------------------------------------------
    # Generate case ID
    # -------------------------------------------------------------------------

    case_id = (
        "AML-"
        + str(customer_id)
    )


    # -------------------------------------------------------------------------
    # Case row
    # -------------------------------------------------------------------------

    case_rows.append({

        "case_id":
            case_id,

        "customer_id":
            customer_id,

        "case_priority":
            case_priority,

        "case_risk_level":
            case_risk_level,

        "highest_risk_score":
            round(
                highest_risk_score,
                2
            ),

        "alert_count":
            alert_count,

        "total_transaction_volume_usd":
            round(
                total_volume,
                2
            ),

        "alert_transaction_volume_usd":
            round(
                alert_volume,
                2
            ),

        "customer_risk_rating":
            profile.get(
                "risk_rating",
                ""
            ),

        "customer_type":
            profile.get(
                "customer_type",
                ""
            ),

        "occupation":
            profile.get(
                "occupation",
                ""
            ),

        "annual_income_usd":
            profile.get(
                "annual_income_usd",
                0
            ),

        "expected_monthly_volume_usd":
            profile.get(
                "expected_monthly_volume_usd",
                0
            ),

        "pep_flag":
            profile.get(
                "pep_flag",
                0
            ),

        "sanctions_flag":
            profile.get(
                "sanctions_flag",
                0
            ),

        "adverse_media_flag":
            profile.get(
                "adverse_media_flag",
                0
            ),

        "red_flag_count":
            len(
                red_flag_codes
            ),

        "red_flag_codes":
            " | ".join(
                red_flag_codes
            ),

        "red_flag_descriptions":
            " | ".join(
                red_flag_descriptions
            ),

        "top_shap_drivers":
            " | ".join(
                shap_positive_unique[:10]
            ),

        "primary_alert_type":
            highest_risk_row[
                "alert_type"
            ],

        "primary_alert_reason":
            highest_risk_row[
                "alert_reason"
            ],

        "model_rule_agreement":
            highest_risk_row[
                "model_rule_agreement"
            ],

        "preliminary_disposition":
            preliminary_disposition,

        # -------------------------------------------------------------
        # Analyst-controlled fields
        # -------------------------------------------------------------

        "analyst_disposition":
            "PENDING_REVIEW",

        "investigation_status":
            "OPEN",

        "analyst_notes":
            "",

        "escalation_required":
            (
                "YES"
                if case_priority in [
                    "P1",
                    "P2"
                ]
                else "NO"
            ),

        "sar_consideration":
            "PENDING_HUMAN_REVIEW"

    })


case_df = pd.DataFrame(
    case_rows
)


case_df = (
    case_df
    .sort_values(
        [
            "case_priority",
            "highest_risk_score"
        ],
        ascending=[
            True,
            False
        ]
    )
)


case_df.to_csv(
    CASE_FILE,
    index=False
)


print(
    f"\nInvestigation cases created: "
    f"{len(case_df):,}"
)


print(
    "\nSaved:"
)

print(
    CASE_FILE
)


# =============================================================================
# 13. CASE SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("13. CASE SUMMARY")
print("=" * 80)


if len(case_df) > 0:

    case_summary = (
        case_df
        .groupby(
            [
                "case_priority",
                "case_risk_level"
            ]
        )
        .agg(

            case_count=(
                "case_id",
                "count"
            ),

            average_risk_score=(
                "highest_risk_score",
                "mean"
            ),

            maximum_risk_score=(
                "highest_risk_score",
                "max"
            ),

            total_alert_count=(
                "alert_count",
                "sum"
            )

        )
        .reset_index()
    )

else:

    case_summary = pd.DataFrame(
        columns=[
            "case_priority",
            "case_risk_level",
            "case_count",
            "average_risk_score",
            "maximum_risk_score",
            "total_alert_count"
        ]
    )


case_summary.to_csv(
    CASE_SUMMARY_FILE,
    index=False
)


print(
    "\nSaved:"
)

print(
    CASE_SUMMARY_FILE
)


# =============================================================================
# 14. ANALYST INVESTIGATION GUIDE
# =============================================================================

print("\n" + "=" * 80)
print("14. CREATING ANALYST INVESTIGATION GUIDE")
print("=" * 80)


with open(
    ANALYST_GUIDE_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AML TRANSACTION MONITORING - ANALYST INVESTIGATION GUIDE\n"
    )

    file.write(
        "=" * 75 + "\n\n"
    )


    file.write(
        "PURPOSE\n"
    )

    file.write(
        "This guide provides a structured investigation workflow "
        "for AML transaction-monitoring alerts.\n\n"
    )


    file.write(
        "STEP 1 - REVIEW ALERT\n"
    )

    file.write(
        "-" * 75 + "\n"
    )

    file.write(
        "Review alert score, priority, alert type, rule signals, "
        "ML score and SHAP drivers.\n\n"
    )


    file.write(
        "STEP 2 - REVIEW CUSTOMER PROFILE\n"
    )

    file.write(
        "-" * 75 + "\n"
    )

    file.write(
        "Review customer type, occupation, expected activity, "
        "risk rating, PEP indicator, sanctions indicator and "
        "adverse-media indicator.\n\n"
    )


    file.write(
        "STEP 3 - REVIEW TRANSACTION ACTIVITY\n"
    )

    file.write(
        "-" * 75 + "\n"
    )

    file.write(
        "Review transaction amounts, frequency, counterparties, "
        "cash activity, international activity, geographic activity "
        "and transaction timing.\n\n"
    )


    file.write(
        "STEP 4 - REVIEW RED FLAGS\n"
    )

    file.write(
        "-" * 75 + "\n"
    )

    file.write(
        "Assess whether identified red flags have a reasonable "
        "explanation based on the customer's known profile and "
        "expected activity.\n\n"
    )


    file.write(
        "STEP 5 - REVIEW MODEL EXPLANATION\n"
    )

    file.write(
        "-" * 75 + "\n"
    )

    file.write(
        "Use SHAP drivers to understand which behavioral features "
        "contributed to the model's risk prediction.\n\n"
    )


    file.write(
        "STEP 6 - CUSTOMER CONTEXT\n"
    )

    file.write(
        "-" * 75 + "\n"
    )

    file.write(
        "Consider whether the activity is consistent with the "
        "customer's occupation, income, expected volume, geography "
        "and known business purpose.\n\n"
    )


    file.write(
        "STEP 7 - DISPOSITION\n"
    )

    file.write(
        "-" * 75 + "\n"
    )

    file.write(
        "Possible analyst dispositions in this project include:\n"
    )

    file.write(
        "  - FALSE_POSITIVE\n"
    )

    file.write(
        "  - CLEARED\n"
    )

    file.write(
        "  - MONITOR\n"
    )

    file.write(
        "  - INVESTIGATE_FURTHER\n"
    )

    file.write(
        "  - ESCALATE\n\n"
    )


    file.write(
        "STEP 8 - SAR CONSIDERATION\n"
    )

    file.write(
        "-" * 75 + "\n"
    )

    file.write(
        "SAR consideration should be based on appropriate "
        "investigation, applicable requirements, institutional "
        "procedures and human review. A machine-learning score "
        "alone should not automatically determine SAR filing.\n\n"
    )


    file.write(
        "IMPORTANT MODEL GOVERNANCE NOTE\n"
    )

    file.write(
        "-" * 75 + "\n"
    )

    file.write(
        "Rules, thresholds and ML scores are risk indicators. "
        "They are not legal conclusions and should be reviewed "
        "within the institution's governance framework.\n"
    )


print(
    "\nSaved:"
)

print(
    ANALYST_GUIDE_FILE
)


# =============================================================================
# 15. TOP 20 CASES
# =============================================================================

print("\n" + "=" * 80)
print("15. TOP 20 INVESTIGATION CASES")
print("=" * 80)


display_columns = [
    "case_id",
    "customer_id",
    "case_priority",
    "case_risk_level",
    "highest_risk_score",
    "alert_count",
    "primary_alert_type",
    "red_flag_count",
    "model_rule_agreement"
]


display_columns = [
    column
    for column in display_columns
    if column in case_df.columns
]


if len(case_df) > 0:

    print(
        case_df[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

else:

    print(
        "\nNo investigation cases were created."
    )


# =============================================================================
# 16. FINAL STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("16. INVESTIGATION STATISTICS")
print("=" * 80)


print(
    f"\nTotal AML alerts: "
    f"{len(alerts):,}"
)


print(
    f"Unique customers with alerts: "
    f"{len(alert_customers):,}"
)


print(
    f"Investigation cases created: "
    f"{len(case_df):,}"
)


if len(case_df) > 0:

    print(
        "\nCase priority distribution:"
    )

    print(
        case_df[
            "case_priority"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nCase risk-level distribution:"
    )

    print(
        case_df[
            "case_risk_level"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nPrimary alert type distribution:"
    )

    print(
        case_df[
            "primary_alert_type"
        ]
        .value_counts()
        .to_string()
    )


# =============================================================================
# 17. OUTPUT FILES
# =============================================================================

print("\n" + "=" * 80)
print("17. PHASE 8 OUTPUT FILES")
print("=" * 80)


print(
    "\n1. Investigation cases:"
)

print(
    CASE_FILE
)


print(
    "\n2. Customer investigation profiles:"
)

print(
    CUSTOMER_PROFILE_FILE
)


print(
    "\n3. Transaction review summary:"
)

print(
    TRANSACTION_REVIEW_FILE
)


print(
    "\n4. Investigation red flags:"
)

print(
    RED_FLAG_FILE
)


print(
    "\n5. Case summary:"
)

print(
    CASE_SUMMARY_FILE
)


print(
    "\n6. Analyst investigation guide:"
)

print(
    ANALYST_GUIDE_FILE
)


# =============================================================================
# 18. COMPLETION
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 8 COMPLETED SUCCESSFULLY")
print("=" * 80)


print(
    "\nYour AML project now includes:"
)

print(
    "  [OK] AML alert investigation"
)

print(
    "  [OK] Customer investigation profiles"
)

print(
    "  [OK] Transaction activity review"
)

print(
    "  [OK] Red-flag identification"
)

print(
    "  [OK] SHAP model drivers"
)

print(
    "  [OK] AML investigation cases"
)

print(
    "  [OK] P1-P4 case prioritization"
)

print(
    "  [OK] Analyst disposition fields"
)

print(
    "  [OK] Escalation recommendation"
)

print(
    "  [OK] SAR consideration workflow"
)


print(
    "\nNext phase:"
)

print(
    "PHASE 9 - AML INVESTIGATION DASHBOARD"
)


print(
    "=" * 80
)