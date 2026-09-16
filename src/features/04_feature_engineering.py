"""
======================================================================
US FINANCIAL CRIME AI PROJECT
PHASE 4 - AML FEATURE ENGINEERING FOR MACHINE LEARNING
======================================================================

Purpose:
    Transform transaction-level AML rule-engine output into
    machine-learning-ready behavioral features.

Input:
    data/processed/aml_rule_engine_transactions.csv

Outputs:
    data/processed/aml_ml_features.csv
    data/processed/customer_ml_features.csv
    reports/features/feature_summary.csv
    reports/features/target_distribution.csv

Important:
    This is a synthetic portfolio/demo project.
    Thresholds and features are for analytical demonstration and
    should not be interpreted as legal or regulatory requirements.
======================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ======================================================================
# 1. PROJECT PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "aml_rule_engine_transactions.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 2. HELPER FUNCTIONS
# ======================================================================

def safe_divide(numerator, denominator):
    """
    Safely divide two pandas Series.

    Returns 0 when denominator is zero.
    """
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def print_section(title):
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


# ======================================================================
# 3. LOAD DATA
# ======================================================================

print_section("PHASE 4 - AML FEATURE ENGINEERING")

print(f"Project directory:\n{BASE_DIR}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n\n"
        "Please make sure Phase 3 completed successfully."
    )

print(f"\nLoading:\n{INPUT_FILE}")

transactions = pd.read_csv(INPUT_FILE)

print(f"\nLoaded {len(transactions):,} transaction records.")

print("\nOriginal columns:")
print(transactions.columns.tolist())


# ======================================================================
# 4. BASIC DATA CLEANING
# ======================================================================

print_section("4. DATA CLEANING")

transactions["timestamp"] = pd.to_datetime(
    transactions["timestamp"],
    errors="coerce"
)

transactions["amount_usd"] = pd.to_numeric(
    transactions["amount_usd"],
    errors="coerce"
).fillna(0)

transactions["customer_id"] = (
    transactions["customer_id"]
    .astype(str)
    .str.strip()
)

transactions["receiver_account"] = (
    transactions["receiver_account"]
    .astype(str)
    .str.strip()
)

transactions["origin_country"] = (
    transactions["origin_country"]
    .astype(str)
    .str.strip()
)

transactions["destination_country"] = (
    transactions["destination_country"]
    .astype(str)
    .str.strip()
)

transactions = transactions.dropna(
    subset=["timestamp", "customer_id"]
).copy()

transactions = transactions.sort_values(
    ["customer_id", "timestamp"]
).reset_index(drop=True)

print(f"Clean transaction records: {len(transactions):,}")

print(
    f"Date range: "
    f"{transactions['timestamp'].min()} "
    f"to "
    f"{transactions['timestamp'].max()}"
)


# ======================================================================
# 5. NORMALIZE IMPORTANT BINARY COLUMNS
# ======================================================================

print_section("5. NORMALIZING FLAGS")

binary_columns = [
    "is_international",
    "is_cash",
    "pep_flag",
    "sanctions_flag",
    "adverse_media_flag",
    "known_suspicious",
    "rule_alert",
]

for column in binary_columns:

    if column in transactions.columns:

        transactions[column] = pd.to_numeric(
            transactions[column],
            errors="coerce"
        ).fillna(0)

        transactions[column] = (
            transactions[column] > 0
        ).astype(int)


# ======================================================================
# 6. BASIC TRANSACTION FEATURES
# ======================================================================

print_section("6. BASIC TRANSACTION FEATURES")

transactions["log_amount"] = np.log1p(
    transactions["amount_usd"].clip(lower=0)
)

transactions["amount_squared"] = (
    transactions["amount_usd"] ** 2
)

transactions["is_large_transaction"] = (
    transactions["amount_usd"] >= 10000
).astype(int)

transactions["is_very_large_transaction"] = (
    transactions["amount_usd"] >= 25000
).astype(int)

transactions["is_small_cash_transaction"] = (
    (transactions["is_cash"] == 1)
    & (transactions["amount_usd"] < 10000)
).astype(int)


# ======================================================================
# 7. CUSTOMER TRANSACTION COUNT
# ======================================================================

print_section("7. CUSTOMER TRANSACTION COUNTS")

customer_group = transactions.groupby("customer_id")

transactions["customer_total_transactions"] = (
    customer_group["transaction_id"].transform("count")
)

transactions["customer_total_volume"] = (
    customer_group["amount_usd"].transform("sum")
)

transactions["customer_average_transaction"] = (
    customer_group["amount_usd"].transform("mean")
)

transactions["customer_max_transaction"] = (
    customer_group["amount_usd"].transform("max")
)

transactions["customer_min_transaction"] = (
    customer_group["amount_usd"].transform("min")
)

transactions["customer_transaction_std"] = (
    customer_group["amount_usd"]
    .transform("std")
    .fillna(0)
)


# ======================================================================
# 8. CUSTOMER TRANSACTION RATIOS
# ======================================================================

print_section("8. CUSTOMER BEHAVIOR RATIOS")

transactions["cash_transaction_count"] = (
    customer_group["is_cash"].transform("sum")
)

transactions["international_transaction_count"] = (
    customer_group["is_international"].transform("sum")
)

transactions["suspicious_transaction_count"] = (
    customer_group["known_suspicious"].transform("sum")
)

transactions["rule_alert_count"] = (
    customer_group["rule_alert"].transform("sum")
)


transactions["cash_ratio"] = safe_divide(
    transactions["cash_transaction_count"],
    transactions["customer_total_transactions"]
)

transactions["international_ratio"] = safe_divide(
    transactions["international_transaction_count"],
    transactions["customer_total_transactions"]
)

transactions["suspicious_transaction_ratio"] = safe_divide(
    transactions["suspicious_transaction_count"],
    transactions["customer_total_transactions"]
)

transactions["rule_alert_ratio"] = safe_divide(
    transactions["rule_alert_count"],
    transactions["customer_total_transactions"]
)


# ======================================================================
# 9. CUSTOMER COUNTERPARTY FEATURES
# ======================================================================

print_section("9. COUNTERPARTY FEATURES")

transactions["unique_counterparties"] = (
    customer_group["receiver_account"]
    .transform("nunique")
)

transactions["unique_origin_countries"] = (
    customer_group["origin_country"]
    .transform("nunique")
)

transactions["unique_destination_countries"] = (
    customer_group["destination_country"]
    .transform("nunique")
)

transactions["counterparty_diversity_ratio"] = safe_divide(
    transactions["unique_counterparties"],
    transactions["customer_total_transactions"]
)


# ======================================================================
# 10. TRANSACTION AMOUNT VS CUSTOMER BEHAVIOR
# ======================================================================

print_section("10. TRANSACTION AMOUNT DEVIATION")

transactions["amount_vs_customer_average"] = safe_divide(
    transactions["amount_usd"],
    transactions["customer_average_transaction"]
)

transactions["amount_vs_customer_max"] = safe_divide(
    transactions["amount_usd"],
    transactions["customer_max_transaction"]
)

transactions["amount_deviation_from_average"] = (
    transactions["amount_usd"]
    - transactions["customer_average_transaction"]
)

transactions["amount_zscore"] = safe_divide(
    transactions["amount_usd"]
    - transactions["customer_average_transaction"],
    transactions["customer_transaction_std"]
)


# ======================================================================
# 11. EXPECTED MONTHLY VOLUME FEATURES
# ======================================================================

print_section("11. EXPECTED VOLUME FEATURES")

if "expected_monthly_volume_usd" in transactions.columns:

    transactions["expected_monthly_volume_usd"] = pd.to_numeric(
        transactions["expected_monthly_volume_usd"],
        errors="coerce"
    ).fillna(0)

    transactions["amount_vs_expected_monthly"] = safe_divide(
        transactions["amount_usd"],
        transactions["expected_monthly_volume_usd"]
    )

    transactions["customer_volume_vs_expected"] = safe_divide(
        transactions["customer_total_volume"],
        transactions["expected_monthly_volume_usd"]
    )

else:

    transactions["amount_vs_expected_monthly"] = 0
    transactions["customer_volume_vs_expected"] = 0


# ======================================================================
# 12. CUSTOMER INCOME FEATURES
# ======================================================================

print_section("12. CUSTOMER INCOME FEATURES")

if "annual_income_usd" in transactions.columns:

    transactions["annual_income_usd"] = pd.to_numeric(
        transactions["annual_income_usd"],
        errors="coerce"
    ).fillna(0)

    transactions["monthly_income_estimate"] = (
        transactions["annual_income_usd"] / 12
    )

    transactions["transaction_vs_monthly_income"] = safe_divide(
        transactions["amount_usd"],
        transactions["monthly_income_estimate"]
    )

    transactions["annual_transaction_volume_vs_income"] = safe_divide(
        transactions["customer_total_volume"],
        transactions["annual_income_usd"]
    )

else:

    transactions["monthly_income_estimate"] = 0
    transactions["transaction_vs_monthly_income"] = 0
    transactions["annual_transaction_volume_vs_income"] = 0


# ======================================================================
# 13. TIME-BASED FEATURES
# ======================================================================

print_section("13. TIME FEATURES")

transactions["transaction_hour"] = (
    transactions["timestamp"].dt.hour
)

transactions["transaction_day_of_week"] = (
    transactions["timestamp"].dt.dayofweek
)

transactions["transaction_day"] = (
    transactions["timestamp"].dt.day
)

transactions["transaction_month"] = (
    transactions["timestamp"].dt.month
)

transactions["is_weekend"] = (
    transactions["transaction_day_of_week"] >= 5
).astype(int)

transactions["is_night_transaction"] = (
    (transactions["transaction_hour"] < 6)
    |
    (transactions["transaction_hour"] >= 22)
).astype(int)


# ======================================================================
# 14. TRANSACTION VELOCITY FEATURES
# ======================================================================

print_section("14. TRANSACTION VELOCITY")

transactions["previous_transaction_time"] = (
    transactions
    .groupby("customer_id")["timestamp"]
    .shift(1)
)

transactions["minutes_since_previous_transaction"] = (
    (
        transactions["timestamp"]
        - transactions["previous_transaction_time"]
    )
    .dt.total_seconds()
    / 60
)

transactions["minutes_since_previous_transaction"] = (
    transactions["minutes_since_previous_transaction"]
    .fillna(999999)
    .clip(lower=0)
)

transactions["rapid_transaction_indicator"] = (
    transactions["minutes_since_previous_transaction"] <= 120
).astype(int)


# ======================================================================
# 15. 24-HOUR SLIDING WINDOW
# ======================================================================

print_section("15. 24-HOUR ROLLING FEATURES")

transactions["transactions_24h"] = 0
transactions["volume_24h"] = 0.0
transactions["cash_transactions_24h"] = 0
transactions["international_transactions_24h"] = 0


for customer_id, group in transactions.groupby("customer_id", sort=False):

    indices = group.index.to_numpy()

    timestamps_ns = (
        group["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    amounts = (
        group["amount_usd"]
        .to_numpy(dtype=float)
    )

    cash_values = (
        group["is_cash"]
        .to_numpy(dtype=int)
    )

    international_values = (
        group["is_international"]
        .to_numpy(dtype=int)
    )

    window_ns = 24 * 60 * 60 * 1_000_000_000

    for position in range(len(indices)):

        start_time = (
            timestamps_ns[position]
            - window_ns
        )

        start_position = np.searchsorted(
            timestamps_ns,
            start_time,
            side="left"
        )

        end_position = position + 1

        transactions.loc[
            indices[position],
            "transactions_24h"
        ] = end_position - start_position

        transactions.loc[
            indices[position],
            "volume_24h"
        ] = amounts[
            start_position:end_position
        ].sum()

        transactions.loc[
            indices[position],
            "cash_transactions_24h"
        ] = cash_values[
            start_position:end_position
        ].sum()

        transactions.loc[
            indices[position],
            "international_transactions_24h"
        ] = international_values[
            start_position:end_position
        ].sum()


# ======================================================================
# 16. 7-DAY AND 30-DAY ROLLING FEATURES
# ======================================================================

print_section("16. 7-DAY AND 30-DAY ROLLING FEATURES")

transactions["transactions_7d"] = 0
transactions["volume_7d"] = 0.0

transactions["transactions_30d"] = 0
transactions["volume_30d"] = 0.0


for customer_id, group in transactions.groupby("customer_id", sort=False):

    indices = group.index.to_numpy()

    timestamps_ns = (
        group["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    amounts = (
        group["amount_usd"]
        .to_numpy(dtype=float)
    )

    seven_days_ns = (
        7 * 24 * 60 * 60 * 1_000_000_000
    )

    thirty_days_ns = (
        30 * 24 * 60 * 60 * 1_000_000_000
    )

    for position in range(len(indices)):

        current_time = timestamps_ns[position]

        start_7d = current_time - seven_days_ns
        start_30d = current_time - thirty_days_ns

        start_position_7d = np.searchsorted(
            timestamps_ns,
            start_7d,
            side="left"
        )

        start_position_30d = np.searchsorted(
            timestamps_ns,
            start_30d,
            side="left"
        )

        end_position = position + 1

        transactions.loc[
            indices[position],
            "transactions_7d"
        ] = end_position - start_position_7d

        transactions.loc[
            indices[position],
            "volume_7d"
        ] = amounts[
            start_position_7d:end_position
        ].sum()

        transactions.loc[
            indices[position],
            "transactions_30d"
        ] = end_position - start_position_30d

        transactions.loc[
            indices[position],
            "volume_30d"
        ] = amounts[
            start_position_30d:end_position
        ].sum()


# ======================================================================
# 17. ROLLING RATIOS
# ======================================================================

print_section("17. ROLLING RATIOS")

transactions["cash_ratio_24h"] = safe_divide(
    transactions["cash_transactions_24h"],
    transactions["transactions_24h"]
)

transactions["international_ratio_24h"] = safe_divide(
    transactions["international_transactions_24h"],
    transactions["transactions_24h"]
)

transactions["average_transaction_24h"] = safe_divide(
    transactions["volume_24h"],
    transactions["transactions_24h"]
)

transactions["average_transaction_7d"] = safe_divide(
    transactions["volume_7d"],
    transactions["transactions_7d"]
)

transactions["average_transaction_30d"] = safe_divide(
    transactions["volume_30d"],
    transactions["transactions_30d"]
)


# ======================================================================
# 18. STRUCTURING FEATURES
# ======================================================================

print_section("18. STRUCTURING FEATURES")

transactions["near_10000_threshold"] = (
    (
        transactions["amount_usd"] >= 8000
    )
    &
    (
        transactions["amount_usd"] < 10000
    )
).astype(int)

transactions["structuring_customer_count"] = (
    transactions.groupby("customer_id")[
        "near_10000_threshold"
    ].transform("sum")
)

transactions["structuring_ratio"] = safe_divide(
    transactions["structuring_customer_count"],
    transactions["customer_total_transactions"]
)

transactions["multiple_near_threshold"] = (
    transactions["structuring_customer_count"] >= 3
).astype(int)


# ======================================================================
# 19. RAPID MOVEMENT FEATURES
# ======================================================================

print_section("19. RAPID MOVEMENT FEATURES")

transactions["rapid_movement_flag"] = (
    (
        transactions["minutes_since_previous_transaction"] <= 120
    )
    &
    (
        transactions["amount_usd"] >= 5000
    )
).astype(int)

transactions["rapid_movement_customer_count"] = (
    transactions.groupby("customer_id")[
        "rapid_movement_flag"
    ].transform("sum")
)

transactions["rapid_movement_ratio"] = safe_divide(
    transactions["rapid_movement_customer_count"],
    transactions["customer_total_transactions"]
)


# ======================================================================
# 20. RED FLAG FEATURES
# ======================================================================

print_section("20. RED FLAG FEATURES")

if "red_flag_count" in transactions.columns:

    transactions["red_flag_count"] = pd.to_numeric(
        transactions["red_flag_count"],
        errors="coerce"
    ).fillna(0)

else:

    transactions["red_flag_count"] = 0


transactions["customer_average_red_flags"] = (
    transactions.groupby("customer_id")[
        "red_flag_count"
    ].transform("mean")
)

transactions["customer_max_red_flags"] = (
    transactions.groupby("customer_id")[
        "red_flag_count"
    ].transform("max")
)


# ======================================================================
# 21. RULE ENGINE RISK FEATURES
# ======================================================================

print_section("21. RULE ENGINE FEATURES")

if "rule_risk_score" in transactions.columns:

    transactions["rule_risk_score"] = pd.to_numeric(
        transactions["rule_risk_score"],
        errors="coerce"
    ).fillna(0)

else:

    transactions["rule_risk_score"] = 0


transactions["customer_average_rule_score"] = (
    transactions.groupby("customer_id")[
        "rule_risk_score"
    ].transform("mean")
)

transactions["customer_max_rule_score"] = (
    transactions.groupby("customer_id")[
        "rule_risk_score"
    ].transform("max")
)

transactions["high_risk_rule_indicator"] = (
    transactions["rule_risk_score"] >= 50
).astype(int)


# ======================================================================
# 22. CUSTOMER RISK PROFILE FEATURES
# ======================================================================

print_section("22. CUSTOMER RISK PROFILE")

if "risk_rating" in transactions.columns:

    risk_map = {
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2,
        "CRITICAL": 3,
    }

    transactions["customer_risk_numeric"] = (
        transactions["risk_rating"]
        .astype(str)
        .str.upper()
        .map(risk_map)
        .fillna(0)
    )

else:

    transactions["customer_risk_numeric"] = 0


transactions["high_customer_risk"] = (
    transactions["customer_risk_numeric"] >= 2
).astype(int)


# ======================================================================
# 23. PEP / SANCTIONS / ADVERSE MEDIA FEATURES
# ======================================================================

print_section("23. KYC / CUSTOMER RISK FEATURES")

transactions["high_risk_customer_profile"] = (
    (
        transactions["pep_flag"] == 1
    )
    |
    (
        transactions["sanctions_flag"] == 1
    )
    |
    (
        transactions["adverse_media_flag"] == 1
    )
).astype(int)

transactions["customer_risk_indicator_count"] = (
    transactions["pep_flag"]
    +
    transactions["sanctions_flag"]
    +
    transactions["adverse_media_flag"]
)


# ======================================================================
# 24. FUNNEL / LAYERING FEATURES
# ======================================================================

print_section("24. FUNNEL AND LAYERING FEATURES")

transactions["funnel_behavior_score"] = (
    (
        transactions["unique_origin_countries"] >= 3
    )
    &
    (
        transactions["transactions_30d"] >= 10
    )
).astype(int)

transactions["layering_behavior_score"] = (
    (
        transactions["rapid_movement_flag"] == 1
    )
    &
    (
        transactions["unique_counterparties"] >= 8
    )
).astype(int)


# ======================================================================
# 25. MULE BEHAVIOR FEATURES
# ======================================================================

print_section("25. MULE BEHAVIOR FEATURES")

transactions["mule_behavior_indicator"] = (
    (
        transactions["customer_total_transactions"] >= 15
    )
    &
    (
        transactions["customer_average_transaction"] >= 5000
    )
).astype(int)


# ======================================================================
# 26. TRANSACTION CONCENTRATION
# ======================================================================

print_section("26. TRANSACTION CONCENTRATION")

transactions["customer_volume_share"] = safe_divide(
    transactions["amount_usd"],
    transactions["customer_total_volume"]
)

transactions["top_transaction_indicator"] = (
    transactions["amount_usd"]
    ==
    transactions["customer_max_transaction"]
).astype(int)


# ======================================================================
# 27. ACCOUNT / CHANNEL FEATURES
# ======================================================================

print_section("27. CHANNEL FEATURES")

if "channel" in transactions.columns:

    channel_counts = (
        transactions.groupby(
            ["customer_id", "channel"]
        )["transaction_id"]
        .transform("count")
    )

    transactions["channel_transaction_count"] = (
        channel_counts
    )

    transactions["channel_transaction_ratio"] = safe_divide(
        transactions["channel_transaction_count"],
        transactions["customer_total_transactions"]
    )

else:

    transactions["channel_transaction_count"] = 0
    transactions["channel_transaction_ratio"] = 0


# ======================================================================
# 28. LABEL CREATION
# ======================================================================

print_section("28. MACHINE LEARNING TARGET")

"""
known_suspicious is the synthetic ground-truth label generated during
dataset creation.

1 = suspicious transaction
0 = normal transaction

This is NOT a real-world SAR filing label.
"""

if "known_suspicious" not in transactions.columns:

    raise ValueError(
        "Column 'known_suspicious' is required for ML target creation."
    )

transactions["target"] = (
    transactions["known_suspicious"]
    .astype(int)
)


# ======================================================================
# 29. FINAL FEATURE LIST
# ======================================================================

print_section("29. SELECTING ML FEATURES")

feature_columns = [
    # Transaction amount
    "amount_usd",
    "log_amount",
    "amount_squared",
    "is_large_transaction",
    "is_very_large_transaction",
    "is_small_cash_transaction",

    # Customer behavior
    "customer_total_transactions",
    "customer_total_volume",
    "customer_average_transaction",
    "customer_max_transaction",
    "customer_min_transaction",
    "customer_transaction_std",

    # Ratios
    "cash_ratio",
    "international_ratio",
    "suspicious_transaction_ratio",
    "rule_alert_ratio",

    # Counterparties
    "unique_counterparties",
    "unique_origin_countries",
    "unique_destination_countries",
    "counterparty_diversity_ratio",

    # Amount deviation
    "amount_vs_customer_average",
    "amount_vs_customer_max",
    "amount_deviation_from_average",
    "amount_zscore",

    # Expected behavior
    "amount_vs_expected_monthly",
    "customer_volume_vs_expected",

    # Income behavior
    "transaction_vs_monthly_income",
    "annual_transaction_volume_vs_income",

    # Time
    "transaction_hour",
    "transaction_day_of_week",
    "transaction_day",
    "transaction_month",
    "is_weekend",
    "is_night_transaction",

    # Velocity
    "minutes_since_previous_transaction",
    "rapid_transaction_indicator",

    # 24h
    "transactions_24h",
    "volume_24h",
    "cash_transactions_24h",
    "international_transactions_24h",
    "cash_ratio_24h",
    "international_ratio_24h",
    "average_transaction_24h",

    # 7d
    "transactions_7d",
    "volume_7d",
    "average_transaction_7d",

    # 30d
    "transactions_30d",
    "volume_30d",
    "average_transaction_30d",

    # Structuring
    "near_10000_threshold",
    "structuring_customer_count",
    "structuring_ratio",
    "multiple_near_threshold",

    # Rapid movement
    "rapid_movement_flag",
    "rapid_movement_customer_count",
    "rapid_movement_ratio",

    # Red flags
    "red_flag_count",
    "customer_average_red_flags",
    "customer_max_red_flags",

    # Rule engine
    "rule_risk_score",
    "customer_average_rule_score",
    "customer_max_rule_score",
    "high_risk_rule_indicator",

    # Customer risk
    "customer_risk_numeric",
    "high_customer_risk",

    # KYC risk
    "pep_flag",
    "sanctions_flag",
    "adverse_media_flag",
    "high_risk_customer_profile",
    "customer_risk_indicator_count",

    # AML typology behavior
    "funnel_behavior_score",
    "layering_behavior_score",
    "mule_behavior_indicator",

    # Concentration
    "customer_volume_share",
    "top_transaction_indicator",

    # Channel
    "channel_transaction_count",
    "channel_transaction_ratio",
]


# Keep only columns that actually exist
feature_columns = [
    column
    for column in feature_columns
    if column in transactions.columns
]


print(
    f"\nNumber of ML features created: "
    f"{len(feature_columns)}"
)


# ======================================================================
# 30. CREATE ML DATASET
# ======================================================================

print_section("30. CREATING ML DATASET")

identifier_columns = [
    "transaction_id",
    "customer_id",
    "account_id",
    "sender_account",
    "receiver_account",
    "timestamp",
]

metadata_columns = [
    "scenario",
    "transaction_type",
    "channel",
    "origin_country",
    "destination_country",
]

target_columns = [
    "known_suspicious",
    "target",
]


final_columns = []

for column in (
    identifier_columns
    + metadata_columns
    + feature_columns
    + target_columns
):

    if column in transactions.columns and column not in final_columns:
        final_columns.append(column)


ml_dataset = transactions[final_columns].copy()


# ======================================================================
# 31. HANDLE INFINITE / MISSING VALUES
# ======================================================================

print_section("31. FINAL DATA QUALITY CHECK")

numeric_feature_columns = [
    column
    for column in feature_columns
    if column in ml_dataset.columns
]

ml_dataset[numeric_feature_columns] = (
    ml_dataset[numeric_feature_columns]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


missing_features = (
    ml_dataset[numeric_feature_columns]
    .isnull()
    .sum()
    .sum()
)

print(
    f"Remaining missing feature values: "
    f"{missing_features}"
)


# ======================================================================
# 32. CUSTOMER-LEVEL DATASET
# ======================================================================

print_section("32. CUSTOMER-LEVEL FEATURE DATASET")

customer_features = (
    transactions
    .groupby("customer_id")
    .agg(
        total_transactions=(
            "transaction_id",
            "count"
        ),

        total_transaction_volume=(
            "amount_usd",
            "sum"
        ),

        average_transaction=(
            "amount_usd",
            "mean"
        ),

        maximum_transaction=(
            "amount_usd",
            "max"
        ),

        transaction_std=(
            "amount_usd",
            "std"
        ),

        cash_transactions=(
            "is_cash",
            "sum"
        ),

        international_transactions=(
            "is_international",
            "sum"
        ),

        suspicious_transactions=(
            "known_suspicious",
            "sum"
        ),

        rule_alerts=(
            "rule_alert",
            "sum"
        ),

        unique_counterparties=(
            "receiver_account",
            "nunique"
        ),

        unique_origin_countries=(
            "origin_country",
            "nunique"
        ),

        unique_destination_countries=(
            "destination_country",
            "nunique"
        ),

        average_rule_score=(
            "rule_risk_score",
            "mean"
        ),

        maximum_rule_score=(
            "rule_risk_score",
            "max"
        ),

        average_red_flags=(
            "red_flag_count",
            "mean"
        ),

        maximum_red_flags=(
            "red_flag_count",
            "max"
        ),

        PEP_flag=(
            "pep_flag",
            "max"
        ),

        sanctions_flag=(
            "sanctions_flag",
            "max"
        ),

        adverse_media_flag=(
            "adverse_media_flag",
            "max"
        ),

        structuring_events=(
            "near_10000_threshold",
            "sum"
        ),

        rapid_movement_events=(
            "rapid_movement_flag",
            "sum"
        ),

        funnel_behavior=(
            "funnel_behavior_score",
            "max"
        ),

        layering_behavior=(
            "layering_behavior_score",
            "max"
        ),

        mule_behavior=(
            "mule_behavior_indicator",
            "max"
        ),

        target=(
            "target",
            "max"
        ),
    )
    .reset_index()
)


customer_features["cash_ratio"] = safe_divide(
    customer_features["cash_transactions"],
    customer_features["total_transactions"]
)

customer_features["international_ratio"] = safe_divide(
    customer_features["international_transactions"],
    customer_features["total_transactions"]
)

customer_features["suspicious_ratio"] = safe_divide(
    customer_features["suspicious_transactions"],
    customer_features["total_transactions"]
)

customer_features["rule_alert_ratio"] = safe_divide(
    customer_features["rule_alerts"],
    customer_features["total_transactions"]
)

customer_features["structuring_ratio"] = safe_divide(
    customer_features["structuring_events"],
    customer_features["total_transactions"]
)

customer_features["rapid_movement_ratio"] = safe_divide(
    customer_features["rapid_movement_events"],
    customer_features["total_transactions"]
)


customer_features = (
    customer_features
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ======================================================================
# 33. FEATURE SUMMARY
# ======================================================================

print_section("33. FEATURE SUMMARY")

feature_summary = pd.DataFrame({
    "feature": numeric_feature_columns,
    "dtype": [
        str(ml_dataset[column].dtype)
        for column in numeric_feature_columns
    ],
    "missing_values": [
        ml_dataset[column].isna().sum()
        for column in numeric_feature_columns
    ],
    "unique_values": [
        ml_dataset[column].nunique()
        for column in numeric_feature_columns
    ],
    "mean": [
        ml_dataset[column].mean()
        for column in numeric_feature_columns
    ],
    "std": [
        ml_dataset[column].std()
        for column in numeric_feature_columns
    ],
    "min": [
        ml_dataset[column].min()
        for column in numeric_feature_columns
    ],
    "max": [
        ml_dataset[column].max()
        for column in numeric_feature_columns
    ],
})


# ======================================================================
# 34. TARGET DISTRIBUTION
# ======================================================================

target_distribution = (
    ml_dataset["target"]
    .value_counts()
    .sort_index()
    .rename_axis("target")
    .reset_index(name="transaction_count")
)

target_distribution["percentage"] = (
    target_distribution["transaction_count"]
    / len(ml_dataset)
    * 100
)


# ======================================================================
# 35. SAVE OUTPUTS
# ======================================================================

print_section("35. SAVING OUTPUT FILES")

ml_output = (
    OUTPUT_DIR
    / "aml_ml_features.csv"
)

customer_output = (
    OUTPUT_DIR
    / "customer_ml_features.csv"
)

feature_summary_output = (
    REPORT_DIR
    / "feature_summary.csv"
)

target_output = (
    REPORT_DIR
    / "target_distribution.csv"
)


ml_dataset.to_csv(
    ml_output,
    index=False
)

customer_features.to_csv(
    customer_output,
    index=False
)

feature_summary.to_csv(
    feature_summary_output,
    index=False
)

target_distribution.to_csv(
    target_output,
    index=False
)


# ======================================================================
# 36. FINAL REPORT
# ======================================================================

print_section("PHASE 4 SUMMARY")

print(
    f"Transactions processed: "
    f"{len(ml_dataset):,}"
)

print(
    f"Customers processed: "
    f"{len(customer_features):,}"
)

print(
    f"ML features created: "
    f"{len(feature_columns)}"
)

print(
    f"Suspicious transactions: "
    f"{int(ml_dataset['target'].sum()):,}"
)

print(
    f"Normal transactions: "
    f"{int((ml_dataset['target'] == 0).sum()):,}"
)

suspicious_percentage = (
    ml_dataset["target"].mean() * 100
)

print(
    f"Suspicious transaction percentage: "
    f"{suspicious_percentage:.2f}%"
)

print("\nOutput files:")

print(
    f"1. {ml_output}"
)

print(
    f"2. {customer_output}"
)

print(
    f"3. {feature_summary_output}"
)

print(
    f"4. {target_output}"
)


# ======================================================================
# 37. SAMPLE FEATURES
# ======================================================================

print_section("SAMPLE ML FEATURES")

display_columns = [
    "transaction_id",
    "customer_id",
    "amount_usd",
    "transactions_24h",
    "volume_24h",
    "transactions_7d",
    "volume_7d",
    "transactions_30d",
    "volume_30d",
    "unique_counterparties",
    "cash_ratio",
    "international_ratio",
    "structuring_ratio",
    "rapid_movement_flag",
    "red_flag_count",
    "rule_risk_score",
    "target",
]

display_columns = [
    column
    for column in display_columns
    if column in ml_dataset.columns
]

print(
    ml_dataset[display_columns]
    .head(10)
    .to_string(index=False)
)


# ======================================================================
# 38. COMPLETION
# ======================================================================

print("\n")
print("=" * 80)
print("PHASE 4 COMPLETED SUCCESSFULLY")
print("=" * 80)

print(
    "\nThe AML transaction data has been transformed into "
    "machine-learning-ready behavioral features."
)

print(
    "\nNext phase:"
)

print(
    "PHASE 5 - AML MACHINE LEARNING MODEL"
)

print(
    "\nWe will train and compare AML classification models "
    "using the engineered features."
)

print("=" * 80)