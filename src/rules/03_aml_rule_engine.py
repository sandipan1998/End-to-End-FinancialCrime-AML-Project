"""
======================================================================
US FINANCIAL CRIME AI PROJECT
PHASE 3 - AML RULE ENGINE & RED FLAG DETECTION
======================================================================

Project:
US FINANCIAL CRIME AI PROJECT

Purpose:
Production-style AML transaction monitoring rule engine.

This module applies AML red-flag rules to synthetic financial
institution transaction data.

Detection areas:
1. Structuring / Smurfing
2. High-value transactions
3. Customer profile mismatch
4. International activity
5. High-risk country exposure
6. High transaction velocity
7. Rapid movement of funds
8. Multiple counterparties
9. Cash-intensive activity
10. PEP activity
11. Sanctions indicators
12. Adverse media indicators
13. Mule-account behavior
14. Layering behavior
15. Funnel-account behavior

IMPORTANT:
A red flag is an indicator requiring investigation.
A red flag is NOT proof of criminal activity.

This project uses synthetic data for educational/portfolio purposes.
======================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# 1. PROJECT PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "US_AML_Synthetic_Dataset.xlsx"
)

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "rules"
)


PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# 2. PROJECT HEADER
# ======================================================================

print("=" * 70)
print("US FINANCIAL CRIME AI PROJECT")
print("PHASE 3 - AML RULE ENGINE & RED FLAG DETECTION")
print("=" * 70)


# ======================================================================
# 3. CHECK INPUT FILE
# ======================================================================

print("\nChecking AML dataset...")


if not DATA_FILE.exists():

    raise FileNotFoundError(
        "\nAML dataset was not found.\n\n"
        f"Expected location:\n{DATA_FILE}\n\n"
        "Please make sure that "
        "US_AML_Synthetic_Dataset.xlsx "
        "exists inside data\\raw."
    )


print("Dataset found:")
print(DATA_FILE)


# ======================================================================
# 4. LOAD DATA
# ======================================================================

print("\nLoading transaction and customer data...")


transactions = pd.read_excel(
    DATA_FILE,
    sheet_name="Transactions"
)


customers = pd.read_excel(
    DATA_FILE,
    sheet_name="Customers"
)


print(
    f"Transactions loaded: {len(transactions):,}"
)

print(
    f"Customers loaded: {len(customers):,}"
)


# ======================================================================
# 5. BASIC DATA PREPARATION
# ======================================================================

print("\nPreparing transaction data...")


# Convert timestamp

transactions["timestamp"] = pd.to_datetime(
    transactions["timestamp"],
    errors="coerce"
)


# Convert amount

transactions["amount_usd"] = pd.to_numeric(
    transactions["amount_usd"],
    errors="coerce"
)


# Convert IDs to strings

transactions["customer_id"] = (
    transactions["customer_id"]
    .astype(str)
)

customers["customer_id"] = (
    customers["customer_id"]
    .astype(str)
)


# Remove invalid timestamps

invalid_timestamp_count = int(
    transactions["timestamp"].isna().sum()
)


if invalid_timestamp_count > 0:

    print(
        "Removing invalid timestamp rows:",
        invalid_timestamp_count
    )

    transactions = transactions.dropna(
        subset=["timestamp"]
    ).copy()


# Create a clean index

transactions = (
    transactions
    .reset_index(drop=True)
)


# ======================================================================
# 6. CUSTOMER PROFILE
# ======================================================================

print("\nPreparing customer profile...")


customer_columns = [

    "customer_id",

    "customer_type",

    "annual_income_usd",

    "expected_monthly_volume_usd",

    "pep_flag",

    "sanctions_flag",

    "adverse_media_flag",

    "risk_rating"

]


available_customer_columns = [

    column

    for column in customer_columns

    if column in customers.columns

]


customer_profile = (
    customers[
        available_customer_columns
    ]
    .drop_duplicates(
        subset=["customer_id"]
    )
    .copy()
)


# ======================================================================
# 7. MERGE CUSTOMER INFORMATION
# ======================================================================

transactions = transactions.merge(

    customer_profile,

    on="customer_id",

    how="left",

    suffixes=(
        "",
        "_customer"
    )

)


# Reset index after merge

transactions = (
    transactions
    .reset_index(drop=True)
)


print(
    "Customer information merged successfully."
)


# ======================================================================
# 8. HELPER FUNCTION
# ======================================================================

def flag_column(condition):
    """
    Convert a Boolean condition into integer 0/1.
    """

    return (
        condition
        .fillna(False)
        .astype(int)
    )


# ======================================================================
# RULE 1 - STRUCTURING
# ======================================================================

print(
    "\nApplying Rule 1 - Structuring..."
)


transactions[
    "RF_STRUCTURING"
] = flag_column(

    (
        transactions[
            "amount_usd"
        ] >= 8000
    )

    &

    (
        transactions[
            "amount_usd"
        ] < 10000
    )

)


# ======================================================================
# RULE 2 - HIGH VALUE TRANSACTION
# ======================================================================

print(
    "Applying Rule 2 - High-value transaction..."
)


transactions[
    "RF_HIGH_VALUE"
] = flag_column(

    transactions[
        "amount_usd"
    ] >= 25000

)


# ======================================================================
# RULE 3 - CUSTOMER PROFILE MISMATCH
# ======================================================================

print(
    "Applying Rule 3 - Customer profile mismatch..."
)


if (
    "expected_monthly_volume_usd"
    in transactions.columns
):

    expected_volume = pd.to_numeric(

        transactions[
            "expected_monthly_volume_usd"
        ],

        errors="coerce"

    )


    transactions[
        "RF_PROFILE_MISMATCH"
    ] = flag_column(

        transactions[
            "amount_usd"
        ]

        >

        (
            expected_volume
            * 3
        )

    )

else:

    transactions[
        "RF_PROFILE_MISMATCH"
    ] = 0


# ======================================================================
# RULE 4 - INTERNATIONAL ACTIVITY
# ======================================================================

print(
    "Applying Rule 4 - International activity..."
)


if (
    "is_international"
    in transactions.columns
):

    transactions[
        "RF_INTERNATIONAL"
    ] = flag_column(

        transactions[
            "is_international"
        ] == 1

    )

else:

    transactions[
        "RF_INTERNATIONAL"
    ] = 0


# ======================================================================
# RULE 5 - HIGH-RISK COUNTRY
# ======================================================================

print(
    "Applying Rule 5 - High-risk country exposure..."
)


# Synthetic demonstration list.
#
# This is NOT a statement that these countries are legally
# designated as universally "high risk" under US law.
#
# In a production environment, country risk should come from
# current institution-approved compliance data.

HIGH_RISK_COUNTRIES = {

    "IR",

    "KP",

    "SY",

    "MM"

}


origin_high_risk = (

    transactions[
        "origin_country"
    ]
    .isin(
        HIGH_RISK_COUNTRIES
    )

)


destination_high_risk = (

    transactions[
        "destination_country"
    ]
    .isin(
        HIGH_RISK_COUNTRIES
    )

)


transactions[
    "RF_HIGH_RISK_COUNTRY"
] = flag_column(

    origin_high_risk

    |

    destination_high_risk

)


# ======================================================================
# RULE 6 - 24-HOUR TRANSACTION VELOCITY
# ======================================================================

print(
    "Applying Rule 6 - Transaction velocity..."
)


def calculate_24h_transaction_count(group):
    """
    Calculate the number of transactions for the same customer
    occurring within the previous 24 hours.

    This implementation deliberately avoids pandas rolling()
    assignment issues caused by MultiIndex / duplicate labels.
    """

    group = (
        group
        .sort_values(
            "timestamp"
        )
        .copy()
    )


    # Convert timestamp to nanoseconds.

    timestamp_values = (
        group[
            "timestamp"
        ]
        .astype("int64")
        .to_numpy()
    )


    # 24 hours in nanoseconds.

    window_ns = (
        24
        * 60
        * 60
        * 1_000_000_000
    )


    counts = np.zeros(
        len(group),
        dtype=int
    )


    for position in range(
        len(group)
    ):

        current_time = (
            timestamp_values[
                position
            ]
        )


        window_start = (
            current_time
            -
            window_ns
        )


        left_position = np.searchsorted(

            timestamp_values,

            window_start,

            side="left"

        )


        counts[position] = (

            position
            -
            left_position
            +
            1

        )


    group[
        "transactions_24h"
    ] = counts


    return group


# ======================================================================
# CALCULATE VELOCITY
# ======================================================================

velocity_results = []


for customer_id, group in transactions.groupby(

    "customer_id",

    sort=False

):

    result = (
        calculate_24h_transaction_count(
            group
        )
    )


    velocity_results.append(

        result[
            [
                "customer_id",

                "transaction_id",

                "transactions_24h"

            ]
        ]

    )


if len(velocity_results) > 0:

    velocity_df = pd.concat(

        velocity_results,

        ignore_index=True

    )

else:

    velocity_df = pd.DataFrame(

        columns=[
            "customer_id",
            "transaction_id",
            "transactions_24h"
        ]

    )


# Make sure transaction IDs are unique.

velocity_df = (
    velocity_df
    .drop_duplicates(
        subset=[
            "customer_id",
            "transaction_id"
        ]
    )
)


# ======================================================================
# MERGE VELOCITY BACK
# ======================================================================

transactions = transactions.merge(

    velocity_df,

    on=[
        "customer_id",

        "transaction_id"
    ],

    how="left",

    suffixes=(
        "",
        "_velocity"
    )

)


transactions[
    "transactions_24h"
] = (

    transactions[
        "transactions_24h"
    ]

    .fillna(1)

    .astype(int)

)


transactions[
    "RF_HIGH_VELOCITY"
] = flag_column(

    transactions[
        "transactions_24h"
    ] >= 10

)


transactions = (
    transactions
    .reset_index(drop=True)
)


# ======================================================================
# RULE 7 - RAPID MOVEMENT OF FUNDS
# ======================================================================

print(
    "Applying Rule 7 - Rapid movement of funds..."
)


transactions[
    "RF_RAPID_MOVEMENT"
] = 0


for customer_id, group in transactions.groupby(
    "customer_id"
):

    group = (
        group
        .sort_values(
            "timestamp"
        )
    )


    previous_timestamp = None

    previous_amount = None


    for idx, row in group.iterrows():

        current_timestamp = (
            row["timestamp"]
        )


        current_amount = (
            row["amount_usd"]
        )


        if (

            previous_timestamp is not None

            and

            previous_amount is not None

        ):

            time_difference = (

                current_timestamp
                -
                previous_timestamp

            ).total_seconds() / 3600


            if (

                time_difference <= 2

                and

                current_amount >= 5000

                and

                previous_amount >= 5000

            ):

                transactions.loc[
                    idx,
                    "RF_RAPID_MOVEMENT"
                ] = 1


        previous_timestamp = (
            current_timestamp
        )


        previous_amount = (
            current_amount
        )


# ======================================================================
# RULE 8 - MULTIPLE COUNTERPARTIES
# ======================================================================

print(
    "Applying Rule 8 - Multiple counterparties..."
)


if (
    "receiver_account"
    in transactions.columns
):

    transactions[
        "counterparties_30d"
    ] = (

        transactions

        .groupby(
            "customer_id"
        )

        [
            "receiver_account"
        ]

        .transform(
            "nunique"
        )

    )

else:

    transactions[
        "counterparties_30d"
    ] = 0


transactions[
    "RF_MULTIPLE_COUNTERPARTIES"
] = flag_column(

    transactions[
        "counterparties_30d"
    ] >= 8

)


# ======================================================================
# RULE 9 - CASH ACTIVITY
# ======================================================================

print(
    "Applying Rule 9 - Cash-intensive activity..."
)


if (
    "is_cash"
    in transactions.columns
):

    transactions[
        "RF_CASH_ACTIVITY"
    ] = flag_column(

        transactions[
            "is_cash"
        ] == 1

    )

else:

    transactions[
        "RF_CASH_ACTIVITY"
    ] = 0


# ======================================================================
# RULE 10 - PEP ACTIVITY
# ======================================================================

print(
    "Applying Rule 10 - PEP activity..."
)


if (
    "pep_flag"
    in transactions.columns
):

    transactions[
        "RF_PEP"
    ] = flag_column(

        transactions[
            "pep_flag"
        ] == 1

    )

else:

    transactions[
        "RF_PEP"
    ] = 0


# ======================================================================
# RULE 11 - SANCTIONS INDICATOR
# ======================================================================

print(
    "Applying Rule 11 - Sanctions indicator..."
)


if (
    "sanctions_flag"
    in transactions.columns
):

    transactions[
        "RF_SANCTIONS"
    ] = flag_column(

        transactions[
            "sanctions_flag"
        ] == 1

    )

else:

    transactions[
        "RF_SANCTIONS"
    ] = 0


# ======================================================================
# RULE 12 - ADVERSE MEDIA
# ======================================================================

print(
    "Applying Rule 12 - Adverse media..."
)


if (
    "adverse_media_flag"
    in transactions.columns
):

    transactions[
        "RF_ADVERSE_MEDIA"
    ] = flag_column(

        transactions[
            "adverse_media_flag"
        ] == 1

    )

else:

    transactions[
        "RF_ADVERSE_MEDIA"
    ] = 0


# ======================================================================
# RULE 13 - MULE ACCOUNT BEHAVIOR
# ======================================================================

print(
    "Applying Rule 13 - Mule-account behavior..."
)


transaction_count_by_customer = (

    transactions

    .groupby(
        "customer_id"
    )

    .size()

)


transactions[
    "customer_transaction_count"
] = (

    transactions[
        "customer_id"
    ]

    .map(
        transaction_count_by_customer
    )

    .fillna(0)

)


transactions[
    "RF_MULE_BEHAVIOR"
] = flag_column(

    (

        transactions[
            "customer_transaction_count"
        ] >= 15

    )

    &

    (

        transactions[
            "amount_usd"
        ] >= 5000

    )

)


# ======================================================================
# RULE 14 - LAYERING
# ======================================================================

print(
    "Applying Rule 14 - Layering behavior..."
)


transactions[
    "RF_LAYERING"
] = flag_column(

    (

        transactions[
            "RF_RAPID_MOVEMENT"
        ] == 1

    )

    &

    (

        transactions[
            "RF_MULTIPLE_COUNTERPARTIES"
        ] == 1

    )

)


# ======================================================================
# RULE 15 - FUNNEL ACCOUNT
# ======================================================================

print(
    "Applying Rule 15 - Funnel-account behavior..."
)


origin_country_count = (

    transactions

    .groupby(
        "customer_id"
    )

    [
        "origin_country"
    ]

    .transform(
        "nunique"
    )

)


transactions[
    "origin_country_count"
] = origin_country_count


transactions[
    "RF_FUNNEL_ACCOUNT"
] = flag_column(

    (

        transactions[
            "origin_country_count"
        ] >= 3

    )

    &

    (

        transactions[
            "customer_transaction_count"
        ] >= 10

    )

)


# ======================================================================
# 16. RED FLAG COLUMNS
# ======================================================================

RED_FLAG_COLUMNS = [

    "RF_STRUCTURING",

    "RF_HIGH_VALUE",

    "RF_PROFILE_MISMATCH",

    "RF_INTERNATIONAL",

    "RF_HIGH_RISK_COUNTRY",

    "RF_HIGH_VELOCITY",

    "RF_RAPID_MOVEMENT",

    "RF_MULTIPLE_COUNTERPARTIES",

    "RF_CASH_ACTIVITY",

    "RF_PEP",

    "RF_SANCTIONS",

    "RF_ADVERSE_MEDIA",

    "RF_MULE_BEHAVIOR",

    "RF_LAYERING",

    "RF_FUNNEL_ACCOUNT"

]


# ======================================================================
# 17. TOTAL RED FLAG COUNT
# ======================================================================

print(
    "\nCalculating total red flags..."
)


transactions[
    "red_flag_count"
] = (

    transactions[
        RED_FLAG_COLUMNS
    ]

    .sum(
        axis=1
    )

)


# ======================================================================
# 18. AML RULE WEIGHTS
# ======================================================================

print(
    "Calculating AML rule score..."
)


RULE_WEIGHTS = {

    "RF_STRUCTURING": 15,

    "RF_HIGH_VALUE": 10,

    "RF_PROFILE_MISMATCH": 15,

    "RF_INTERNATIONAL": 5,

    "RF_HIGH_RISK_COUNTRY": 25,

    "RF_HIGH_VELOCITY": 10,

    "RF_RAPID_MOVEMENT": 15,

    "RF_MULTIPLE_COUNTERPARTIES": 10,

    "RF_CASH_ACTIVITY": 5,

    "RF_PEP": 10,

    "RF_SANCTIONS": 50,

    "RF_ADVERSE_MEDIA": 15,

    "RF_MULE_BEHAVIOR": 20,

    "RF_LAYERING": 20,

    "RF_FUNNEL_ACCOUNT": 20

}


transactions[
    "rule_risk_score"
] = 0


for rule, weight in RULE_WEIGHTS.items():

    transactions[
        "rule_risk_score"
    ] += (

        transactions[
            rule
        ]

        *

        weight

    )


# Cap score at 100

transactions[
    "rule_risk_score"
] = (

    transactions[
        "rule_risk_score"
    ]

    .clip(
        upper=100
    )

)


# ======================================================================
# 19. RISK LEVEL
# ======================================================================

print(
    "Assigning AML risk levels..."
)


def assign_risk_level(score):

    if score >= 70:

        return "CRITICAL"

    elif score >= 50:

        return "HIGH"

    elif score >= 25:

        return "MEDIUM"

    else:

        return "LOW"


transactions[
    "rule_risk_level"
] = (

    transactions[
        "rule_risk_score"
    ]

    .apply(
        assign_risk_level
    )

)


# ======================================================================
# 20. RULE ALERT
# ======================================================================

transactions[
    "rule_alert"
] = flag_column(

    transactions[
        "rule_risk_score"
    ] >= 25

)


# ======================================================================
# 21. PRIMARY RED FLAG
# ======================================================================

def get_primary_red_flag(row):

    active_rules = []


    for rule in RED_FLAG_COLUMNS:

        if row[rule] == 1:

            active_rules.append(
                rule
            )


    if len(active_rules) == 0:

        return "NONE"


    return active_rules[0]


transactions[
    "primary_red_flag"
] = transactions.apply(

    get_primary_red_flag,

    axis=1

)


# ======================================================================
# 22. RED FLAG DESCRIPTIONS
# ======================================================================

RED_FLAG_DESCRIPTIONS = {

    "RF_STRUCTURING":
        "Potential structuring/smurfing pattern",

    "RF_HIGH_VALUE":
        "High-value transaction",

    "RF_PROFILE_MISMATCH":
        "Transaction inconsistent with expected customer activity",

    "RF_INTERNATIONAL":
        "International transaction activity",

    "RF_HIGH_RISK_COUNTRY":
        "Transaction involving high-risk country",

    "RF_HIGH_VELOCITY":
        "High transaction velocity",

    "RF_RAPID_MOVEMENT":
        "Rapid movement of funds",

    "RF_MULTIPLE_COUNTERPARTIES":
        "Multiple counterparties",

    "RF_CASH_ACTIVITY":
        "Cash-intensive activity",

    "RF_PEP":
        "PEP-related activity",

    "RF_SANCTIONS":
        "Potential sanctions indicator",

    "RF_ADVERSE_MEDIA":
        "Adverse media indicator",

    "RF_MULE_BEHAVIOR":
        "Potential mule-account behavior",

    "RF_LAYERING":
        "Potential layering behavior",

    "RF_FUNNEL_ACCOUNT":
        "Potential funnel-account behavior"

}


transactions[
    "primary_red_flag_description"
] = (

    transactions[
        "primary_red_flag"
    ]

    .map(
        RED_FLAG_DESCRIPTIONS
    )

    .fillna(
        "No significant red flag"
    )

)


# ======================================================================
# 23. INVESTIGATION PRIORITY
# ======================================================================

def investigation_priority(row):

    score = row[
        "rule_risk_score"
    ]


    if score >= 70:

        return "P1 - URGENT"

    elif score >= 50:

        return "P2 - HIGH"

    elif score >= 25:

        return "P3 - STANDARD"

    else:

        return "P4 - LOW"


transactions[
    "investigation_priority"
] = transactions.apply(

    investigation_priority,

    axis=1

)


# ======================================================================
# 24. SAVE TRANSACTION RULE RESULTS
# ======================================================================

OUTPUT_FILE = (

    PROCESSED_DIR

    /

    "aml_rule_engine_transactions.csv"

)


transactions.to_csv(

    OUTPUT_FILE,

    index=False

)


print(
    "\nRule-engine transaction file saved:"
)

print(
    OUTPUT_FILE
)


# ======================================================================
# 25. RULE SUMMARY
# ======================================================================

rule_summary = []


for rule in RED_FLAG_COLUMNS:

    flagged = int(

        transactions[
            rule
        ].sum()

    )


    percentage = (

        flagged

        /

        len(transactions)

        *

        100

    )


    rule_summary.append({

        "rule": rule,

        "flagged_transactions":
            flagged,

        "percentage_of_transactions":
            round(
                percentage,
                2
            )

    })


rule_summary_df = (

    pd.DataFrame(
        rule_summary
    )

    .sort_values(

        "flagged_transactions",

        ascending=False

    )

)


RULE_SUMMARY_FILE = (

    REPORT_DIR

    /

    "rule_summary.csv"

)


rule_summary_df.to_csv(

    RULE_SUMMARY_FILE,

    index=False

)


# ======================================================================
# 26. RISK LEVEL SUMMARY
# ======================================================================

risk_summary = (

    transactions[
        "rule_risk_level"
    ]

    .value_counts()

    .rename_axis(
        "risk_level"
    )

    .reset_index(
        name="transaction_count"
    )

)


RISK_SUMMARY_FILE = (

    REPORT_DIR

    /

    "risk_level_summary.csv"

)


risk_summary.to_csv(

    RISK_SUMMARY_FILE,

    index=False

)


# ======================================================================
# 27. CUSTOMER RISK SUMMARY
# ======================================================================

customer_risk = (

    transactions

    .groupby(
        "customer_id"
    )

    .agg(

        total_transactions=(
            "transaction_id",
            "count"
        ),

        total_transaction_value=(
            "amount_usd",
            "sum"
        ),

        average_transaction_value=(
            "amount_usd",
            "mean"
        ),

        maximum_transaction_value=(
            "amount_usd",
            "max"
        ),

        total_red_flags=(
            "red_flag_count",
            "sum"
        ),

        maximum_rule_score=(
            "rule_risk_score",
            "max"
        ),

        alert_count=(
            "rule_alert",
            "sum"
        )

    )

    .reset_index()

)


customer_risk[
    "customer_risk_level"
] = (

    customer_risk[
        "maximum_rule_score"
    ]

    .apply(
        assign_risk_level
    )

)


CUSTOMER_RISK_FILE = (

    REPORT_DIR

    /

    "customer_risk_summary.csv"

)


customer_risk.to_csv(

    CUSTOMER_RISK_FILE,

    index=False

)


# ======================================================================
# 28. TOP 100 RISK CUSTOMERS
# ======================================================================

top_risk_customers = (

    customer_risk

    .sort_values(

        [

            "maximum_rule_score",

            "total_red_flags"

        ],

        ascending=False

    )

    .head(100)

)


TOP_CUSTOMERS_FILE = (

    REPORT_DIR

    /

    "top_100_risk_customers.csv"

)


top_risk_customers.to_csv(

    TOP_CUSTOMERS_FILE,

    index=False

)


# ======================================================================
# 29. CONSOLE SUMMARY
# ======================================================================

print("\n")

print("=" * 70)

print(
    "AML RULE ENGINE RESULTS"
)

print("=" * 70)


# Calculate values separately.
# This prevents f-string syntax problems.

total_transactions = len(
    transactions
)


alert_count = int(

    transactions[
        "rule_alert"
    ].sum()

)


alert_rate = (

    alert_count

    /

    total_transactions

    *

    100

)


critical_count = int(

    (
        transactions[
            "rule_risk_level"
        ]
        == "CRITICAL"
    ).sum()

)


high_count = int(

    (
        transactions[
            "rule_risk_level"
        ]
        == "HIGH"
    ).sum()

)


medium_count = int(

    (
        transactions[
            "rule_risk_level"
        ]
        == "MEDIUM"
    ).sum()

)


low_count = int(

    (
        transactions[
            "rule_risk_level"
        ]
        == "LOW"
    ).sum()

)


print(
    "\nTotal transactions:",
    f"{total_transactions:,}"
)


print(
    "Transactions with alerts:",
    f"{alert_count:,}"
)


print(
    "Alert rate:",
    f"{alert_rate:.2f}%"
)


print(
    "Critical transactions:",
    f"{critical_count:,}"
)


print(
    "High-risk transactions:",
    f"{high_count:,}"
)


print(
    "Medium-risk transactions:",
    f"{medium_count:,}"
)


print(
    "Low-risk transactions:",
    f"{low_count:,}"
)


# ======================================================================
# 30. RULE SUMMARY OUTPUT
# ======================================================================

print("\n")

print(
    "Top AML rules by number of alerts:"
)

print("-" * 70)


print(

    rule_summary_df.to_string(
        index=False
    )

)


# ======================================================================
# 31. GENERATED FILES
# ======================================================================

print("\n")

print("=" * 70)

print(
    "GENERATED FILES"
)

print("=" * 70)


print(
    "\n1.",
    OUTPUT_FILE
)


print(
    "2.",
    RULE_SUMMARY_FILE
)


print(
    "3.",
    RISK_SUMMARY_FILE
)


print(
    "4.",
    CUSTOMER_RISK_FILE
)


print(
    "5.",
    TOP_CUSTOMERS_FILE
)


# ======================================================================
# 32. COMPLETION MESSAGE
# ======================================================================

print("\n")

print("=" * 70)

print(
    "PHASE 3 COMPLETED SUCCESSFULLY"
)

print("=" * 70)


print(
    "\nPhase 4:"
)

print(
    "AML FEATURE ENGINEERING FOR MACHINE LEARNING"
)

print(
    "\nThe rule-engine output is now ready for ML feature engineering."
)