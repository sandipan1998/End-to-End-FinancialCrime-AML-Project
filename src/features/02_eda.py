# ============================================================
# US AML FINANCIAL CRIME PROJECT
# PHASE 2 - AML TRANSACTION MONITORING EDA
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 0. CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("US AML FINANCIAL CRIME PROJECT")
print("PHASE 2 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# Project directories
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORT_DIR = BASE_DIR / "reports"
EDA_DIR = REPORT_DIR / "eda"


PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EDA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Input file
# ------------------------------------------------------------

INPUT_FILE = (
    RAW_DIR
    /
    "US_AML_Synthetic_Dataset.xlsx"
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading AML dataset...")


customers = pd.read_excel(

    INPUT_FILE,

    sheet_name="Customers"

)


accounts = pd.read_excel(

    INPUT_FILE,

    sheet_name="Accounts"

)


transactions = pd.read_excel(

    INPUT_FILE,

    sheet_name="Transactions"

)


alerts = pd.read_excel(

    INPUT_FILE,

    sheet_name="Alerts"

)


# Convert timestamp

transactions["timestamp"] = pd.to_datetime(

    transactions["timestamp"]

)


customers["customer_since"] = pd.to_datetime(

    customers["customer_since"]

)


print("\nDataset loaded successfully.")


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)


print(
    f"\nCustomers      : {len(customers):,}"
)

print(
    f"Accounts       : {len(accounts):,}"
)

print(
    f"Transactions   : {len(transactions):,}"
)

print(
    f"Alerts         : {len(alerts):,}"
)


# ============================================================
# 3. DATASET SHAPES
# ============================================================

print("\n" + "=" * 70)
print("2. DATASET SHAPES")
print("=" * 70)


print(
    "\nCustomers shape:",
    customers.shape
)

print(
    "Accounts shape:",
    accounts.shape
)

print(
    "Transactions shape:",
    transactions.shape
)

print(
    "Alerts shape:",
    alerts.shape
)


# ============================================================
# 4. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("3. TRANSACTION DATA TYPES")
print("=" * 70)


print(
    transactions.dtypes.to_string()
)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("4. MISSING VALUE ANALYSIS")
print("=" * 70)


missing_transactions = (

    transactions.isnull()
    .sum()
    .sort_values(
        ascending=False
    )

)


missing_transactions = (

    missing_transactions[
        missing_transactions > 0
    ]

)


if len(missing_transactions) == 0:

    print(
        "\nNo missing values found in Transactions."
    )

else:

    print(
        missing_transactions.to_string()
    )


# ============================================================
# 6. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("5. DUPLICATE ANALYSIS")
print("=" * 70)


duplicate_transactions = (

    transactions[
        "transaction_id"
    ]
    .duplicated()
    .sum()

)


duplicate_customers = (

    customers[
        "customer_id"
    ]
    .duplicated()
    .sum()

)


duplicate_accounts = (

    accounts[
        "account_id"
    ]
    .duplicated()
    .sum()

)


print(
    f"\nDuplicate transactions : {duplicate_transactions}"
)

print(
    f"Duplicate customers    : {duplicate_customers}"
)

print(
    f"Duplicate accounts     : {duplicate_accounts}"
)


# ============================================================
# 7. CUSTOMER RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("6. CUSTOMER RISK DISTRIBUTION")
print("=" * 70)


customer_risk = (

    customers[
        "risk_rating"
    ]
    .value_counts()

)


print(
    customer_risk.to_string()
)


# ------------------------------------------------------------
# Save customer risk summary
# ------------------------------------------------------------

customer_risk.to_csv(

    EDA_DIR
    /
    "customer_risk_distribution.csv"

)


# ------------------------------------------------------------
# Chart
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

customer_risk.plot(
    kind="bar"
)

plt.title(
    "Customer Risk Rating Distribution"
)

plt.xlabel(
    "Risk Rating"
)

plt.ylabel(
    "Number of Customers"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(

    EDA_DIR
    /
    "01_customer_risk_distribution.png",

    dpi=150

)

plt.close()


# ============================================================
# 8. PEP / SANCTIONS / ADVERSE MEDIA
# ============================================================

print("\n" + "=" * 70)
print("7. KYC / CDD RISK INDICATORS")
print("=" * 70)


pep_count = customers["pep_flag"].sum()

sanctions_count = customers["sanctions_flag"].sum()

adverse_media_count = customers[
    "adverse_media_flag"
].sum()


print(
    f"\nPEP customers            : {pep_count:,}"
)

print(
    f"Sanctions flagged        : {sanctions_count:,}"
)

print(
    f"Adverse media flagged    : {adverse_media_count:,}"
)


kyc_summary = pd.DataFrame({

    "indicator": [

        "PEP",
        "Sanctions",
        "Adverse Media"

    ],

    "customer_count": [

        pep_count,
        sanctions_count,
        adverse_media_count

    ]

})


kyc_summary.to_csv(

    EDA_DIR
    /
    "kyc_risk_indicators.csv",

    index=False

)


# ============================================================
# 9. TRANSACTION TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("8. TRANSACTION TYPE DISTRIBUTION")
print("=" * 70)


transaction_type_counts = (

    transactions[
        "transaction_type"
    ]
    .value_counts()

)


print(
    transaction_type_counts.to_string()
)


transaction_type_counts.to_csv(

    EDA_DIR
    /
    "transaction_type_distribution.csv"

)


plt.figure(
    figsize=(10, 6)
)


transaction_type_counts.plot(
    kind="bar"
)


plt.title(
    "Transaction Type Distribution"
)

plt.xlabel(
    "Transaction Type"
)

plt.ylabel(
    "Number of Transactions"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()


plt.savefig(

    EDA_DIR
    /
    "02_transaction_type_distribution.png",

    dpi=150

)


plt.close()


# ============================================================
# 10. TRANSACTION AMOUNT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("9. TRANSACTION AMOUNT ANALYSIS")
print("=" * 70)


print(
    "\nTransaction amount statistics:"
)


print(

    transactions[
        "amount_usd"
    ]
    .describe()
    .to_string()

)


amount_summary = (

    transactions[
        "amount_usd"
    ]
    .describe()

)


amount_summary.to_csv(

    EDA_DIR
    /
    "transaction_amount_statistics.csv"

)


# ------------------------------------------------------------
# Histogram
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.hist(

    transactions[
        "amount_usd"
    ],

    bins=100

)


plt.title(
    "Transaction Amount Distribution"
)

plt.xlabel(
    "Transaction Amount (USD)"
)

plt.ylabel(
    "Frequency"
)


plt.tight_layout()


plt.savefig(

    EDA_DIR
    /
    "03_transaction_amount_distribution.png",

    dpi=150

)


plt.close()


# ============================================================
# 11. NORMAL VS SUSPICIOUS TRANSACTIONS
# ============================================================

print("\n" + "=" * 70)
print("10. NORMAL VS SUSPICIOUS TRANSACTIONS")
print("=" * 70)


suspicious_counts = (

    transactions[
        "known_suspicious"
    ]

    .map({

        0: "Normal",
        1: "Suspicious"

    })

    .value_counts()

)


print(
    suspicious_counts.to_string()
)


suspicious_counts.to_csv(

    EDA_DIR
    /
    "normal_vs_suspicious.csv"

)


plt.figure(
    figsize=(7, 5)
)


suspicious_counts.plot(
    kind="bar"
)


plt.title(
    "Normal vs Suspicious Transactions"
)

plt.xlabel(
    "Transaction Classification"
)

plt.ylabel(
    "Number of Transactions"
)

plt.xticks(
    rotation=0
)


plt.tight_layout()


plt.savefig(

    EDA_DIR
    /
    "04_normal_vs_suspicious.png",

    dpi=150

)


plt.close()


# ============================================================
# 12. AML SCENARIO DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("11. AML SCENARIO DISTRIBUTION")
print("=" * 70)


scenario_counts = (

    transactions[
        transactions[
            "known_suspicious"
        ]
        ==
        1
    ]

    [
        "scenario"
    ]

    .value_counts()

)


print(
    scenario_counts.to_string()
)


scenario_counts.to_csv(

    EDA_DIR
    /
    "aml_scenario_distribution.csv"

)


plt.figure(
    figsize=(10, 6)
)


scenario_counts.plot(
    kind="bar"
)


plt.title(
    "AML Suspicious Transaction Scenarios"
)

plt.xlabel(
    "AML Scenario"
)

plt.ylabel(
    "Number of Transactions"
)

plt.xticks(
    rotation=45,
    ha="right"
)


plt.tight_layout()


plt.savefig(

    EDA_DIR
    /
    "05_aml_scenario_distribution.png",

    dpi=150

)


plt.close()


# ============================================================
# 13. RED FLAG ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("12. AML RED FLAG ANALYSIS")
print("=" * 70)


red_flag_columns = [

    "RF_STRUCTURING",
    "RF_HIGH_AMOUNT",
    "RF_INTERNATIONAL",
    "RF_PROFILE_MISMATCH"

]


red_flag_summary = {}


for column in red_flag_columns:

    red_flag_summary[column] = (

        transactions[
            column
        ]
        .sum()

    )


red_flag_df = pd.DataFrame({

    "red_flag":
        list(
            red_flag_summary.keys()
        ),

    "transaction_count":
        list(
            red_flag_summary.values()
        )

})


print(
    red_flag_df.to_string(
        index=False
    )
)


red_flag_df.to_csv(

    EDA_DIR
    /
    "red_flag_summary.csv",

    index=False

)


plt.figure(
    figsize=(10, 6)
)


plt.bar(

    red_flag_df[
        "red_flag"
    ],

    red_flag_df[
        "transaction_count"
    ]

)


plt.title(
    "AML Red Flag Frequency"
)

plt.xlabel(
    "Red Flag"
)

plt.ylabel(
    "Number of Transactions"
)

plt.xticks(
    rotation=45,
    ha="right"
)


plt.tight_layout()


plt.savefig(

    EDA_DIR
    /
    "06_red_flag_frequency.png",

    dpi=150

)


plt.close()


# ============================================================
# 14. RED FLAG COUNT
# ============================================================

print("\n" + "=" * 70)
print("13. RED FLAG COUNT DISTRIBUTION")
print("=" * 70)


red_flag_count_distribution = (

    transactions[
        "red_flag_count"
    ]

    .value_counts()

    .sort_index()

)


print(
    red_flag_count_distribution.to_string()
)


red_flag_count_distribution.to_csv(

    EDA_DIR
    /
    "red_flag_count_distribution.csv"

)


plt.figure(
    figsize=(8, 5)
)


red_flag_count_distribution.plot(
    kind="bar"
)


plt.title(
    "Number of Red Flags per Transaction"
)

plt.xlabel(
    "Red Flag Count"
)

plt.ylabel(
    "Transactions"
)

plt.xticks(
    rotation=0
)


plt.tight_layout()


plt.savefig(

    EDA_DIR
    /
    "07_red_flag_count_distribution.png",

    dpi=150

)


plt.close()


# ============================================================
# 15. RULE ALERT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("14. RULE-BASED ALERT ANALYSIS")
print("=" * 70)


alert_counts = (

    transactions[
        "rule_alert"
    ]

    .map({

        0: "No Alert",
        1: "Alert"

    })

    .value_counts()

)


print(
    alert_counts.to_string()
)


alert_counts.to_csv(

    EDA_DIR
    /
    "rule_alert_distribution.csv"

)


# ============================================================
# 16. RISK LEVEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("15. RULE RISK LEVEL DISTRIBUTION")
print("=" * 70)


risk_level_counts = (

    transactions[
        "risk_level"
    ]

    .value_counts()

)


print(
    risk_level_counts.to_string()
)


risk_level_counts.to_csv(

    EDA_DIR
    /
    "risk_level_distribution.csv"

)


plt.figure(
    figsize=(8, 5)
)


risk_level_counts.plot(
    kind="bar"
)


plt.title(
    "Transaction Rule-Based Risk Level"
)

plt.xlabel(
    "Risk Level"
)

plt.ylabel(
    "Transactions"
)

plt.xticks(
    rotation=0
)


plt.tight_layout()


plt.savefig(

    EDA_DIR
    /
    "08_risk_level_distribution.png",

    dpi=150

)


plt.close()


# ============================================================
# 17. INTERNATIONAL TRANSACTION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("16. INTERNATIONAL TRANSACTION ANALYSIS")
print("=" * 70)


international_counts = (

    transactions[
        "is_international"
    ]

    .map({

        0: "Domestic",
        1: "International"

    })

    .value_counts()

)


print(
    international_counts.to_string()
)


international_counts.to_csv(

    EDA_DIR
    /
    "international_transaction_distribution.csv"

)


# ============================================================
# 18. CASH TRANSACTION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("17. CASH TRANSACTION ANALYSIS")
print("=" * 70)


cash_counts = (

    transactions[
        "is_cash"
    ]

    .map({

        0: "Non-Cash",
        1: "Cash"

    })

    .value_counts()

)


print(
    cash_counts.to_string()
)


cash_counts.to_csv(

    EDA_DIR
    /
    "cash_transaction_distribution.csv"

)


# ============================================================
# 19. SUSPICIOUS VS NORMAL AMOUNT
# ============================================================

print("\n" + "=" * 70)
print("18. SUSPICIOUS VS NORMAL TRANSACTION AMOUNT")
print("=" * 70)


amount_comparison = (

    transactions

    .groupby(
        "known_suspicious"
    )

    ["amount_usd"]

    .agg(

        [
            "count",
            "mean",
            "median",
            "min",
            "max"

        ]

    )

)


print(
    amount_comparison.to_string()
)


amount_comparison.to_csv(

    EDA_DIR
    /
    "suspicious_vs_normal_amount.csv"

)


# ============================================================
# 20. TOP CUSTOMERS BY TRANSACTION VOLUME
# ============================================================

print("\n" + "=" * 70)
print("19. TOP CUSTOMERS BY TRANSACTION VOLUME")
print("=" * 70)


top_customers_volume = (

    transactions

    .groupby(
        "customer_id"
    )

    .agg(

        transaction_count=(

            "transaction_id",
            "count"

        ),

        total_amount=(

            "amount_usd",
            "sum"

        ),

        average_amount=(

            "amount_usd",
            "mean"

        ),

        suspicious_transactions=(

            "known_suspicious",
            "sum"

        )

    )

    .sort_values(

        "total_amount",

        ascending=False

    )

    .head(20)

)


print(
    top_customers_volume.to_string()
)


top_customers_volume.to_csv(

    EDA_DIR
    /
    "top_20_customers_by_volume.csv"

)


# ============================================================
# 21. TOP CUSTOMERS BY SUSPICIOUS ACTIVITY
# ============================================================

print("\n" + "=" * 70)
print("20. TOP CUSTOMERS BY SUSPICIOUS ACTIVITY")
print("=" * 70)


top_suspicious_customers = (

    transactions

    .groupby(
        "customer_id"
    )

    .agg(

        suspicious_transactions=(

            "known_suspicious",
            "sum"

        ),

        total_transactions=(

            "transaction_id",
            "count"

        ),

        total_amount=(

            "amount_usd",
            "sum"

        )

    )

    .sort_values(

        [

            "suspicious_transactions",
            "total_amount"

        ],

        ascending=False

    )

    .head(20)

)


print(
    top_suspicious_customers.to_string()
)


top_suspicious_customers.to_csv(

    EDA_DIR
    /
    "top_suspicious_customers.csv"

)


# ============================================================
# 22. COUNTRY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("21. ORIGIN COUNTRY ANALYSIS")
print("=" * 70)


origin_country_counts = (

    transactions[
        "origin_country"
    ]

    .value_counts()

)


print(
    origin_country_counts.to_string()
)


origin_country_counts.to_csv(

    EDA_DIR
    /
    "origin_country_distribution.csv"

)


print("\nDestination country analysis:")


destination_country_counts = (

    transactions[
        "destination_country"
    ]

    .value_counts()

)


print(
    destination_country_counts.to_string()
)


destination_country_counts.to_csv(

    EDA_DIR
    /
    "destination_country_distribution.csv"

)


# ============================================================
# 23. AML SCENARIO VS AVERAGE TRANSACTION AMOUNT
# ============================================================

print("\n" + "=" * 70)
print("22. AML SCENARIO TRANSACTION AMOUNT")
print("=" * 70)


scenario_amount_analysis = (

    transactions

    .groupby(
        "scenario"
    )

    ["amount_usd"]

    .agg(

        [

            "count",
            "mean",
            "median",
            "min",
            "max"

        ]

    )

    .sort_values(

        "mean",

        ascending=False

    )

)


print(
    scenario_amount_analysis.to_string()
)


scenario_amount_analysis.to_csv(

    EDA_DIR
    /
    "scenario_amount_analysis.csv"

)


# ============================================================
# 24. AML SCENARIO RED FLAGS
# ============================================================

print("\n" + "=" * 70)
print("23. AML SCENARIO VS RED FLAGS")
print("=" * 70)


scenario_red_flags = (

    transactions

    .groupby(
        "scenario"
    )

    [

        red_flag_columns

        +
        [
            "red_flag_count"
        ]

    ]

    .mean()

)


print(
    scenario_red_flags.to_string()
)


scenario_red_flags.to_csv(

    EDA_DIR
    /
    "scenario_red_flag_analysis.csv"

)


# ============================================================
# 25. ALERT PRECISION BASELINE
# ============================================================

print("\n" + "=" * 70)
print("24. RULE ENGINE BASELINE")
print("=" * 70)


true_suspicious = (

    transactions[
        "known_suspicious"
    ]
    ==
    1

)


predicted_alert = (

    transactions[
        "rule_alert"
    ]
    ==
    1

)


true_positive = (

    true_suspicious
    &
    predicted_alert

).sum()


false_positive = (

    (~true_suspicious)
    &
    predicted_alert

).sum()


false_negative = (

    true_suspicious
    &
    (~predicted_alert)

).sum()


true_negative = (

    (~true_suspicious)
    &
    (~predicted_alert)

).sum()


print(
    f"\nTrue Positive  : {true_positive:,}"
)

print(
    f"False Positive : {false_positive:,}"
)

print(
    f"False Negative : {false_negative:,}"
)

print(
    f"True Negative  : {true_negative:,}"
)


precision = (

    true_positive
    /
    (
        true_positive
        +
        false_positive
    )

    if
    (
        true_positive
        +
        false_positive
    )
    >
    0

    else 0

)


recall = (

    true_positive
    /
    (
        true_positive
        +
        false_negative
    )

    if
    (
        true_positive
        +
        false_negative
    )
    >
    0

    else 0

)


f1_score = (

    2
    *
    precision
    *
    recall

    /

    (
        precision
        +
        recall
    )

    if
    (
        precision
        +
        recall
    )
    >
    0

    else 0

)


print(
    f"\nPrecision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1_score:.4f}"
)


baseline_metrics = pd.DataFrame({

    "metric": [

        "True Positive",
        "False Positive",
        "False Negative",
        "True Negative",
        "Precision",
        "Recall",
        "F1 Score"

    ],

    "value": [

        true_positive,
        false_positive,
        false_negative,
        true_negative,
        precision,
        recall,
        f1_score

    ]

})


baseline_metrics.to_csv(

    EDA_DIR
    /
    "rule_engine_baseline.csv",

    index=False

)


# ============================================================
# 26. CREATE MASTER EDA SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("25. CREATING MASTER EDA SUMMARY")
print("=" * 70)


suspicious_transaction_count = (

    transactions[
        "known_suspicious"
    ]
    .sum()

)


suspicious_percentage = (

    suspicious_transaction_count
    /
    len(transactions)
    *
    100

)


alert_count = (

    transactions[
        "rule_alert"
    ]
    .sum()

)


alert_percentage = (

    alert_count
    /
    len(transactions)
    *
    100

)


master_summary = pd.DataFrame({

    "metric": [

        "Total Customers",
        "Total Accounts",
        "Total Transactions",
        "Total Alerts",
        "Suspicious Transactions",
        "Suspicious Transaction %",
        "Alert %",
        "PEP Customers",
        "Sanctions Flagged Customers",
        "Adverse Media Customers",
        "International Transactions",
        "Cash Transactions",
        "True Positive",
        "False Positive",
        "False Negative",
        "Precision",
        "Recall",
        "F1 Score"

    ],

    "value": [

        len(customers),
        len(accounts),
        len(transactions),
        alert_count,
        suspicious_transaction_count,
        suspicious_percentage,
        alert_percentage,
        pep_count,
        sanctions_count,
        adverse_media_count,
        transactions["is_international"].sum(),
        transactions["is_cash"].sum(),
        true_positive,
        false_positive,
        false_negative,
        precision,
        recall,
        f1_score

    ]

})


print(
    master_summary.to_string(
        index=False
    )
)


master_summary.to_csv(

    EDA_DIR
    /
    "MASTER_EDA_SUMMARY.csv",

    index=False

)


# ============================================================
# 27. SAVE PROCESSED TRANSACTION DATA
# ============================================================

print("\nSaving processed transaction data...")


transactions.to_csv(

    PROCESSED_DIR
    /
    "transactions_eda_ready.csv",

    index=False

)


customers.to_csv(

    PROCESSED_DIR
    /
    "customers_eda_ready.csv",

    index=False

)


alerts.to_csv(

    PROCESSED_DIR
    /
    "alerts_eda_ready.csv",

    index=False

)


# ============================================================
# 28. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print(
    "PHASE 2 EDA COMPLETED SUCCESSFULLY"
)

print("=" * 70)


print(
    "\nEDA reports saved to:"
)

print(
    EDA_DIR
)


print(
    "\nProcessed datasets saved to:"
)

print(
    PROCESSED_DIR
)


print(
    "\nNext step:"
)

print(
    "PHASE 3 - AML RULE ENGINE & RED FLAG DETECTION"
)

print("=" * 70)