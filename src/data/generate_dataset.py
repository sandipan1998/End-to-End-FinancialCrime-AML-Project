# ============================================================
# US AML FINANCIAL CRIME PROJECT
# PHASE 1 - SYNTHETIC DATASET GENERATOR
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 42

np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "US_AML_Synthetic_Dataset.xlsx"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_customer_ids(n):
    return [
        f"CUST{i:06d}"
        for i in range(1, n + 1)
    ]


def create_account_ids(n):
    return [
        f"ACC{i:07d}"
        for i in range(1, n + 1)
    ]


def normalize_probabilities(probabilities):
    """
    Automatically normalize probabilities so their sum = 1.
    This prevents NumPy probability-sum errors.
    """
    probabilities = np.array(
        probabilities,
        dtype=float
    )

    total = probabilities.sum()

    if total <= 0:
        raise ValueError(
            "Probability values must have a positive sum."
        )

    return probabilities / total


# ============================================================
# 1. CUSTOMER DATA
# ============================================================

print("\nCreating customer dataset...")

N_CUSTOMERS = 5000

customer_ids = create_customer_ids(
    N_CUSTOMERS
)


# ------------------------------------------------------------
# US States
# ------------------------------------------------------------

states = [
    "NY",
    "CA",
    "TX",
    "FL",
    "IL",
    "NJ",
    "GA",
    "PA",
    "WA",
    "MA",
    "OH",
    "VA",
    "NC",
    "MD",
    "CO"
]


# ------------------------------------------------------------
# Occupations
# ------------------------------------------------------------

occupations = [
    "Software Engineer",
    "Teacher",
    "Doctor",
    "Restaurant Owner",
    "Consultant",
    "Retail Worker",
    "Accountant",
    "Construction Worker",
    "Real Estate Agent",
    "Student",
    "Small Business Owner",
    "Sales Manager",
    "Lawyer",
    "Trader"
]


# ------------------------------------------------------------
# Customer Type
# ------------------------------------------------------------

customer_types = np.random.choice(

    [
        "Individual",
        "Business"
    ],

    N_CUSTOMERS,

    p=normalize_probabilities(
        [
            0.82,
            0.18
        ]
    )
)


# ------------------------------------------------------------
# Create Customer DataFrame
# ------------------------------------------------------------

customers = pd.DataFrame({

    "customer_id":
        customer_ids,

    "customer_type":
        customer_types,

    "state":
        np.random.choice(
            states,
            N_CUSTOMERS
        ),

    "occupation":
        np.random.choice(
            occupations,
            N_CUSTOMERS
        ),

    "annual_income_usd":
        np.round(

            np.exp(

                np.random.normal(

                    np.log(75000),

                    0.65,

                    N_CUSTOMERS

                )

            ),

            2

        ),

    "customer_since":
        pd.to_datetime(

            np.random.choice(

                pd.date_range(
                    "2017-01-01",
                    "2025-12-31"
                ),

                N_CUSTOMERS

            )

        ),

    "pep_flag":
        np.random.choice(

            [
                0,
                1
            ],

            N_CUSTOMERS,

            p=normalize_probabilities(
                [
                    0.97,
                    0.03
                ]
            )

        ),

    "sanctions_flag":
        np.random.choice(

            [
                0,
                1
            ],

            N_CUSTOMERS,

            p=normalize_probabilities(
                [
                    0.995,
                    0.005
                ]
            )

        ),

    "adverse_media_flag":
        np.random.choice(

            [
                0,
                1
            ],

            N_CUSTOMERS,

            p=normalize_probabilities(
                [
                    0.96,
                    0.04
                ]
            )

        )

})


# ============================================================
# EXPECTED MONTHLY TRANSACTION VOLUME
# ============================================================

customers["expected_monthly_volume_usd"] = np.round(

    (

        customers["annual_income_usd"]
        /
        12

    )

    *

    np.random.uniform(

        0.3,
        1.8,
        N_CUSTOMERS

    ),

    2

)


# ============================================================
# INITIAL CUSTOMER RISK
# ============================================================

customers["risk_rating"] = np.select(

    [

        (

            customers["pep_flag"]
            ==
            1

        )

        |

        (

            customers["sanctions_flag"]
            ==
            1

        ),

        customers["adverse_media_flag"]
        ==
        1

    ],

    [

        "HIGH",

        "MEDIUM"

    ],

    default="LOW"

)


# ============================================================
# 2. ACCOUNT DATA
# ============================================================

print("Creating account dataset...")


accounts = pd.DataFrame({

    "account_id":
        create_account_ids(
            N_CUSTOMERS
        ),

    "customer_id":
        customer_ids,

    "account_type":
        np.random.choice(

            [

                "Checking",
                "Savings",
                "Business Checking"

            ],

            N_CUSTOMERS,

            p=normalize_probabilities(

                [
                    0.55,
                    0.25,
                    0.20
                ]

            )

        ),

    "currency":
        "USD",

    "opening_balance_usd":
        np.round(

            np.random.uniform(

                500,
                25000,
                N_CUSTOMERS

            ),

            2

        )

})


# ============================================================
# 3. NORMAL TRANSACTIONS
# ============================================================

print("Creating normal transactions...")


N_TRANSACTIONS = 120000


transaction_ids = [

    f"TXN{i:08d}"

    for i in range(

        1,
        N_TRANSACTIONS + 1

    )

]


# ------------------------------------------------------------
# Transaction Dates
# ------------------------------------------------------------

start_date = datetime(
    2025,
    1,
    1
)


random_minutes = np.random.randint(

    0,

    365 * 24 * 60,

    N_TRANSACTIONS

)


timestamps = [

    start_date
    +
    timedelta(
        minutes=int(x)
    )

    for x in random_minutes

]


# ------------------------------------------------------------
# Sender / Receiver
# ------------------------------------------------------------

sender_index = np.random.randint(

    0,

    N_CUSTOMERS,

    N_TRANSACTIONS

)


receiver_index = np.random.randint(

    0,

    N_CUSTOMERS,

    N_TRANSACTIONS

)


# Prevent most self-transfers

same_account = (

    sender_index
    ==
    receiver_index

)


receiver_index[same_account] = (

    receiver_index[same_account]
    +
    1

) % N_CUSTOMERS


# ============================================================
# TRANSACTION TYPES
# ============================================================

transaction_types = np.random.choice(

    [

        "ACH",
        "WIRE",
        "CARD_PAYMENT",
        "CASH_DEPOSIT",
        "CASH_WITHDRAWAL",
        "P2P_TRANSFER",
        "INTERNATIONAL_WIRE",
        "ATM"

    ],

    N_TRANSACTIONS,

    p=normalize_probabilities(

        [

            0.20,
            0.10,
            0.30,
            0.08,
            0.05,
            0.15,
            0.07,
            0.05

        ]

    )

)


# ============================================================
# TRANSACTION AMOUNTS
# ============================================================

amounts = np.round(

    np.exp(

        np.random.normal(

            np.log(350),

            1.25,

            N_TRANSACTIONS

        )

    ),

    2

)


amounts = np.clip(

    amounts,

    5,

    50000

)


# ============================================================
# COUNTRIES
# ============================================================

countries = [

    "United States",
    "Canada",
    "Mexico",
    "United Kingdom",
    "Germany",
    "Singapore",
    "UAE",
    "Turkey",
    "Hong Kong",
    "India",
    "Brazil",
    "Nigeria"

]


# ------------------------------------------------------------
# Country probabilities
#
# IMPORTANT:
# These values are automatically normalized.
# ------------------------------------------------------------

country_probabilities = normalize_probabilities(

    [

        0.72,
        0.05,
        0.04,
        0.03,
        0.03,
        0.02,
        0.02,
        0.02,
        0.02,
        0.03,
        0.02,
        0.02

    ]

)


print(
    "Country probability total:",
    country_probabilities.sum()
)


# ============================================================
# ORIGIN COUNTRIES
# ============================================================

origin_countries = np.random.choice(

    countries,

    N_TRANSACTIONS,

    p=country_probabilities

)


# ============================================================
# DESTINATION COUNTRIES
# ============================================================

destination_countries = np.random.choice(

    countries,

    N_TRANSACTIONS,

    p=country_probabilities

)


# ============================================================
# CHANNELS
# ============================================================

channels = np.random.choice(

    [

        "Online Banking",
        "Mobile App",
        "Branch",
        "ATM",
        "Phone"

    ],

    N_TRANSACTIONS,

    p=normalize_probabilities(

        [

            0.40,
            0.35,
            0.10,
            0.10,
            0.05

        ]

    )

)


# ============================================================
# CREATE TRANSACTION DATAFRAME
# ============================================================

transactions = pd.DataFrame({

    "transaction_id":
        transaction_ids,

    "timestamp":
        timestamps,

    "sender_account":

        [

            f"ACC{x + 1:07d}"

            for x in sender_index

        ],

    "receiver_account":

        [

            f"ACC{x + 1:07d}"

            for x in receiver_index

        ],

    "transaction_type":
        transaction_types,

    "amount_usd":
        amounts,

    "channel":
        channels,

    "origin_country":
        origin_countries,

    "destination_country":
        destination_countries

})


# ============================================================
# 4. DERIVED TRANSACTION FIELDS
# ============================================================

transactions["is_international"] = (

    transactions["origin_country"]
    !=
    transactions["destination_country"]

).astype(int)


transactions["is_cash"] = (

    transactions["transaction_type"]
    .isin(

        [

            "CASH_DEPOSIT",
            "CASH_WITHDRAWAL",
            "ATM"

        ]

    )

).astype(int)


# ============================================================
# 5. LINK TRANSACTIONS TO CUSTOMERS
# ============================================================

transactions["customer_id"] = (

    transactions["sender_account"]

    .str.extract(
        r"ACC0*(\d+)"
    )[0]

    .astype(int)

    .astype(str)

    .str.zfill(6)

)


transactions["customer_id"] = (

    "CUST"
    +
    transactions["customer_id"]

)


# ============================================================
# 6. INJECT AML SCENARIOS
# ============================================================

print("Injecting AML scenarios...")


scenario_rows = []


def add_scenario(

    customer_number,

    scenario,

    n_transactions=20

):

    customer_id = (

        f"CUST{customer_number:06d}"

    )


    account_id = (

        f"ACC{customer_number:07d}"

    )


    base_time = (

        datetime(
            2025,
            6,
            1
        )

        +

        timedelta(

            days=int(

                np.random.randint(

                    0,
                    150

                )

            )

        )

    )


    for j in range(

        n_transactions

    ):

        receiver_number = (

            (

                customer_number
                +
                j * 37

            )

            %

            N_CUSTOMERS

        ) + 1


        receiver_account = (

            f"ACC{receiver_number:07d}"

        )


        # ====================================================
        # STRUCTURING
        # ====================================================

        if scenario == "STRUCTURING":

            amount = np.random.uniform(

                8500,
                9950

            )

            transaction_type = (

                "CASH_DEPOSIT"

            )

            origin_country = (

                "United States"

            )

            destination_country = (

                "United States"

            )


        # ====================================================
        # RAPID MOVEMENT
        # ====================================================

        elif scenario == "RAPID_MOVEMENT":

            amount = np.random.uniform(

                15000,
                45000

            )

            transaction_type = np.random.choice(

                [

                    "WIRE",
                    "ACH"

                ]

            )

            origin_country = (

                "United States"

            )

            destination_country = (

                "United States"

            )


        # ====================================================
        # FUNNEL ACCOUNT
        # ====================================================

        elif scenario == "FUNNEL_ACCOUNT":

            amount = np.random.uniform(

                4000,
                12000

            )

            transaction_type = (

                "P2P_TRANSFER"

            )

            origin_country = (

                "United States"

            )

            destination_country = (

                "United States"

            )


        # ====================================================
        # LAYERING
        # ====================================================

        elif scenario == "LAYERING":

            amount = np.random.uniform(

                10000,
                30000

            )

            transaction_type = (

                "INTERNATIONAL_WIRE"

            )

            origin_country = (

                "United States"

            )

            destination_country = np.random.choice(

                [

                    "UAE",
                    "Hong Kong",
                    "Turkey",
                    "Singapore"

                ]

            )


        # ====================================================
        # MULE ACCOUNT
        # ====================================================

        elif scenario == "MULE":

            amount = np.random.uniform(

                2000,
                10000

            )

            transaction_type = (

                "P2P_TRANSFER"

            )

            origin_country = (

                "United States"

            )

            destination_country = (

                "United States"

            )


        # ====================================================
        # FRAUD-LINKED
        # ====================================================

        else:

            amount = np.random.uniform(

                1000,
                15000

            )

            transaction_type = np.random.choice(

                [

                    "P2P_TRANSFER",
                    "WIRE",
                    "ACH"

                ]

            )

            origin_country = (

                "United States"

            )

            destination_country = (

                "United States"

            )


        scenario_rows.append({

            "transaction_id":

                f"SCN{customer_number:04d}{j:04d}",

            "timestamp":

                base_time

                +

                timedelta(

                    hours=j

                ),

            "sender_account":

                account_id,

            "receiver_account":

                receiver_account,

            "transaction_type":

                transaction_type,

            "amount_usd":

                round(

                    amount,

                    2

                ),

            "channel":

                "Online Banking",

            "origin_country":

                origin_country,

            "destination_country":

                destination_country,

            "is_international":

                int(

                    origin_country
                    !=
                    destination_country

                ),

            "is_cash":

                int(

                    transaction_type

                    in

                    [

                        "CASH_DEPOSIT",
                        "CASH_WITHDRAWAL",
                        "ATM"

                    ]

                ),

            "customer_id":

                customer_id,

            "scenario":

                scenario,

            "known_suspicious":

                1

        })


# ============================================================
# AML SCENARIOS
# ============================================================

scenario_names = [

    "STRUCTURING",
    "RAPID_MOVEMENT",
    "FUNNEL_ACCOUNT",
    "LAYERING",
    "MULE",
    "FRAUD_LINKED"

]


# ------------------------------------------------------------
# 25 customers per scenario
# ------------------------------------------------------------

for scenario_index, scenario in enumerate(

    scenario_names

):

    start_customer = (

        10
        +
        scenario_index * 25

    )


    end_customer = (

        start_customer
        +
        25

    )


    for customer_number in range(

        start_customer,
        end_customer

    ):

        add_scenario(

            customer_number,

            scenario,

            20

        )


# ============================================================
# 7. ADD SCENARIO DATA
# ============================================================

transactions["scenario"] = (

    "NORMAL"

)


transactions["known_suspicious"] = (

    0

)


scenario_df = pd.DataFrame(

    scenario_rows

)


transactions = pd.concat(

    [

        transactions,
        scenario_df

    ],

    ignore_index=True

)


# ============================================================
# 8. CUSTOMER RISK INFORMATION
# ============================================================

transactions = transactions.merge(

    customers[

        [

            "customer_id",
            "annual_income_usd",
            "occupation",
            "expected_monthly_volume_usd",
            "risk_rating",
            "pep_flag",
            "sanctions_flag",
            "adverse_media_flag"

        ]

    ],

    on="customer_id",

    how="left"

)


# ============================================================
# 9. BASIC AML FEATURES
# ============================================================

transactions["amount_vs_expected"] = (

    transactions["amount_usd"]

    /

    (

        transactions[
            "expected_monthly_volume_usd"
        ]

        /

        30

    ).clip(

        lower=1

    )

).round(3)


# ============================================================
# RED FLAG 1 - STRUCTURING
# ============================================================

transactions["RF_STRUCTURING"] = (

    (

        transactions["is_cash"]
        ==
        1

    )

    &

    (

        transactions["amount_usd"]

        .between(

            8500,
            9999.99

        )

    )

).astype(int)


# ============================================================
# RED FLAG 2 - HIGH AMOUNT
# ============================================================

transactions["RF_HIGH_AMOUNT"] = (

    transactions["amount_usd"]
    >
    10000

).astype(int)


# ============================================================
# RED FLAG 3 - INTERNATIONAL ACTIVITY
# ============================================================

transactions["RF_INTERNATIONAL"] = (

    transactions["is_international"]
    ==
    1

).astype(int)


# ============================================================
# RED FLAG 4 - CUSTOMER PROFILE MISMATCH
# ============================================================

transactions["RF_PROFILE_MISMATCH"] = (

    transactions["amount_usd"]

    >

    transactions[
        "expected_monthly_volume_usd"
    ]

    *

    0.25

).astype(int)


# ============================================================
# NUMBER OF RED FLAGS
# ============================================================

red_flag_columns = [

    "RF_STRUCTURING",
    "RF_HIGH_AMOUNT",
    "RF_INTERNATIONAL",
    "RF_PROFILE_MISMATCH"

]


transactions["red_flag_count"] = (

    transactions[
        red_flag_columns
    ]

    .sum(
        axis=1
    )

)


# ============================================================
# 10. RULE-BASED RISK SCORE
# ============================================================

transactions["rule_alert"] = (

    (

        transactions["red_flag_count"]
        >=
        2

    )

    |

    (

        transactions["known_suspicious"]
        ==
        1

    )

).astype(int)


transactions["rule_risk_score"] = np.clip(

    (

        transactions["red_flag_count"]
        *
        18

        +

        transactions["known_suspicious"]
        *
        45

        +

        transactions["pep_flag"]
        *
        10

        +

        transactions["adverse_media_flag"]
        *
        10

        +

        transactions["sanctions_flag"]
        *
        30

    ),

    0,

    100

)


transactions["risk_level"] = pd.cut(

    transactions["rule_risk_score"],

    bins=[

        -1,
        24,
        49,
        74,
        100

    ],

    labels=[

        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"

    ]

)


# ============================================================
# 11. ALERT DATASET
# ============================================================

print("Creating AML alerts...")


alerts = transactions[

    transactions["rule_alert"]
    ==
    1

].copy()


alerts["alert_id"] = [

    f"ALT{i:07d}"

    for i in range(

        1,
        len(alerts) + 1

    )

]


def create_red_flag_description(row):

    flags = []


    if row["RF_STRUCTURING"] == 1:

        flags.append(
            "Structuring"
        )


    if row["RF_HIGH_AMOUNT"] == 1:

        flags.append(
            "High Amount"
        )


    if row["RF_INTERNATIONAL"] == 1:

        flags.append(
            "International Activity"
        )


    if row["RF_PROFILE_MISMATCH"] == 1:

        flags.append(
            "Profile Mismatch"
        )


    if row["known_suspicious"] == 1:

        flags.append(
            "Scenario Indicator"
        )


    return ", ".join(flags)


alerts["red_flags"] = alerts.apply(

    create_red_flag_description,

    axis=1

)


alerts["status"] = (

    "OPEN"

)


alerts["analyst_decision"] = (

    "PENDING"

)


alerts["sar_recommendation"] = np.where(

    alerts["rule_risk_score"]
    >=
    75,

    "REVIEW_FOR_SAR",

    "NO_AUTO_SAR"

)


# ============================================================
# 12. README
# ============================================================

readme = pd.DataFrame({

    "Item": [

        "Project",
        "Dataset Type",
        "Customers",
        "Accounts",
        "Transactions",
        "AML Scenarios",
        "Purpose",
        "Important Note"

    ],

    "Description": [

        "US AML Financial Crime Transaction Monitoring System",

        "Synthetic / educational dataset",

        f"{len(customers):,}",

        f"{len(accounts):,}",

        f"{len(transactions):,}",

        (

            "Structuring; Rapid Movement; Funnel Account; "
            "Layering; Mule Account; Fraud Linked"

        ),

        (

            "Portfolio project for AML transaction monitoring, "
            "rules, machine learning and investigation"

        ),

        (

            "All data is synthetic. It does not represent real "
            "customers, transactions or regulatory records."

        )

    ]

})


# ============================================================
# 13. DATA DICTIONARY
# ============================================================

data_dictionary = pd.DataFrame({

    "Table": [

        "Customers",
        "Accounts",
        "Transactions",
        "Alerts"

    ],

    "Purpose": [

        "KYC/CDD customer profile and risk information",

        "Bank account master information",

        "Transaction monitoring dataset",

        "Transactions converted into investigation alerts"

    ]

})


# ============================================================
# 14. SAVE EXCEL WORKBOOK
# ============================================================

print("\nSaving Excel workbook...")


with pd.ExcelWriter(

    OUTPUT_FILE,

    engine="openpyxl"

) as writer:

    customers.to_excel(

        writer,

        sheet_name="Customers",

        index=False

    )


    accounts.to_excel(

        writer,

        sheet_name="Accounts",

        index=False

    )


    transactions.to_excel(

        writer,

        sheet_name="Transactions",

        index=False

    )


    alerts.to_excel(

        writer,

        sheet_name="Alerts",

        index=False

    )


    readme.to_excel(

        writer,

        sheet_name="README",

        index=False

    )


    data_dictionary.to_excel(

        writer,

        sheet_name="Data_Dictionary",

        index=False

    )


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)

print(
    "US AML DATASET CREATED SUCCESSFULLY"
)

print("=" * 60)


print(

    f"Customers      : "
    f"{len(customers):,}"

)


print(

    f"Accounts       : "
    f"{len(accounts):,}"

)


print(

    f"Transactions   : "
    f"{len(transactions):,}"

)


print(

    f"AML Alerts     : "
    f"{len(alerts):,}"

)


print(

    f"Suspicious TXs : "
    f"{transactions['known_suspicious'].sum():,}"

)


print(
    "\nScenario distribution:"
)


print(

    transactions[

        transactions["known_suspicious"]
        ==
        1

    ]["scenario"]

    .value_counts()

    .to_string()

)


print(
    "\nExcel file:"
)


print(
    OUTPUT_FILE
)


print(
    "\nPhase 1 completed."
)


print(
    "Next step: Exploratory Data Analysis (EDA)."
)