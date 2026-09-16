# =============================================================================
# PHASE 10 - AI-ASSISTED SAR NARRATIVE GENERATOR
# US FINANCIAL CRIME / AML TRANSACTION MONITORING PROJECT
# =============================================================================
#
# Purpose:
#   Generate structured, analyst-reviewable SAR narrative drafts from
#   AML investigation evidence.
#
# Design:
#   Investigation Case
#        ->
#   Customer Profile
#        ->
#   Transaction Activity
#        ->
#   Red Flags
#        ->
#   ML / SHAP Evidence
#        ->
#   Narrative Draft
#        ->
#   Quality Validation
#
# IMPORTANT:
#   This system generates a DRAFT only.
#
#   It does NOT:
#       - determine criminal activity
#       - make a final SAR filing decision
#       - automatically file a SAR
#       - replace a qualified AML/BSA investigator
#
#   All generated narratives require human review.
#
# =============================================================================

import os
import re
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


CASE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "aml_investigation_cases.csv"
)


ALERT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "aml_alerts_scored.csv"
)


PROFILE_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "investigation",
    "customer_investigation_profiles.csv"
)


RED_FLAG_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "investigation",
    "investigation_red_flags.csv"
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "sar"
)


OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "sar_narrative_drafts.csv"
)


QUALITY_FILE = os.path.join(
    OUTPUT_DIR,
    "sar_narrative_quality_report.csv"
)


TEXT_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "narratives"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


os.makedirs(
    TEXT_OUTPUT_DIR,
    exist_ok=True
)


# =============================================================================
# 2. HEADER
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 10 - AI-ASSISTED SAR NARRATIVE GENERATOR")
print("=" * 80)


print(
    "\nThis system generates analyst-reviewable SAR narrative drafts."
)


print(
    "It does NOT automatically determine or file a SAR."
)


# =============================================================================
# 3. LOAD DATA
# =============================================================================

print("\n" + "=" * 80)
print("3. LOADING INVESTIGATION DATA")
print("=" * 80)


if not os.path.exists(CASE_FILE):

    print(
        "\nERROR: Investigation case file not found."
    )

    print(
        CASE_FILE
    )

    print(
        "\nPlease run Phase 8 first."
    )

    raise SystemExit(1)


cases = pd.read_csv(
    CASE_FILE
)


print(
    f"\nInvestigation cases loaded: {len(cases):,}"
)


if os.path.exists(ALERT_FILE):

    alerts = pd.read_csv(
        ALERT_FILE
    )

else:

    alerts = pd.DataFrame()


if os.path.exists(PROFILE_FILE):

    profiles = pd.read_csv(
        PROFILE_FILE
    )

else:

    profiles = pd.DataFrame()


if os.path.exists(RED_FLAG_FILE):

    red_flags = pd.read_csv(
        RED_FLAG_FILE
    )

else:

    red_flags = pd.DataFrame()


# =============================================================================
# 4. CLEAN DATA
# =============================================================================

for dataframe in [
    cases,
    alerts,
    profiles,
    red_flags
]:

    dataframe.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


if "timestamp" in alerts.columns:

    alerts[
        "timestamp"
    ] = pd.to_datetime(
        alerts[
            "timestamp"
        ],
        errors="coerce"
    )


# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================

def safe_value(
    row,
    column,
    default=""
):

    if column not in row.index:

        return default


    value = row[column]


    if pd.isna(value):

        return default


    return value


def money(
    value
):

    try:

        return "${:,.2f}".format(
            float(value)
        )

    except Exception:

        return "$0.00"


def clean_text(
    value
):

    if value is None:

        return ""


    if pd.isna(value):

        return ""


    return str(
        value
    ).strip()


def unique_list(
    values
):

    result = []


    for value in values:

        value = clean_text(
            value
        )


        if (
            value
            and
            value not in result
        ):

            result.append(
                value
            )


    return result


# =============================================================================
# 6. TRANSACTION SUMMARY
# =============================================================================

def build_transaction_summary(
    customer_transactions
):

    if customer_transactions.empty:

        return {
            "transaction_count": 0,
            "total_volume": 0,
            "maximum_amount": 0,
            "minimum_amount": 0,
            "first_date": "",
            "last_date": "",
            "international_count": 0,
            "cash_count": 0,
            "transaction_types": [],
            "channels": [],
            "origin_countries": [],
            "destination_countries": []
        }


    amounts = pd.to_numeric(
        customer_transactions[
            "amount_usd"
        ],
        errors="coerce"
    ).fillna(0)


    transaction_count = len(
        customer_transactions
    )


    total_volume = amounts.sum()


    maximum_amount = amounts.max()


    minimum_amount = amounts.min()


    if "timestamp" in customer_transactions.columns:

        timestamps = pd.to_datetime(
            customer_transactions[
                "timestamp"
            ],
            errors="coerce"
        ).dropna()

    else:

        timestamps = pd.Series(
            dtype="datetime64[ns]"
        )


    if len(timestamps) > 0:

        first_date = timestamps.min().strftime(
            "%Y-%m-%d"
        )

        last_date = timestamps.max().strftime(
            "%Y-%m-%d"
        )

    else:

        first_date = ""

        last_date = ""


    if "is_international" in customer_transactions.columns:

        international_count = int(
            pd.to_numeric(
                customer_transactions[
                    "is_international"
                ],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

    else:

        international_count = 0


    if "is_cash" in customer_transactions.columns:

        cash_count = int(
            pd.to_numeric(
                customer_transactions[
                    "is_cash"
                ],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

    else:

        cash_count = 0


    transaction_types = []


    if "transaction_type" in customer_transactions.columns:

        transaction_types = unique_list(
            customer_transactions[
                "transaction_type"
            ]
            .dropna()
            .tolist()
        )


    channels = []


    if "channel" in customer_transactions.columns:

        channels = unique_list(
            customer_transactions[
                "channel"
            ]
            .dropna()
            .tolist()
        )


    origin_countries = []


    if "origin_country" in customer_transactions.columns:

        origin_countries = unique_list(
            customer_transactions[
                "origin_country"
            ]
            .dropna()
            .tolist()
        )


    destination_countries = []


    if "destination_country" in customer_transactions.columns:

        destination_countries = unique_list(
            customer_transactions[
                "destination_country"
            ]
            .dropna()
            .tolist()
        )


    return {

        "transaction_count":
            transaction_count,

        "total_volume":
            total_volume,

        "maximum_amount":
            maximum_amount,

        "minimum_amount":
            minimum_amount,

        "first_date":
            first_date,

        "last_date":
            last_date,

        "international_count":
            international_count,

        "cash_count":
            cash_count,

        "transaction_types":
            transaction_types,

        "channels":
            channels,

        "origin_countries":
            origin_countries,

        "destination_countries":
            destination_countries
    }


# =============================================================================
# 7. BUILD RED FLAG SUMMARY
# =============================================================================

def build_red_flag_summary(
    customer_id
):

    if red_flags.empty:

        return []


    if "customer_id" not in red_flags.columns:

        return []


    customer_flags = red_flags[
        red_flags[
            "customer_id"
        ]
        .astype(str)
        ==
        str(customer_id)
    ]


    if customer_flags.empty:

        return []


    descriptions = []


    if "red_flag_description" in customer_flags.columns:

        descriptions = (
            customer_flags[
                "red_flag_description"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )


    return unique_list(
        descriptions
    )


# =============================================================================
# 8. BUILD NARRATIVE
# =============================================================================

def generate_sar_narrative(
    case,
    customer_profile,
    transaction_summary,
    red_flag_descriptions,
    customer_transactions
):

    customer_id = clean_text(
        safe_value(
            case,
            "customer_id"
        )
    )


    case_id = clean_text(
        safe_value(
            case,
            "case_id"
        )
    )


    customer_type = clean_text(
        safe_value(
            case,
            "customer_type",
            "customer"
        )
    )


    occupation = clean_text(
        safe_value(
            case,
            "occupation"
        )
    )


    state = clean_text(
        safe_value(
            customer_profile,
            "state"
        )
    )


    risk_rating = clean_text(
        safe_value(
            case,
            "customer_risk_rating"
        )
    )


    annual_income = safe_value(
        case,
        "annual_income_usd",
        0
    )


    expected_volume = safe_value(
        case,
        "expected_monthly_volume_usd",
        0
    )


    case_priority = clean_text(
        safe_value(
            case,
            "case_priority"
        )
    )


    risk_level = clean_text(
        safe_value(
            case,
            "case_risk_level"
        )
    )


    risk_score = safe_value(
        case,
        "highest_risk_score",
        0
    )


    primary_alert_type = clean_text(
        safe_value(
            case,
            "primary_alert_type"
        )
    )


    primary_alert_reason = clean_text(
        safe_value(
            case,
            "primary_alert_reason"
        )
    )


    model_rule_agreement = clean_text(
        safe_value(
            case,
            "model_rule_agreement"
        )
    )


    # -------------------------------------------------------------------------
    # Customer identification
    # -------------------------------------------------------------------------

    customer_description = (
        f"Customer {customer_id}"
    )


    if customer_type:

        customer_description += (
            f" is identified in the synthetic dataset "
            f"as a {customer_type}"
        )


    if occupation:

        customer_description += (
            f" with an occupation recorded as {occupation}"
        )


    if state:

        customer_description += (
            f" in {state}"
        )


    customer_description += "."


    # -------------------------------------------------------------------------
    # Customer profile context
    # -------------------------------------------------------------------------

    profile_sentence = (
        f"The recorded annual income is "
        f"{money(annual_income)}, and the expected "
        f"monthly transaction volume is "
        f"{money(expected_volume)}."
    )


    if risk_rating:

        profile_sentence += (
            f" The recorded customer risk rating is "
            f"{risk_rating}."
        )


    # -------------------------------------------------------------------------
    # Activity period
    # -------------------------------------------------------------------------

    first_date = transaction_summary[
        "first_date"
    ]


    last_date = transaction_summary[
        "last_date"
    ]


    if first_date and last_date:

        period_sentence = (
            f"During the period from {first_date} "
            f"through {last_date}, "
        )

    elif first_date:

        period_sentence = (
            f"On activity beginning {first_date}, "
        )

    else:

        period_sentence = (
            "During the reviewed activity period, "
        )


    period_sentence += (
        f"the customer conducted "
        f"{transaction_summary['transaction_count']:,} "
        f"reviewed transactions totaling "
        f"{money(transaction_summary['total_volume'])}."
    )


    # -------------------------------------------------------------------------
    # Transaction behavior
    # -------------------------------------------------------------------------

    behavior_sentences = []


    if transaction_summary[
        "international_count"
    ] > 0:

        behavior_sentences.append(
            f"{transaction_summary['international_count']:,} "
            f"transactions were identified as international."
        )


    if transaction_summary[
        "cash_count"
    ] > 0:

        behavior_sentences.append(
            f"{transaction_summary['cash_count']:,} "
            f"transactions involved cash activity."
        )


    if transaction_summary[
        "transaction_types"
    ]:

        behavior_sentences.append(
            "Transaction types observed included "
            +
            ", ".join(
                transaction_summary[
                    "transaction_types"
                ][:8]
            )
            +
            "."
        )


    if transaction_summary[
        "channels"
    ]:

        behavior_sentences.append(
            "Channels observed included "
            +
            ", ".join(
                transaction_summary[
                    "channels"
                ][:8]
            )
            +
            "."
        )


    if transaction_summary[
        "origin_countries"
    ]:

        behavior_sentences.append(
            "Observed origin countries included "
            +
            ", ".join(
                transaction_summary[
                    "origin_countries"
                ][:10]
            )
            +
            "."
        )


    if transaction_summary[
        "destination_countries"
    ]:

        behavior_sentences.append(
            "Observed destination countries included "
            +
            ", ".join(
                transaction_summary[
                    "destination_countries"
                ][:10]
            )
            +
            "."
        )


    # -------------------------------------------------------------------------
    # Alert description
    # -------------------------------------------------------------------------

    alert_sentence = ""


    if primary_alert_type:

        alert_sentence = (
            f"The activity generated an AML monitoring alert "
            f"categorized as {primary_alert_type}."
        )


    if primary_alert_reason:

        alert_sentence += (
            f" The alert was generated because "
            f"{primary_alert_reason}"
        )


        if not alert_sentence.endswith("."):

            alert_sentence += "."


    # -------------------------------------------------------------------------
    # Red flags
    # -------------------------------------------------------------------------

    red_flag_sentence = ""


    if red_flag_descriptions:

        red_flag_sentence = (
            "The investigation identified the following "
            "behavioral indicators: "
            +
            "; ".join(
                red_flag_descriptions[:10]
            )
            +
            "."
        )


    else:

        red_flag_sentence = (
            "No additional behavioral indicators were "
            "identified by the current investigation rules."
        )


    # -------------------------------------------------------------------------
    # Model evidence
    # -------------------------------------------------------------------------

    model_sentence = (
        f"The combined AML risk score for the case was "
        f"{float(risk_score):.2f}, with a case risk level "
        f"of {risk_level} and investigation priority "
        f"{case_priority}."
    )


    if model_rule_agreement:

        model_sentence += (
            f" Model and rule assessment was recorded as "
            f"{model_rule_agreement}."
        )


    # -------------------------------------------------------------------------
    # Why unusual
    # -------------------------------------------------------------------------

    why_sentence = (
        "The activity was considered unusual for review because "
        "the observed transaction behavior, transaction volume, "
        "frequency, geographic activity and/or other risk indicators "
        "required additional assessment against the customer's "
        "known profile and expected activity."
    )


    if expected_volume:

        if transaction_summary[
            "total_volume"
        ] > float(
            expected_volume
        ):

            why_sentence += (
                f" The reviewed transaction volume of "
                f"{money(transaction_summary['total_volume'])} "
                f"was greater than the recorded expected monthly "
                f"volume of {money(expected_volume)}."
            )


    # -------------------------------------------------------------------------
    # Transaction examples
    # -------------------------------------------------------------------------

    transaction_examples = ""


    if not customer_transactions.empty:

        sample_transactions = (
            customer_transactions
            .copy()
        )


        if "timestamp" in sample_transactions.columns:

            sample_transactions = (
                sample_transactions
                .sort_values(
                    "timestamp"
                )
            )


        sample_transactions = (
            sample_transactions
            .head(5)
        )


        example_parts = []


        for _, transaction in sample_transactions.iterrows():

            transaction_id = clean_text(
                transaction.get(
                    "transaction_id",
                    ""
                )
            )


            amount = transaction.get(
                "amount_usd",
                0
            )


            timestamp = ""


            if "timestamp" in transaction.index:

                timestamp_value = pd.to_datetime(
                    transaction[
                        "timestamp"
                    ],
                    errors="coerce"
                )


                if not pd.isna(
                    timestamp_value
                ):

                    timestamp = timestamp_value.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )


            if transaction_id:

                example_parts.append(
                    f"transaction {transaction_id} "
                    f"for {money(amount)} "
                    f"on {timestamp}"
                )


        if example_parts:

            transaction_examples = (
                "Examples of reviewed transactions include "
                +
                "; ".join(
                    example_parts
                )
                +
                "."
            )


    # -------------------------------------------------------------------------
    # Final narrative
    # -------------------------------------------------------------------------

    narrative_parts = [

        f"Case {case_id} was reviewed for potential "
        f"suspicious financial activity involving "
        f"{customer_description}",

        profile_sentence,

        period_sentence,

        alert_sentence,

        " ".join(
            behavior_sentences
        ),

        red_flag_sentence,

        model_sentence,

        why_sentence,

        transaction_examples,

        "The information above represents the available "
        "synthetic investigation evidence for analyst review. "
        "Any SAR filing determination should be made by an "
        "authorized financial-crime professional after review "
        "of the relevant facts and supporting information."
    ]


    narrative = " ".join(
        part.strip()
        for part in narrative_parts
        if part
        and part.strip()
    )


    # Clean excessive whitespace

    narrative = re.sub(
        r"\s+",
        " ",
        narrative
    ).strip()


    return narrative


# =============================================================================
# 9. NARRATIVE QUALITY CHECKS
# =============================================================================

def evaluate_narrative(
    narrative
):

    narrative_lower = narrative.lower()


    checks = {

        "has_who":
            any(
                term in narrative_lower
                for term in [
                    "customer",
                    "entity",
                    "individual"
                ]
            ),

        "has_what":
            any(
                term in narrative_lower
                for term in [
                    "transaction",
                    "activity",
                    "transfer",
                    "payment"
                ]
            ),

        "has_when":
            bool(
                re.search(
                    r"\b20\d{2}-\d{2}-\d{2}\b",
                    narrative
                )
            ),

        "has_where":
            any(
                term in narrative_lower
                for term in [
                    "country",
                    "state",
                    "international",
                    "geographic"
                ]
            ),

        "has_why":
            any(
                term in narrative_lower
                for term in [
                    "unusual",
                    "suspicious",
                    "risk",
                    "expected"
                ]
            ),

        "has_amount":
            "$" in narrative,

        "chronological_language":
            any(
                term in narrative_lower
                for term in [
                    "during the period",
                    "on activity",
                    "on ",
                    "from "
                ]
            ),

        "human_review_statement":
            "human" in narrative_lower
            or
            "analyst" in narrative_lower
    }


    passed = sum(
        1
        for value in checks.values()
        if value
    )


    total = len(
        checks
    )


    quality_score = (
        passed
        /
        total
        *
        100
    )


    return (
        checks,
        round(
            quality_score,
            2
        )
    )


# =============================================================================
# 10. GENERATE NARRATIVES
# =============================================================================

print("\n" + "=" * 80)
print("10. GENERATING SAR NARRATIVE DRAFTS")
print("=" * 80)


narrative_rows = []
quality_rows = []


# -------------------------------------------------------------------------
# Use every investigation case
# -------------------------------------------------------------------------

for _, case in cases.iterrows():

    customer_id = clean_text(
        safe_value(
            case,
            "customer_id"
        )
    )


    # -------------------------------------------------------------------------
    # Customer profile
    # -------------------------------------------------------------------------

    if not profiles.empty:

        matching_profiles = profiles[
            profiles[
                "customer_id"
            ]
            .astype(str)
            ==
            str(customer_id)
        ]

    else:

        matching_profiles = pd.DataFrame()


    if not matching_profiles.empty:

        customer_profile = (
            matching_profiles
            .iloc[0]
        )

    else:

        customer_profile = pd.Series(
            dtype=object
        )


    # -------------------------------------------------------------------------
    # Customer transactions
    # -------------------------------------------------------------------------

    if not alerts.empty:

        customer_transactions = alerts[
            alerts[
                "customer_id"
            ]
            .astype(str)
            ==
            str(customer_id)
        ].copy()

    else:

        customer_transactions = pd.DataFrame()


    # -------------------------------------------------------------------------
    # Transaction summary
    # -------------------------------------------------------------------------

    transaction_summary = (
        build_transaction_summary(
            customer_transactions
        )
    )


    # -------------------------------------------------------------------------
    # Red flags
    # -------------------------------------------------------------------------

    red_flag_descriptions = (
        build_red_flag_summary(
            customer_id
        )
    )


    # -------------------------------------------------------------------------
    # Generate narrative
    # -------------------------------------------------------------------------

    narrative = generate_sar_narrative(
        case,
        customer_profile,
        transaction_summary,
        red_flag_descriptions,
        customer_transactions
    )


    # -------------------------------------------------------------------------
    # Quality evaluation
    # -------------------------------------------------------------------------

    checks, quality_score = (
        evaluate_narrative(
            narrative
        )
    )


    # -------------------------------------------------------------------------
    # Save narrative row
    # -------------------------------------------------------------------------

    narrative_rows.append({

        "case_id":
            clean_text(
                safe_value(
                    case,
                    "case_id"
                )
            ),

        "customer_id":
            customer_id,

        "case_priority":
            clean_text(
                safe_value(
                    case,
                    "case_priority"
                )
            ),

        "case_risk_level":
            clean_text(
                safe_value(
                    case,
                    "case_risk_level"
                )
            ),

        "risk_score":
            safe_value(
                case,
                "highest_risk_score",
                0
            ),

        "primary_alert_type":
            clean_text(
                safe_value(
                    case,
                    "primary_alert_type"
                )
            ),

        "red_flag_count":
            len(
                red_flag_descriptions
            ),

        "transaction_count":
            transaction_summary[
                "transaction_count"
            ],

        "total_transaction_volume_usd":
            round(
                transaction_summary[
                    "total_volume"
                ],
                2
            ),

        "narrative_quality_score":
            quality_score,

        "sar_narrative_draft":
            narrative,

        "human_review_required":
            "YES",

        "automatic_sar_filing":
            "NO"

    })


    quality_rows.append({

        "case_id":
            clean_text(
                safe_value(
                    case,
                    "case_id"
                )
            ),

        "customer_id":
            customer_id,

        "quality_score":
            quality_score,

        "has_who":
            checks[
                "has_who"
            ],

        "has_what":
            checks[
                "has_what"
            ],

        "has_when":
            checks[
                "has_when"
            ],

        "has_where":
            checks[
                "has_where"
            ],

        "has_why":
            checks[
                "has_why"
            ],

        "has_amount":
            checks[
                "has_amount"
            ],

        "chronological_language":
            checks[
                "chronological_language"
            ],

        "human_review_statement":
            checks[
                "human_review_statement"
            ]

    })


# =============================================================================
# 11. SAVE CSV OUTPUT
# =============================================================================

narrative_df = pd.DataFrame(
    narrative_rows
)


quality_df = pd.DataFrame(
    quality_rows
)


narrative_df.to_csv(
    OUTPUT_FILE,
    index=False
)


quality_df.to_csv(
    QUALITY_FILE,
    index=False
)


print(
    f"\nSAR narrative drafts generated: "
    f"{len(narrative_df):,}"
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    QUALITY_FILE
)


# =============================================================================
# 12. SAVE INDIVIDUAL TEXT NARRATIVES
# =============================================================================

print("\n" + "=" * 80)
print("12. SAVING INDIVIDUAL NARRATIVE FILES")
print("=" * 80)


for _, row in narrative_df.iterrows():

    case_id = clean_text(
        row[
            "case_id"
        ]
    )


    safe_case_id = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        case_id
    )


    output_path = os.path.join(
        TEXT_OUTPUT_DIR,
        f"{safe_case_id}_SAR_DRAFT.txt"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SAR NARRATIVE DRAFT\n"
        )

        file.write(
            "=" * 75 + "\n\n"
        )


        file.write(
            f"Case ID: {case_id}\n"
        )


        file.write(
            f"Customer ID: "
            f"{row['customer_id']}\n"
        )


        file.write(
            f"Risk Level: "
            f"{row['case_risk_level']}\n"
        )


        file.write(
            f"Risk Score: "
            f"{row['risk_score']}\n"
        )


        file.write(
            f"Primary Alert Type: "
            f"{row['primary_alert_type']}\n\n"
        )


        file.write(
            "NARRATIVE\n"
        )

        file.write(
            "-" * 75 + "\n\n"
        )


        file.write(
            row[
                "sar_narrative_draft"
            ]
        )


        file.write(
            "\n\n"
        )


        file.write(
            "HUMAN REVIEW REQUIRED: YES\n"
        )


        file.write(
            "AUTOMATIC SAR FILING: NO\n"
        )


# =============================================================================
# 13. SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("13. SAR NARRATIVE QUALITY SUMMARY")
print("=" * 80)


if not quality_df.empty:

    average_quality = (
        quality_df[
            "quality_score"
        ]
        .mean()
    )


    minimum_quality = (
        quality_df[
            "quality_score"
        ]
        .min()
    )


    maximum_quality = (
        quality_df[
            "quality_score"
        ]
        .max()
    )


    print(
        f"\nAverage narrative quality score: "
        f"{average_quality:.2f}%"
    )


    print(
        f"Minimum narrative quality score: "
        f"{minimum_quality:.2f}%"
    )


    print(
        f"Maximum narrative quality score: "
        f"{maximum_quality:.2f}%"
    )


# =============================================================================
# 14. SAMPLE NARRATIVE
# =============================================================================

print("\n" + "=" * 80)
print("14. SAMPLE SAR NARRATIVE")
print("=" * 80)


if not narrative_df.empty:

    sample = (
        narrative_df
        .sort_values(
            "risk_score",
            ascending=False
        )
        .iloc[0]
    )


    print(
        "\nCase:"
    )

    print(
        sample[
            "case_id"
        ]
    )


    print(
        "\nRisk level:"
    )

    print(
        sample[
            "case_risk_level"
        ]
    )


    print(
        "\nRisk score:"
    )

    print(
        sample[
            "risk_score"
        ]
    )


    print(
        "\nNarrative:"
    )

    print(
        "\n"
        +
        sample[
            "sar_narrative_draft"
        ]
    )


# =============================================================================
# 15. FINAL OUTPUTS
# =============================================================================

print("\n" + "=" * 80)
print("15. PHASE 10 OUTPUT FILES")
print("=" * 80)


print(
    "\n1. SAR narrative dataset:"
)

print(
    OUTPUT_FILE
)


print(
    "\n2. Narrative quality report:"
)

print(
    QUALITY_FILE
)


print(
    "\n3. Individual SAR narrative drafts:"
)

print(
    TEXT_OUTPUT_DIR
)


# =============================================================================
# 16. COMPLETION
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 10 COMPLETED SUCCESSFULLY")
print("=" * 80)


print(
    "\nYour AML project now includes:"
)


print(
    "  [OK] Customer investigation evidence"
)


print(
    "  [OK] Transaction behavior analysis"
)


print(
    "  [OK] AML red flags"
)


print(
    "  [OK] ML risk assessment"
)


print(
    "  [OK] SHAP explainability"
)


print(
    "  [OK] Investigation cases"
)


print(
    "  [OK] Analyst disposition workflow"
)


print(
    "  [OK] SAR narrative draft generation"
)


print(
    "  [OK] SAR narrative quality validation"
)


print(
    "  [OK] Human-in-the-loop SAR workflow"
)


print(
    "\nNext phase:"
)


print(
    "PHASE 11 - FINAL AML PROJECT VALIDATION + PORTFOLIO PACKAGING"
)


print(
    "=" * 80
)