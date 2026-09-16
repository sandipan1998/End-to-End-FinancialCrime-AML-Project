# =============================================================================
# PHASE 9 - INTERACTIVE AML INVESTIGATION DASHBOARD
# US FINANCIAL CRIME / AML TRANSACTION MONITORING PROJECT
# =============================================================================
#
# Technology:
#   Streamlit
#   Pandas
#   Plotly
#
# Purpose:
#   Provide an analyst-style dashboard for:
#       - AML alert monitoring
#       - customer investigation
#       - transaction review
#       - red-flag analysis
#       - ML risk scoring
#       - SHAP explainability
#       - investigation case management
#
# IMPORTANT:
#   This is a synthetic portfolio project.
#   Risk scores and red flags are indicators only.
#   They do not establish criminal activity.
#   SAR decisions require appropriate human investigation,
#   evidence, policies and institutional procedures.
#
# =============================================================================

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
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


CUSTOMER_PROFILE_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "investigation",
    "customer_investigation_profiles.csv"
)


TRANSACTION_REVIEW_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "investigation",
    "transaction_review_summary.csv"
)


RED_FLAG_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "investigation",
    "investigation_red_flags.csv"
)


CASE_SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "investigation",
    "case_summary.csv"
)


SHAP_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "shap",
    "shap_feature_importance.csv"
)


SHAP_TRANSACTION_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "shap",
    "transaction_shap_explanations.csv"
)


UPDATE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "analyst_case_updates.csv"
)


# =============================================================================
# 2. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AML Investigation Dashboard",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# 3. CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 16px;
        margin-bottom: 20px;
    }

    .risk-critical {
        font-size: 24px;
        font-weight: 700;
    }

    .warning-box {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #d6d6d6;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# 4. HEADER
# =============================================================================

st.markdown(
    '<div class="main-title">🔎 AML Investigation Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'US Financial Crime / AML Transaction Monitoring Portfolio Project'
    '</div>',
    unsafe_allow_html=True
)


st.warning(
    "Synthetic portfolio data only. Risk scores, red flags and model "
    "predictions are investigative indicators and are not legal conclusions. "
    "SAR decisions require appropriate human review and institutional procedures."
)


# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def load_csv(path):

    if not os.path.exists(path):

        return pd.DataFrame()

    try:

        data = pd.read_csv(path)

        return data

    except Exception:

        return pd.DataFrame()


def money(value):

    try:

        return "${:,.2f}".format(float(value))

    except Exception:

        return "$0.00"


def integer(value):

    try:

        return "{:,.0f}".format(float(value))

    except Exception:

        return "0"


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)


def get_first_existing_column(
    dataframe,
    candidates,
    default=None
):

    for column in candidates:

        if column in dataframe.columns:

            return column

    return default


# =============================================================================
# 6. LOAD DATA
# =============================================================================

cases = load_csv(
    CASE_FILE
)

alerts = load_csv(
    ALERT_FILE
)

customer_profiles = load_csv(
    CUSTOMER_PROFILE_FILE
)

transaction_review = load_csv(
    TRANSACTION_REVIEW_FILE
)

red_flags = load_csv(
    RED_FLAG_FILE
)

case_summary = load_csv(
    CASE_SUMMARY_FILE
)

shap_features = load_csv(
    SHAP_FILE
)

shap_transactions = load_csv(
    SHAP_TRANSACTION_FILE
)


# =============================================================================
# 7. VALIDATE DATA
# =============================================================================

if cases.empty:

    st.error(
        "Investigation case file was not found."
    )

    st.code(
        CASE_FILE
    )

    st.info(
        "Please run Phase 8 before starting the dashboard."
    )

    st.stop()


if not alerts.empty and "timestamp" in alerts.columns:

    alerts["timestamp"] = pd.to_datetime(
        alerts["timestamp"],
        errors="coerce"
    )


# =============================================================================
# 8. ANALYST UPDATES
# =============================================================================

if os.path.exists(UPDATE_FILE):

    updates = load_csv(
        UPDATE_FILE
    )

else:

    updates = pd.DataFrame(
        columns=[
            "case_id",
            "analyst_disposition",
            "investigation_status",
            "analyst_notes",
            "escalation_required",
            "sar_consideration",
            "updated_at"
        ]
    )


# =============================================================================
# 9. APPLY ANALYST UPDATES
# =============================================================================

if not updates.empty:

    update_columns = [
        "case_id",
        "analyst_disposition",
        "investigation_status",
        "analyst_notes",
        "escalation_required",
        "sar_consideration",
        "updated_at"
    ]

    available_updates = [
        column
        for column in update_columns
        if column in updates.columns
    ]

    if "case_id" in available_updates:

        cases = cases.merge(
            updates[
                available_updates
            ],
            on="case_id",
            how="left",
            suffixes=(
                "",
                "_update"
            )
        )

        for column in [
            "analyst_disposition",
            "investigation_status",
            "analyst_notes",
            "escalation_required",
            "sar_consideration",
            "updated_at"
        ]:

            update_column = column + "_update"

            if update_column in cases.columns:

                cases[column] = cases[
                    update_column
                ].fillna(
                    cases.get(
                        column,
                        ""
                    )
                )

                cases.drop(
                    columns=[
                        update_column
                    ],
                    inplace=True
                )


# =============================================================================
# 10. SIDEBAR
# =============================================================================

st.sidebar.title(
    "AML Analyst Controls"
)


st.sidebar.markdown(
    "Use the filters below to prioritize investigation cases."
)


# -----------------------------------------------------------------------------
# Priority filter
# -----------------------------------------------------------------------------

priority_values = [
    "ALL"
]

if "case_priority" in cases.columns:

    priority_values.extend(
        sorted(
            cases[
                "case_priority"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )


selected_priority = st.sidebar.selectbox(
    "Case Priority",
    priority_values
)


# -----------------------------------------------------------------------------
# Risk level filter
# -----------------------------------------------------------------------------

risk_values = [
    "ALL"
]

if "case_risk_level" in cases.columns:

    risk_values.extend(
        sorted(
            cases[
                "case_risk_level"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )


selected_risk = st.sidebar.selectbox(
    "Risk Level",
    risk_values
)


# -----------------------------------------------------------------------------
# Alert type filter
# -----------------------------------------------------------------------------

alert_type_values = [
    "ALL"
]

if "primary_alert_type" in cases.columns:

    alert_type_values.extend(
        sorted(
            cases[
                "primary_alert_type"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )


selected_alert_type = st.sidebar.selectbox(
    "Alert Type",
    alert_type_values
)


# -----------------------------------------------------------------------------
# Disposition filter
# -----------------------------------------------------------------------------

disposition_values = [
    "ALL"
]

if "analyst_disposition" in cases.columns:

    disposition_values.extend(
        sorted(
            cases[
                "analyst_disposition"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )


selected_disposition = st.sidebar.selectbox(
    "Analyst Disposition",
    disposition_values
)


# =============================================================================
# 11. FILTER CASES
# =============================================================================

filtered_cases = cases.copy()


if selected_priority != "ALL":

    filtered_cases = filtered_cases[
        filtered_cases[
            "case_priority"
        ]
        .astype(str)
        ==
        selected_priority
    ]


if selected_risk != "ALL":

    filtered_cases = filtered_cases[
        filtered_cases[
            "case_risk_level"
        ]
        .astype(str)
        ==
        selected_risk
    ]


if selected_alert_type != "ALL":

    filtered_cases = filtered_cases[
        filtered_cases[
            "primary_alert_type"
        ]
        .astype(str)
        ==
        selected_alert_type
    ]


if selected_disposition != "ALL":

    filtered_cases = filtered_cases[
        filtered_cases[
            "analyst_disposition"
        ]
        .astype(str)
        ==
        selected_disposition
    ]


# =============================================================================
# 12. KPI SECTION
# =============================================================================

st.subheader(
    "📊 AML Investigation Overview"
)


total_cases = len(
    cases
)


filtered_case_count = len(
    filtered_cases
)


critical_cases = (
    len(
        cases[
            cases[
                "case_risk_level"
            ]
            .astype(str)
            .str.upper()
            ==
            "CRITICAL"
        ]
    )
    if "case_risk_level" in cases.columns
    else 0
)


high_cases = (
    len(
        cases[
            cases[
                "case_risk_level"
            ]
            .astype(str)
            .str.upper()
            ==
            "HIGH"
        ]
    )
    if "case_risk_level" in cases.columns
    else 0
)


open_cases = (
    len(
        cases[
            cases[
                "investigation_status"
            ]
            .astype(str)
            .str.upper()
            ==
            "OPEN"
        ]
    )
    if "investigation_status" in cases.columns
    else 0
)


total_alerts = len(
    alerts
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Total Cases",
        integer(total_cases)
    )


with col2:

    st.metric(
        "Filtered Cases",
        integer(filtered_case_count)
    )


with col3:

    st.metric(
        "Critical Cases",
        integer(critical_cases)
    )


with col4:

    st.metric(
        "High Risk Cases",
        integer(high_cases)
    )


with col5:

    st.metric(
        "Open Cases",
        integer(open_cases)
    )


# =============================================================================
# 13. CASE PRIORITY / RISK DISTRIBUTION
# =============================================================================

st.subheader(
    "📈 Case Risk & Priority Analytics"
)


chart_col1, chart_col2 = st.columns(2)


with chart_col1:

    if "case_priority" in cases.columns:

        priority_df = (
            cases[
                "case_priority"
            ]
            .value_counts()
            .reset_index()
        )

        priority_df.columns = [
            "priority",
            "case_count"
        ]

        fig_priority = px.bar(
            priority_df,
            x="priority",
            y="case_count",
            title="Cases by Investigation Priority",
            text="case_count"
        )

        fig_priority.update_layout(
            xaxis_title="Priority",
            yaxis_title="Cases"
        )

        st.plotly_chart(
            fig_priority,
            use_container_width=True
        )


with chart_col2:

    if "case_risk_level" in cases.columns:

        risk_df = (
            cases[
                "case_risk_level"
            ]
            .value_counts()
            .reset_index()
        )

        risk_df.columns = [
            "risk_level",
            "case_count"
        ]

        fig_risk = px.bar(
            risk_df,
            x="risk_level",
            y="case_count",
            title="Cases by Risk Level",
            text="case_count"
        )

        fig_risk.update_layout(
            xaxis_title="Risk Level",
            yaxis_title="Cases"
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )


# =============================================================================
# 14. ALERT TYPE DISTRIBUTION
# =============================================================================

if not alerts.empty:

    st.subheader(
        "🚨 AML Alert Type Distribution"
    )

    alert_type_column = get_first_existing_column(
        alerts,
        [
            "alert_type",
            "primary_alert_type"
        ]
    )

    if alert_type_column:

        alert_type_df = (
            alerts[
                alert_type_column
            ]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        alert_type_df.columns = [
            "alert_type",
            "alert_count"
        ]

        fig_alert = px.bar(
            alert_type_df,
            x="alert_type",
            y="alert_count",
            title="AML Alerts by Type",
            text="alert_count"
        )

        fig_alert.update_layout(
            xaxis_title="Alert Type",
            yaxis_title="Alerts"
        )

        st.plotly_chart(
            fig_alert,
            use_container_width=True
        )


# =============================================================================
# 15. INVESTIGATION QUEUE
# =============================================================================

st.subheader(
    "🧾 Analyst Investigation Queue"
)


queue_columns = [
    "case_id",
    "customer_id",
    "case_priority",
    "case_risk_level",
    "highest_risk_score",
    "alert_count",
    "primary_alert_type",
    "red_flag_count",
    "model_rule_agreement",
    "analyst_disposition",
    "investigation_status"
]


queue_columns = [
    column
    for column in queue_columns
    if column in filtered_cases.columns
]


queue_display = (
    filtered_cases[
        queue_columns
    ]
    .sort_values(
        [
            column
            for column in [
                "case_priority",
                "highest_risk_score"
            ]
            if column in filtered_cases.columns
        ],
        ascending=[
            True,
            False
        ]
    )
)


st.dataframe(
    queue_display,
    use_container_width=True,
    hide_index=True
)


# =============================================================================
# 16. CASE SELECTION
# =============================================================================

st.subheader(
    "🔎 Select Case for Investigation"
)


case_options = (
    filtered_cases[
        "case_id"
    ]
    .astype(str)
    .tolist()
    if "case_id" in filtered_cases.columns
    else []
)


if len(case_options) == 0:

    st.info(
        "No cases match the selected filters."
    )

    st.stop()


selected_case_id = st.selectbox(
    "Investigation Case",
    case_options
)


selected_case = filtered_cases[
    filtered_cases[
        "case_id"
    ]
    .astype(str)
    ==
    str(selected_case_id)
].iloc[0]


customer_id = str(
    selected_case[
        "customer_id"
    ]
)


# =============================================================================
# 17. CASE SUMMARY
# =============================================================================

st.subheader(
    "📋 Case Summary"
)


summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)


with summary_col1:

    st.metric(
        "Case Priority",
        str(
            selected_case.get(
                "case_priority",
                ""
            )
        )
    )


with summary_col2:

    st.metric(
        "Risk Level",
        str(
            selected_case.get(
                "case_risk_level",
                ""
            )
        )
    )


with summary_col3:

    st.metric(
        "Risk Score",
        "{:.2f}".format(
            float(
                selected_case.get(
                    "highest_risk_score",
                    0
                )
            )
        )
    )


with summary_col4:

    st.metric(
        "Alert Count",
        integer(
            selected_case.get(
                "alert_count",
                0
            )
        )
    )


# =============================================================================
# 18. CASE DETAILS
# =============================================================================

detail_col1, detail_col2 = st.columns(2)


with detail_col1:

    st.markdown(
        "### 🚨 Alert Information"
    )

    st.write(
        "**Primary Alert Type:**",
        selected_case.get(
            "primary_alert_type",
            ""
        )
    )

    st.write(
        "**Alert Reason:**",
        selected_case.get(
            "primary_alert_reason",
            ""
        )
    )

    st.write(
        "**Model / Rule Agreement:**",
        selected_case.get(
            "model_rule_agreement",
            ""
        )
    )

    st.write(
        "**Preliminary Disposition:**",
        selected_case.get(
            "preliminary_disposition",
            ""
        )
    )


with detail_col2:

    st.markdown(
        "### 💰 Transaction Exposure"
    )

    st.write(
        "**Total Transaction Volume:**",
        money(
            selected_case.get(
                "total_transaction_volume_usd",
                0
            )
        )
    )

    st.write(
        "**Alert Transaction Volume:**",
        money(
            selected_case.get(
                "alert_transaction_volume_usd",
                0
            )
        )
    )

    st.write(
        "**Expected Monthly Volume:**",
        money(
            selected_case.get(
                "expected_monthly_volume_usd",
                0
            )
        )
    )

    st.write(
        "**Annual Income:**",
        money(
            selected_case.get(
                "annual_income_usd",
                0
            )
        )
    )


# =============================================================================
# 19. CUSTOMER PROFILE
# =============================================================================

st.subheader(
    "👤 Customer Investigation Profile"
)


profile_match = pd.DataFrame()


if not customer_profiles.empty:

    profile_match = customer_profiles[
        customer_profiles[
            "customer_id"
        ]
        .astype(str)
        ==
        str(customer_id)
    ]


if not profile_match.empty:

    profile = profile_match.iloc[0]

else:

    profile = selected_case


profile_col1, profile_col2, profile_col3 = st.columns(3)


with profile_col1:

    st.markdown(
        "### Customer Information"
    )

    st.write(
        "**Customer ID:**",
        customer_id
    )

    st.write(
        "**Customer Type:**",
        profile.get(
            "customer_type",
            ""
        )
    )

    st.write(
        "**State:**",
        profile.get(
            "state",
            ""
        )
    )

    st.write(
        "**Occupation:**",
        profile.get(
            "occupation",
            ""
        )
    )

    st.write(
        "**Risk Rating:**",
        profile.get(
            "risk_rating",
            ""
        )
    )


with profile_col2:

    st.markdown(
        "### Activity Profile"
    )

    st.write(
        "**Total Transactions:**",
        integer(
            profile.get(
                "total_transactions",
                0
            )
        )
    )

    st.write(
        "**Total Volume:**",
        money(
            profile.get(
                "total_transaction_volume_usd",
                0
            )
        )
    )

    st.write(
        "**Average Transaction:**",
        money(
            profile.get(
                "average_transaction_usd",
                0
            )
        )
    )

    st.write(
        "**Maximum Transaction:**",
        money(
            profile.get(
                "maximum_transaction_usd",
                0
            )
        )
    )

    st.write(
        "**Unique Counterparties:**",
        integer(
            profile.get(
                "unique_counterparties",
                0
            )
        )
    )


with profile_col3:

    st.markdown(
        "### Risk Indicators"
    )

    st.write(
        "**PEP Indicator:**",
        profile.get(
            "pep_flag",
            0
        )
    )

    st.write(
        "**Sanctions Indicator:**",
        profile.get(
            "sanctions_flag",
            0
        )
    )

    st.write(
        "**Adverse Media:**",
        profile.get(
            "adverse_media_flag",
            0
        )
    )

    st.write(
        "**Alert Count:**",
        integer(
            profile.get(
                "alert_count",
                0
            )
        )
    )

    st.write(
        "**Maximum Risk Score:**",
        "{:.2f}".format(
            float(
                profile.get(
                    "maximum_risk_score",
                    0
                )
            )
        )
    )


# =============================================================================
# 20. RED FLAGS
# =============================================================================

st.subheader(
    "🚩 Investigation Red Flags"
)


customer_red_flags = pd.DataFrame()


if not red_flags.empty:

    customer_red_flags = red_flags[
        red_flags[
            "customer_id"
        ]
        .astype(str)
        ==
        str(customer_id)
    ]


if customer_red_flags.empty:

    st.success(
        "No red flags identified by the current synthetic rule set."
    )

else:

    st.dataframe(
        customer_red_flags,
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# 21. SHAP DRIVERS
# =============================================================================

st.subheader(
    "🤖 Machine Learning Explainability — SHAP"
)


shap_driver_text = selected_case.get(
    "top_shap_drivers",
    ""
)


if pd.isna(
    shap_driver_text
):

    shap_driver_text = ""


if str(
    shap_driver_text
).strip():

    drivers = [
        item.strip()
        for item in str(
            shap_driver_text
        ).split("|")
        if item.strip()
    ]

    for index, driver in enumerate(
        drivers[:10],
        start=1
    ):

        st.write(
            f"**{index}.** {driver}"
        )

else:

    st.info(
        "No SHAP transaction drivers are available for this case."
    )


# =============================================================================
# 22. GLOBAL SHAP FEATURE IMPORTANCE
# =============================================================================

if not shap_features.empty:

    st.markdown(
        "### Global Model Drivers"
    )

    shap_feature_column = get_first_existing_column(
        shap_features,
        [
            "feature",
            "feature_name"
        ]
    )

    shap_importance_column = get_first_existing_column(
        shap_features,
        [
            "mean_abs_shap",
            "mean_absolute_shap",
            "importance"
        ]
    )

    if (
        shap_feature_column
        and
        shap_importance_column
    ):

        shap_plot_df = (
            shap_features
            .copy()
            .sort_values(
                shap_importance_column,
                ascending=False
            )
            .head(15)
        )

        fig_shap = px.bar(
            shap_plot_df,
            x=shap_importance_column,
            y=shap_feature_column,
            orientation="h",
            title="Top Global SHAP Features"
        )

        fig_shap.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig_shap,
            use_container_width=True
        )


# =============================================================================
# 23. CUSTOMER TRANSACTION ANALYSIS
# =============================================================================

st.subheader(
    "💳 Customer Transaction Activity"
)


customer_transactions = pd.DataFrame()


if not alerts.empty:

    customer_transactions = alerts[
        alerts[
            "customer_id"
        ]
        .astype(str)
        ==
        str(customer_id)
    ].copy()


if customer_transactions.empty:

    st.info(
        "No alert transactions available for this customer."
    )

else:

    amount_column = get_first_existing_column(
        customer_transactions,
        [
            "amount_usd"
        ]
    )


    if (
        amount_column
        and
        "timestamp" in customer_transactions.columns
    ):

        transaction_chart = (
            customer_transactions[
                [
                    "timestamp",
                    amount_column
                ]
            ]
            .dropna()
            .sort_values(
                "timestamp"
            )
        )


        fig_timeline = px.line(
            transaction_chart,
            x="timestamp",
            y=amount_column,
            markers=True,
            title="Customer Alert Transaction Timeline"
        )


        fig_timeline.update_layout(
            xaxis_title="Transaction Time",
            yaxis_title="Transaction Amount (USD)"
        )


        st.plotly_chart(
            fig_timeline,
            use_container_width=True
        )


# =============================================================================
# 24. TRANSACTION DETAILS
# =============================================================================

if not customer_transactions.empty:

    st.markdown(
        "### Transaction Details"
    )

    transaction_columns = [
        "transaction_id",
        "timestamp",
        "amount_usd",
        "transaction_type",
        "channel",
        "origin_country",
        "destination_country",
        "is_international",
        "is_cash",
        "alert_type",
        "final_aml_risk_score"
    ]


    transaction_columns = [
        column
        for column in transaction_columns
        if column in customer_transactions.columns
    ]


    transaction_display = (
        customer_transactions[
            transaction_columns
        ]
        .sort_values(
            "timestamp",
            ascending=False
        )
        .head(100)
    )


    st.dataframe(
        transaction_display,
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# 25. TRANSACTION STATISTICS
# =============================================================================

if not customer_transactions.empty:

    transaction_stat_col1, transaction_stat_col2, transaction_stat_col3, transaction_stat_col4 = st.columns(4)


    amount_values = safe_numeric(
        customer_transactions[
            "amount_usd"
        ]
    )


    with transaction_stat_col1:

        st.metric(
            "Alert Transactions",
            integer(
                len(
                    customer_transactions
                )
            )
        )


    with transaction_stat_col2:

        st.metric(
            "Alert Volume",
            money(
                amount_values.sum()
            )
        )


    with transaction_stat_col3:

        st.metric(
            "Average Alert",
            money(
                amount_values.mean()
            )
        )


    with transaction_stat_col4:

        st.metric(
            "Largest Alert",
            money(
                amount_values.max()
            )
        )


# =============================================================================
# 26. GEOGRAPHIC ACTIVITY
# =============================================================================

if not customer_transactions.empty:

    geographic_columns = [
        column
        for column in [
            "origin_country",
            "destination_country"
        ]
        if column in customer_transactions.columns
    ]


    if geographic_columns:

        st.subheader(
            "🌎 Geographic Activity"
        )


        geo_frames = []


        if "origin_country" in customer_transactions.columns:

            origin_df = (
                customer_transactions[
                    "origin_country"
                ]
                .astype(str)
                .value_counts()
                .reset_index()
            )

            origin_df.columns = [
                "country",
                "count"
            ]

            origin_df[
                "activity_type"
            ] = "Origin"


            geo_frames.append(
                origin_df
            )


        if "destination_country" in customer_transactions.columns:

            destination_df = (
                customer_transactions[
                    "destination_country"
                ]
                .astype(str)
                .value_counts()
                .reset_index()
            )

            destination_df.columns = [
                "country",
                "count"
            ]

            destination_df[
                "activity_type"
            ] = "Destination"


            geo_frames.append(
                destination_df
            )


        if geo_frames:

            geo_df = pd.concat(
                geo_frames,
                ignore_index=True
            )


            fig_geo = px.bar(
                geo_df,
                x="country",
                y="count",
                color="activity_type",
                barmode="group",
                title="Transaction Activity by Country"
            )


            st.plotly_chart(
                fig_geo,
                use_container_width=True
            )


# =============================================================================
# 27. CASE MANAGEMENT
# =============================================================================

st.subheader(
    "📝 Analyst Case Management"
)


current_disposition = selected_case.get(
    "analyst_disposition",
    "PENDING_REVIEW"
)


if pd.isna(
    current_disposition
):

    current_disposition = "PENDING_REVIEW"


disposition_options = [
    "PENDING_REVIEW",
    "FALSE_POSITIVE",
    "CLEARED",
    "MONITOR",
    "INVESTIGATE_FURTHER",
    "ESCALATE"
]


if current_disposition not in disposition_options:

    current_disposition = "PENDING_REVIEW"


new_disposition = st.selectbox(
    "Analyst Disposition",
    disposition_options,
    index=disposition_options.index(
        current_disposition
    )
)


current_status = selected_case.get(
    "investigation_status",
    "OPEN"
)


if pd.isna(
    current_status
):

    current_status = "OPEN"


status_options = [
    "OPEN",
    "IN_PROGRESS",
    "CLOSED"
]


if current_status not in status_options:

    current_status = "OPEN"


new_status = st.selectbox(
    "Investigation Status",
    status_options,
    index=status_options.index(
        current_status
    )
)


current_notes = selected_case.get(
    "analyst_notes",
    ""
)


if pd.isna(
    current_notes
):

    current_notes = ""


new_notes = st.text_area(
    "Analyst Investigation Notes",
    value=str(
        current_notes
    ),
    height=150,
    placeholder=(
        "Document investigation observations, "
        "customer context, transaction rationale, "
        "supporting evidence and escalation reasoning."
    )
)


escalation_options = [
    "NO",
    "YES"
]


current_escalation = selected_case.get(
    "escalation_required",
    "NO"
)


if pd.isna(
    current_escalation
):

    current_escalation = "NO"


if current_escalation not in escalation_options:

    current_escalation = "NO"


new_escalation = st.selectbox(
    "Escalation Required",
    escalation_options,
    index=escalation_options.index(
        str(
            current_escalation
        )
    )
)


sar_options = [
    "PENDING_HUMAN_REVIEW",
    "NO_SAR_CONSIDERATION",
    "SAR_CONSIDERATION"
]


current_sar = selected_case.get(
    "sar_consideration",
    "PENDING_HUMAN_REVIEW"
)


if pd.isna(
    current_sar
):

    current_sar = "PENDING_HUMAN_REVIEW"


if current_sar not in sar_options:

    current_sar = "PENDING_HUMAN_REVIEW"


new_sar = st.selectbox(
    "SAR Consideration",
    sar_options,
    index=sar_options.index(
        str(
            current_sar
        )
    )
)


# =============================================================================
# 28. SAVE CASE UPDATE
# =============================================================================

if st.button(
    "💾 Save Analyst Update",
    type="primary"
):

    update_row = pd.DataFrame(
        [
            {
                "case_id":
                    selected_case_id,

                "analyst_disposition":
                    new_disposition,

                "investigation_status":
                    new_status,

                "analyst_notes":
                    new_notes,

                "escalation_required":
                    new_escalation,

                "sar_consideration":
                    new_sar,

                "updated_at":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
            }
        ]
    )


    if os.path.exists(
        UPDATE_FILE
    ):

        existing_updates = pd.read_csv(
            UPDATE_FILE
        )

    else:

        existing_updates = pd.DataFrame()


    if not existing_updates.empty:

        existing_updates = existing_updates[
            existing_updates[
                "case_id"
            ]
            .astype(str)
            !=
            str(selected_case_id)
        ]


    updated_data = pd.concat(
        [
            existing_updates,
            update_row
        ],
        ignore_index=True
    )


    updated_data.to_csv(
        UPDATE_FILE,
        index=False
    )


    st.success(
        "Analyst case update saved successfully."
    )


    st.cache_data.clear()


# =============================================================================
# 29. CASE DOWNLOAD
# =============================================================================

st.subheader(
    "⬇️ Investigation Data Export"
)


case_csv = cases.to_csv(
    index=False
).encode(
    "utf-8"
)


st.download_button(
    label="Download All Investigation Cases",
    data=case_csv,
    file_name="aml_investigation_cases.csv",
    mime="text/csv"
)


if not customer_transactions.empty:

    transaction_csv = (
        customer_transactions
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(
        label="Download Selected Customer Transactions",
        data=transaction_csv,
        file_name=(
            f"customer_{customer_id}_transactions.csv"
        ),
        mime="text/csv"
    )


# =============================================================================
# 30. METHODOLOGY
# =============================================================================

with st.expander(
    "ℹ️ AML Investigation Methodology"
):

    st.markdown(
        """
        ### Investigation Workflow

        **1. Alert Generation**

        Transaction activity is evaluated using synthetic AML rules
        and machine-learning risk scoring.

        **2. Risk Prioritization**

        Alerts are converted into investigation cases and prioritized
        using risk scores and alert priority.

        **3. Customer Profile Review**

        Analysts review customer type, occupation, income,
        expected activity, risk rating and relevant indicators.

        **4. Transaction Review**

        Analysts review transaction amount, frequency, timing,
        counterparties, cash activity and geographic activity.

        **5. Red-Flag Review**

        Identified behavioral indicators are reviewed in context.

        **6. Model Explainability**

        SHAP features provide an explanation of model-driven
        risk indicators.

        **7. Human Investigation**

        The analyst determines whether the observed activity
        has a reasonable explanation and documents the investigation.

        **8. Disposition**

        The analyst can clear, monitor, investigate further,
        escalate or classify an alert as a false positive.

        **9. SAR Consideration**

        SAR consideration is kept as a human-review workflow.
        The machine-learning model does not automatically determine
        whether a SAR should be filed.
        """
    )


# =============================================================================
# 31. FOOTER
# =============================================================================

st.markdown(
    "---"
)


st.caption(
    "AML Transaction Monitoring Portfolio Project | "
    "Synthetic Data | Human-in-the-Loop Investigation | "
    "Rules + Machine Learning + SHAP"
)


st.caption(
    "Dashboard generated for educational and portfolio demonstration purposes."
)