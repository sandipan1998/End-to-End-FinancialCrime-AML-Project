"""
PHASE 12 - GRAPH-BASED AML NETWORK ANALYTICS

Project:
US FINANCIAL CRIME AI PROJECT

Purpose
-------
Build a transaction network from the synthetic US banking dataset and identify
network-level AML risk indicators such as:

- Hub accounts
- High-degree accounts
- Shared counterparties
- Inbound / outbound concentration
- Circular money movement
- Connected transaction networks
- Network centrality
- Account-level network risk
- Combined AML + network risk

IMPORTANT
---------
This is a synthetic AML portfolio project.

Network detection does NOT use the following fields to CREATE network risk:

- known_suspicious
- scenario
- target

Those fields may be used only for validation / comparison after the
network analysis has been completed.

This project does NOT determine that a customer committed a crime.

Network scores are risk indicators only.

SAR decisions require appropriate human investigation and institution-specific
procedures. This script does not automatically file SARs.
"""

import os
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

RAW_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "US_AML_Synthetic_Dataset.xlsx"
)

RISK_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "aml_alerts_scored.csv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "network"
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)


def risk_level(score):
    """
    Convert network risk score into portfolio risk levels.
    These thresholds are synthetic project thresholds.
    """

    if score >= 70:
        return "CRITICAL"

    elif score >= 50:
        return "HIGH"

    elif score >= 25:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# 3. START
# ============================================================

print_header(
    "PHASE 12 - GRAPH-BASED AML NETWORK ANALYTICS"
)

print("\nProject directory:")
print(BASE_DIR)


# ============================================================
# 4. VALIDATE INPUT FILE
# ============================================================

print_header(
    "4. INPUT FILE VALIDATION"
)

if not os.path.exists(RAW_FILE):

    raise FileNotFoundError(
        "\nRaw transaction dataset not found:\n"
        + RAW_FILE
    )

print("[OK] Raw transaction dataset found")

if os.path.exists(RISK_FILE):

    print("[OK] AML risk scoring file found")

else:

    print(
        "[INFO] AML risk scoring file was not found."
    )

    print(
        "[INFO] Network analysis will continue "
        "without AML risk integration."
    )


# ============================================================
# 5. LOAD TRANSACTION DATA
# ============================================================

print_header(
    "5. LOADING TRANSACTION DATA"
)

transactions = pd.read_excel(
    RAW_FILE,
    sheet_name="Transactions"
)

print(
    f"Transactions loaded: {len(transactions):,}"
)


# ============================================================
# 6. REQUIRED COLUMN VALIDATION
# ============================================================

required_columns = [
    "transaction_id",
    "timestamp",
    "sender_account",
    "receiver_account",
    "amount_usd",
    "customer_id"
]

missing_columns = [
    column
    for column in required_columns
    if column not in transactions.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required transaction columns:\n"
        + str(missing_columns)
    )

print("[OK] Required transaction columns found")


# ============================================================
# 7. DATA CLEANING
# ============================================================

print_header(
    "7. DATA CLEANING"
)

transactions["timestamp"] = pd.to_datetime(
    transactions["timestamp"],
    errors="coerce"
)

transactions["amount_usd"] = safe_numeric(
    transactions["amount_usd"]
)

transactions["sender_account"] = (
    transactions["sender_account"]
    .astype(str)
    .str.strip()
)

transactions["receiver_account"] = (
    transactions["receiver_account"]
    .astype(str)
    .str.strip()
)

transactions["customer_id"] = (
    transactions["customer_id"]
    .astype(str)
    .str.strip()
)

transactions = transactions.dropna(
    subset=[
        "sender_account",
        "receiver_account"
    ]
)

transactions = transactions[
    transactions["sender_account"]
    != transactions["receiver_account"]
].copy()

print(
    f"Clean transaction records: "
    f"{len(transactions):,}"
)


# ============================================================
# 8. CREATE DIRECTED TRANSACTION GRAPH
# ============================================================

print_header(
    "8. BUILDING TRANSACTION NETWORK"
)

G = nx.DiGraph()

for row in transactions[
    [
        "sender_account",
        "receiver_account",
        "amount_usd"
    ]
].itertuples(index=False):

    sender = row.sender_account

    receiver = row.receiver_account

    amount = float(row.amount_usd)

    if G.has_edge(
        sender,
        receiver
    ):

        G[sender][receiver][
            "transaction_count"
        ] += 1

        G[sender][receiver][
            "total_amount_usd"
        ] += amount

    else:

        G.add_edge(
            sender,
            receiver,
            transaction_count=1,
            total_amount_usd=amount
        )


print(
    f"Network nodes/accounts: "
    f"{G.number_of_nodes():,}"
)

print(
    f"Network relationships/edges: "
    f"{G.number_of_edges():,}"
)


# ============================================================
# 9. ACCOUNT NETWORK METRICS
# ============================================================

print_header(
    "9. ACCOUNT NETWORK METRICS"
)

network_records = []

for account in G.nodes():

    inbound_edges = list(
        G.in_edges(account)
    )

    outbound_edges = list(
        G.out_edges(account)
    )

    in_degree = len(
        inbound_edges
    )

    out_degree = len(
        outbound_edges
    )

    inbound_amount = 0.0

    for source, target in inbound_edges:

        inbound_amount += (
            G[source][target]
            .get(
                "total_amount_usd",
                0
            )
        )

    outbound_amount = 0.0

    for source, target in outbound_edges:

        outbound_amount += (
            G[source][target]
            .get(
                "total_amount_usd",
                0
            )
        )

    total_degree = (
        in_degree +
        out_degree
    )

    total_network_amount = (
        inbound_amount +
        outbound_amount
    )

    network_records.append(
        {
            "account_id": account,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": total_degree,
            "inbound_amount_usd": inbound_amount,
            "outbound_amount_usd": outbound_amount,
            "total_network_amount_usd": total_network_amount
        }
    )


network_metrics = pd.DataFrame(
    network_records
)

print(
    f"Account metric records: "
    f"{len(network_metrics):,}"
)


# ============================================================
# 10. DEGREE CENTRALITY
# ============================================================

print_header(
    "10. CENTRALITY ANALYSIS"
)

degree_centrality = nx.degree_centrality(
    G
)

network_metrics[
    "degree_centrality"
] = network_metrics[
    "account_id"
].map(
    degree_centrality
).fillna(0)


# ============================================================
# 11. BETWEENNESS CENTRALITY
# ============================================================

print(
    "Calculating betweenness centrality..."
)

# For a large transaction graph, exact betweenness can be
# expensive. We use sampling when the graph is large.

node_count = G.number_of_nodes()

if node_count <= 2000:

    betweenness = nx.betweenness_centrality(
        G,
        normalized=True
    )

else:

    sample_size = min(
        1000,
        node_count
    )

    sampled_nodes = list(
        G.nodes()
    )[:sample_size]

    betweenness = nx.betweenness_centrality(
        G,
        k=sample_size,
        normalized=True,
        seed=42
    )


network_metrics[
    "betweenness_centrality"
] = network_metrics[
    "account_id"
].map(
    betweenness
).fillna(0)


# ============================================================
# 12. INBOUND / OUTBOUND FLOW FEATURES
# ============================================================

print_header(
    "12. FLOW ANALYSIS"
)

network_metrics[
    "in_out_amount_ratio"
] = (
    network_metrics[
        "inbound_amount_usd"
    ]
    /
    network_metrics[
        "outbound_amount_usd"
    ].replace(
        0,
        np.nan
    )
)

network_metrics[
    "in_out_amount_ratio"
] = (
    network_metrics[
        "in_out_amount_ratio"
    ]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ============================================================
# 13. NETWORK HUB DETECTION
# ============================================================

degree_threshold = network_metrics[
    "total_degree"
].quantile(
    0.99
)

network_metrics[
    "network_hub_indicator"
] = (
    network_metrics[
        "total_degree"
    ]
    >= degree_threshold
).astype(int)


# ============================================================
# 14. HIGH INBOUND / OUTBOUND DETECTION
# ============================================================

inbound_threshold = network_metrics[
    "inbound_amount_usd"
].quantile(
    0.99
)

outbound_threshold = network_metrics[
    "outbound_amount_usd"
].quantile(
    0.99
)


network_metrics[
    "high_inbound_indicator"
] = (
    network_metrics[
        "inbound_amount_usd"
    ]
    >= inbound_threshold
).astype(int)


network_metrics[
    "high_outbound_indicator"
] = (
    network_metrics[
        "outbound_amount_usd"
    ]
    >= outbound_threshold
).astype(int)


# ============================================================
# 15. BALANCED FLOW DETECTION
# ============================================================

network_metrics[
    "balanced_flow_indicator"
] = (
    (
        network_metrics[
            "inbound_amount_usd"
        ] > 0
    )
    &
    (
        network_metrics[
            "outbound_amount_usd"
        ] > 0
    )
    &
    (
        network_metrics[
            "in_out_amount_ratio"
        ].between(
            0.5,
            2.0
        )
    )
).astype(int)


# ============================================================
# 16. SHARED COUNTERPARTY ANALYSIS
# ============================================================

print_header(
    "16. SHARED COUNTERPARTY ANALYSIS"
)

counterparty_map = {}

for account in G.nodes():

    inbound_counterparties = set(
        G.predecessors(account)
    )

    outbound_counterparties = set(
        G.successors(account)
    )

    counterparty_map[account] = (
        inbound_counterparties
        |
        outbound_counterparties
    )


# Instead of comparing every account with every other account,
# construct a reverse index:
#
# counterparty -> accounts connected to that counterparty

reverse_counterparty_map = {}

for account, counterparties in counterparty_map.items():

    for counterparty in counterparties:

        if counterparty not in reverse_counterparty_map:

            reverse_counterparty_map[
                counterparty
            ] = set()

        reverse_counterparty_map[
            counterparty
        ].add(account)


shared_counterparty_counts = {}

for account in G.nodes():

    connected_accounts = set()

    for counterparty in counterparty_map.get(
        account,
        set()
    ):

        related_accounts = (
            reverse_counterparty_map
            .get(
                counterparty,
                set()
            )
        )

        connected_accounts.update(
            related_accounts
        )

    connected_accounts.discard(
        account
    )

    shared_counterparty_counts[
        account
    ] = len(
        connected_accounts
    )


network_metrics[
    "shared_counterparty_networks"
] = (
    network_metrics[
        "account_id"
    ]
    .map(
        shared_counterparty_counts
    )
    .fillna(0)
)


network_metrics[
    "shared_counterparty_indicator"
] = (
    network_metrics[
        "shared_counterparty_networks"
    ]
    >= 2
).astype(int)


# ============================================================
# 17. CIRCULAR FLOW DETECTION
# ============================================================

print_header(
    "17. CIRCULAR FLOW DETECTION"
)

print(
    "Searching for transaction cycles of length 2-4..."
)

unique_cycles = set()

try:

    cycles_found = nx.simple_cycles(
        G,
        length_bound=4
    )

    for cycle in cycles_found:

        cycle_length = len(cycle)

        if cycle_length < 2:
            continue

        if cycle_length > 4:
            continue

        # Canonical representation removes duplicate
        # rotations of the same cycle.

        cycle_tuple = tuple(cycle)

        rotations = []

        for i in range(
            len(cycle_tuple)
        ):

            rotated = (
                cycle_tuple[i:]
                +
                cycle_tuple[:i]
            )

            rotations.append(
                rotated
            )

        canonical_cycle = min(
            rotations
        )

        unique_cycles.add(
            canonical_cycle
        )

except Exception as exc:

    print(
        "[WARNING] Circular flow detection "
        "encountered an issue:"
    )

    print(
        str(exc)
    )


print(
    f"Unique circular flow patterns: "
    f"{len(unique_cycles):,}"
)


cycle_accounts = set()

for cycle in unique_cycles:

    for account in cycle:

        cycle_accounts.add(
            account
        )


network_metrics[
    "circular_flow_indicator"
] = (
    network_metrics[
        "account_id"
    ]
    .isin(
        cycle_accounts
    )
    .astype(int)
)


# ============================================================
# 18. NETWORK RISK SCORE
# ============================================================

print_header(
    "18. NETWORK RISK SCORING"
)

network_metrics[
    "network_risk_score"
] = 0.0


# Synthetic portfolio weights.
#
# These are NOT regulatory thresholds.

network_metrics[
    "network_risk_score"
] += (
    network_metrics[
        "network_hub_indicator"
    ]
    * 20
)


network_metrics[
    "network_risk_score"
] += (
    network_metrics[
        "high_inbound_indicator"
    ]
    * 10
)


network_metrics[
    "network_risk_score"
] += (
    network_metrics[
        "high_outbound_indicator"
    ]
    * 10
)


network_metrics[
    "network_risk_score"
] += (
    network_metrics[
        "balanced_flow_indicator"
    ]
    * 10
)


network_metrics[
    "network_risk_score"
] += (
    network_metrics[
        "shared_counterparty_indicator"
    ]
    * 20
)


network_metrics[
    "network_risk_score"
] += (
    network_metrics[
        "circular_flow_indicator"
    ]
    * 30
)


network_metrics[
    "network_risk_score"
] = (
    network_metrics[
        "network_risk_score"
    ]
    .clip(
        0,
        100
    )
)


network_metrics[
    "network_risk_level"
] = (
    network_metrics[
        "network_risk_score"
    ]
    .apply(
        risk_level
    )
)


# ============================================================
# 19. NETWORK RED FLAGS
# ============================================================

print_header(
    "19. NETWORK RED FLAGS"
)


def identify_network_flags(row):

    flags = []

    if row[
        "network_hub_indicator"
    ] == 1:

        flags.append(
            "NETWORK_HUB"
        )

    if row[
        "high_inbound_indicator"
    ] == 1:

        flags.append(
            "HIGH_INBOUND_VOLUME"
        )

    if row[
        "high_outbound_indicator"
    ] == 1:

        flags.append(
            "HIGH_OUTBOUND_VOLUME"
        )

    if row[
        "balanced_flow_indicator"
    ] == 1:

        flags.append(
            "BALANCED_INBOUND_OUTBOUND_FLOW"
        )

    if row[
        "shared_counterparty_indicator"
    ] == 1:

        flags.append(
            "SHARED_COUNTERPARTY_NETWORK"
        )

    if row[
        "circular_flow_indicator"
    ] == 1:

        flags.append(
            "CIRCULAR_FLOW"
        )

    return "|".join(flags)


network_metrics[
    "network_red_flags"
] = network_metrics.apply(
    identify_network_flags,
    axis=1
)


network_metrics[
    "network_red_flag_count"
] = (
    network_metrics[
        "network_red_flags"
    ]
    .apply(
        lambda x:
        0
        if x == ""
        else len(
            x.split("|")
        )
    )
)


# ============================================================
# 20. CUSTOMER / ACCOUNT MAPPING
# ============================================================

print_header(
    "20. CUSTOMER ACCOUNT MAPPING"
)

account_customer_lookup = (
    transactions[
        [
            "customer_id",
            "sender_account"
        ]
    ]
    .drop_duplicates()
    .rename(
        columns={
            "sender_account": "account_id"
        }
    )
)


network_metrics = network_metrics.merge(
    account_customer_lookup,
    on="account_id",
    how="left"
)


# ============================================================
# 21. AML RISK INTEGRATION
# ============================================================

print_header(
    "21. AML MODEL RISK INTEGRATION"
)

network_metrics[
    "max_aml_risk_score"
] = 0.0

network_metrics[
    "aml_alert_count"
] = 0


if os.path.exists(
    RISK_FILE
):

    try:

        risk_df = pd.read_csv(
            RISK_FILE
        )

        print(
            f"AML risk records loaded: "
            f"{len(risk_df):,}"
        )

        if (
            "customer_id" in risk_df.columns
            and
            "final_aml_risk_score"
            in risk_df.columns
        ):

            risk_df[
                "customer_id"
            ] = (
                risk_df[
                    "customer_id"
                ]
                .astype(str)
                .str.strip()
            )

            risk_df[
                "final_aml_risk_score"
            ] = safe_numeric(
                risk_df[
                    "final_aml_risk_score"
                ]
            )

            customer_risk = (
                risk_df
                .groupby(
                    "customer_id",
                    as_index=False
                )
                .agg(
                    max_aml_risk_score=(
                        "final_aml_risk_score",
                        "max"
                    ),
                    aml_alert_count=(
                        "customer_id",
                        "size"
                    )
                )
            )

            network_metrics = network_metrics.merge(
                customer_risk,
                on="customer_id",
                how="left",
                suffixes=(
                    "",
                    "_risk"
                )
            )

            if (
                "max_aml_risk_score_risk"
                in network_metrics.columns
            ):

                network_metrics[
                    "max_aml_risk_score"
                ] = (
                    network_metrics[
                        "max_aml_risk_score_risk"
                    ]
                    .fillna(0)
                )

                network_metrics = (
                    network_metrics
                    .drop(
                        columns=[
                            "max_aml_risk_score_risk"
                        ]
                    )
                )

            if (
                "aml_alert_count_risk"
                in network_metrics.columns
            ):

                network_metrics[
                    "aml_alert_count"
                ] = (
                    network_metrics[
                        "aml_alert_count_risk"
                    ]
                    .fillna(0)
                )

                network_metrics = (
                    network_metrics
                    .drop(
                        columns=[
                            "aml_alert_count_risk"
                        ]
                    )
                )

            print(
                "[OK] Customer AML risk merged."
            )

        else:

            print(
                "[INFO] Required AML risk columns "
                "were not available."
            )

    except Exception as exc:

        print(
            "[WARNING] AML risk integration "
            "could not be completed:"
        )

        print(
            str(exc)
        )


# ============================================================
# 22. COMBINED NETWORK + AML RISK
# ============================================================

print_header(
    "22. COMBINED NETWORK + AML RISK"
)

network_metrics[
    "combined_network_aml_score"
] = (
    (
        network_metrics[
            "network_risk_score"
        ]
        * 0.60
    )
    +
    (
        network_metrics[
            "max_aml_risk_score"
        ]
        * 0.40
    )
).clip(
    0,
    100
)


network_metrics[
    "combined_risk_level"
] = (
    network_metrics[
        "combined_network_aml_score"
    ]
    .apply(
        risk_level
    )
)


# ============================================================
# 23. SORT RESULTS
# ============================================================

network_metrics = (
    network_metrics
    .sort_values(
        "combined_network_aml_score",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 24. SAVE ACCOUNT NETWORK METRICS
# ============================================================

print_header(
    "24. SAVING NETWORK METRICS"
)

network_metrics_file = os.path.join(
    PROCESSED_DIR,
    "aml_network_account_metrics.csv"
)

network_metrics.to_csv(
    network_metrics_file,
    index=False
)

print(
    "[OK] Account network metrics saved:"
)

print(
    network_metrics_file
)


# ============================================================
# 25. TOP 100 NETWORK RISK ACCOUNTS
# ============================================================

top_network_accounts = (
    network_metrics
    .head(100)
    .copy()
)

top_network_file = os.path.join(
    OUTPUT_DIR,
    "top_100_network_risk_accounts.csv"
)

top_network_accounts.to_csv(
    top_network_file,
    index=False
)

print(
    "[OK] Top 100 network-risk accounts saved:"
)

print(
    top_network_file
)


# ============================================================
# 26. NETWORK RISK SUMMARY
# ============================================================

network_risk_summary = (
    network_metrics
    .groupby(
        "network_risk_level"
    )
    .size()
    .reset_index(
        name="account_count"
    )
)


network_risk_summary_file = os.path.join(
    OUTPUT_DIR,
    "network_risk_summary.csv"
)

network_risk_summary.to_csv(
    network_risk_summary_file,
    index=False
)


# ============================================================
# 27. NETWORK RED FLAG SUMMARY
# ============================================================

red_flag_summary = pd.DataFrame(
    {
        "network_red_flag": [
            "NETWORK_HUB",
            "HIGH_INBOUND_VOLUME",
            "HIGH_OUTBOUND_VOLUME",
            "BALANCED_INBOUND_OUTBOUND_FLOW",
            "SHARED_COUNTERPARTY_NETWORK",
            "CIRCULAR_FLOW"
        ],

        "account_count": [
            int(
                network_metrics[
                    "network_hub_indicator"
                ].sum()
            ),

            int(
                network_metrics[
                    "high_inbound_indicator"
                ].sum()
            ),

            int(
                network_metrics[
                    "high_outbound_indicator"
                ].sum()
            ),

            int(
                network_metrics[
                    "balanced_flow_indicator"
                ].sum()
            ),

            int(
                network_metrics[
                    "shared_counterparty_indicator"
                ].sum()
            ),

            int(
                network_metrics[
                    "circular_flow_indicator"
                ].sum()
            )
        ]
    }
)


red_flag_file = os.path.join(
    OUTPUT_DIR,
    "network_red_flag_summary.csv"
)

red_flag_summary.to_csv(
    red_flag_file,
    index=False
)


# ============================================================
# 28. CONNECTED COMPONENT ANALYSIS
# ============================================================

print_header(
    "28. CONNECTED NETWORK COMPONENT ANALYSIS"
)

undirected_graph = G.to_undirected()

components = list(
    nx.connected_components(
        undirected_graph
    )
)

component_records = []

for component_id, component in enumerate(
    components,
    start=1
):

    component_size = len(
        component
    )

    if component_size < 2:
        continue

    component_subgraph = (
        G.subgraph(
            component
        )
    )

    component_edges = (
        component_subgraph.number_of_edges()
    )

    component_amount = 0.0

    for source, target in (
        component_subgraph.edges()
    ):

        component_amount += (
            G[source][target]
            .get(
                "total_amount_usd",
                0
            )
        )

    component_records.append(
        {
            "network_component_id": component_id,
            "account_count": component_size,
            "edge_count": component_edges,
            "total_transaction_amount_usd":
                component_amount
        }
    )


component_df = pd.DataFrame(
    component_records
)


if not component_df.empty:

    component_df = (
        component_df
        .sort_values(
            [
                "account_count",
                "edge_count"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


component_file = os.path.join(
    OUTPUT_DIR,
    "network_components.csv"
)

component_df.to_csv(
    component_file,
    index=False
)


print(
    f"Connected components with 2+ accounts: "
    f"{len(component_df):,}"
)


# ============================================================
# 29. TOP NETWORK COMPONENTS
# ============================================================

if not component_df.empty:

    top_components = (
        component_df
        .head(50)
        .copy()
    )

else:

    top_components = component_df.copy()


top_components_file = os.path.join(
    OUTPUT_DIR,
    "top_50_network_components.csv"
)

top_components.to_csv(
    top_components_file,
    index=False
)


# ============================================================
# 30. NETWORK VISUALIZATION
# ============================================================

print_header(
    "30. CREATING NETWORK VISUALIZATION"
)

# Select the top 50 accounts based on combined network
# and AML risk.

top_accounts = set(
    network_metrics
    .head(50)[
        "account_id"
    ]
)


visual_nodes = set(
    top_accounts
)


# Add immediate neighbors.

for account in list(
    top_accounts
):

    visual_nodes.update(
        list(
            G.predecessors(
                account
            )
        )
    )

    visual_nodes.update(
        list(
            G.successors(
                account
            )
        )
    )


# Prevent excessively large visualization.

if len(visual_nodes) > 250:

    visual_nodes = set(
        list(
            visual_nodes
        )[:250]
    )


visual_graph = G.subgraph(
    visual_nodes
).copy()


network_plot_file = os.path.join(
    OUTPUT_DIR,
    "aml_transaction_network.png"
)


if (
    visual_graph.number_of_nodes()
    > 0
):

    plt.figure(
        figsize=(
            16,
            12
        )
    )

    positions = nx.spring_layout(
        visual_graph,
        seed=42,
        k=0.5
    )

    nx.draw_networkx_nodes(
        visual_graph,
        positions,
        node_size=80,
        alpha=0.75
    )

    nx.draw_networkx_edges(
        visual_graph,
        positions,
        arrows=True,
        alpha=0.25,
        arrowsize=8
    )

    plt.title(
        "AML Transaction Network - "
        "Top Network Risk Accounts"
    )

    plt.axis(
        "off"
    )

    plt.savefig(
        network_plot_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "[OK] Network visualization saved:"
    )

    print(
        network_plot_file
    )

else:

    print(
        "[INFO] No nodes available "
        "for visualization."
    )


# ============================================================
# 31. TOP NETWORK RISK ACCOUNTS
# ============================================================

print_header(
    "31. TOP NETWORK RISK ACCOUNTS"
)

display_columns = [
    "account_id",
    "customer_id",
    "in_degree",
    "out_degree",
    "total_degree",
    "inbound_amount_usd",
    "outbound_amount_usd",
    "degree_centrality",
    "betweenness_centrality",
    "shared_counterparty_networks",
    "network_risk_score",
    "network_risk_level",
    "network_red_flags",
    "max_aml_risk_score",
    "aml_alert_count",
    "combined_network_aml_score",
    "combined_risk_level"
]


display_columns = [
    column
    for column in display_columns
    if column in network_metrics.columns
]


print(
    network_metrics[
        display_columns
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 32. NETWORK SUMMARY VALUES
# ============================================================

print_header(
    "32. NETWORK ANALYTICS SUMMARY"
)

total_accounts = int(
    G.number_of_nodes()
)

total_relationships = int(
    G.number_of_edges()
)

network_hubs = int(
    network_metrics[
        "network_hub_indicator"
    ].sum()
)

high_inbound_accounts = int(
    network_metrics[
        "high_inbound_indicator"
    ].sum()
)

high_outbound_accounts = int(
    network_metrics[
        "high_outbound_indicator"
    ].sum()
)

shared_counterparty_accounts = int(
    network_metrics[
        "shared_counterparty_indicator"
    ].sum()
)

circular_flow_accounts = int(
    network_metrics[
        "circular_flow_indicator"
    ].sum()
)

critical_network_accounts = int(
    (
        network_metrics[
            "network_risk_level"
        ]
        == "CRITICAL"
    ).sum()
)

high_network_accounts = int(
    (
        network_metrics[
            "network_risk_level"
        ]
        == "HIGH"
    ).sum()
)


print(
    f"Total accounts in network: "
    f"{total_accounts:,}"
)

print(
    f"Total account relationships: "
    f"{total_relationships:,}"
)

print(
    f"Accounts flagged as network hubs: "
    f"{network_hubs:,}"
)

print(
    f"Accounts with high inbound volume: "
    f"{high_inbound_accounts:,}"
)

print(
    f"Accounts with high outbound volume: "
    f"{high_outbound_accounts:,}"
)

print(
    f"Accounts with shared-counterparty indicators: "
    f"{shared_counterparty_accounts:,}"
)

print(
    f"Accounts involved in circular flows: "
    f"{circular_flow_accounts:,}"
)

print(
    f"CRITICAL network-risk accounts: "
    f"{critical_network_accounts:,}"
)

print(
    f"HIGH network-risk accounts: "
    f"{high_network_accounts:,}"
)


# ============================================================
# 33. SAVE NETWORK SUMMARY TEXT
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "network_analytics_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PHASE 12 - GRAPH-BASED AML NETWORK ANALYTICS\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        f"Total accounts: {total_accounts:,}\n"
    )

    file.write(
        f"Total relationships: "
        f"{total_relationships:,}\n"
    )

    file.write(
        f"Network hubs: "
        f"{network_hubs:,}\n"
    )

    file.write(
        f"High inbound accounts: "
        f"{high_inbound_accounts:,}\n"
    )

    file.write(
        f"High outbound accounts: "
        f"{high_outbound_accounts:,}\n"
    )

    file.write(
        f"Shared-counterparty accounts: "
        f"{shared_counterparty_accounts:,}\n"
    )

    file.write(
        f"Circular-flow accounts: "
        f"{circular_flow_accounts:,}\n"
    )

    file.write(
        f"Critical network accounts: "
        f"{critical_network_accounts:,}\n"
    )

    file.write(
        f"High network accounts: "
        f"{high_network_accounts:,}\n"
    )


# ============================================================
# 34. GOVERNANCE NOTES
# ============================================================

governance_file = os.path.join(
    OUTPUT_DIR,
    "network_governance_notes.txt"
)

with open(
    governance_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PHASE 12 - AML NETWORK ANALYTICS "
        "GOVERNANCE NOTES\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        "1. Dataset\n\n"
    )

    file.write(
        "The project uses synthetic transaction data "
        "for demonstration and portfolio purposes.\n\n"
    )

    file.write(
        "2. Network analytics\n\n"
    )

    file.write(
        "Network indicators are analytical risk signals "
        "and are not determinations of criminal activity.\n\n"
    )

    file.write(
        "3. Ground truth\n\n"
    )

    file.write(
        "known_suspicious, scenario and target were not "
        "used to create network risk indicators.\n\n"
    )

    file.write(
        "4. Analyst review\n\n"
    )

    file.write(
        "Network risk should be reviewed together with "
        "KYC/CDD, transaction history, expected customer "
        "activity, sanctions, PEP/adverse media information "
        "and other relevant evidence.\n\n"
    )

    file.write(
        "5. SAR decision\n\n"
    )

    file.write(
        "Network risk does not automatically trigger SAR "
        "filing. Appropriate human investigation and "
        "institution-specific procedures are required.\n\n"
    )

    file.write(
        "6. Production improvements\n\n"
    )

    file.write(
        "- Real-time graph updates\n"
        "- Temporal graph analysis\n"
        "- Advanced community detection\n"
        "- Entity resolution\n"
        "- Beneficial ownership relationships\n"
        "- Device and IP intelligence\n"
        "- Cross-account behavioral analysis\n"
        "- Threshold calibration\n"
        "- Model monitoring\n"
        "- Data drift monitoring\n"
    )


# ============================================================
# 35. FINAL OUTPUT FILES
# ============================================================

print_header(
    "35. PHASE 12 OUTPUT FILES"
)

print(
    f"""
1. Account network metrics:
   {network_metrics_file}

2. Top 100 network-risk accounts:
   {top_network_file}

3. Network risk summary:
   {network_risk_summary_file}

4. Network red-flag summary:
   {red_flag_file}

5. Network components:
   {component_file}

6. Top 50 network components:
   {top_components_file}

7. Network visualization:
   {network_plot_file}

8. Network analytics summary:
   {summary_file}

9. Network governance notes:
   {governance_file}
"""
)


# ============================================================
# 36. COMPLETION
# ============================================================

print_header(
    "PHASE 12 COMPLETED SUCCESSFULLY"
)

print(
    """
Graph-based AML network analytics have been completed.

Core capabilities:

  [OK] Transaction graph construction
  [OK] Account degree analysis
  [OK] Inbound/outbound analysis
  [OK] Degree centrality
  [OK] Betweenness centrality
  [OK] Network hub detection
  [OK] Shared-counterparty analysis
  [OK] Circular-flow detection
  [OK] Connected-component analysis
  [OK] Network red flags
  [OK] Network risk scoring
  [OK] AML model risk integration
  [OK] Combined network + AML risk
  [OK] Network visualization
  [OK] Governance documentation

IMPORTANT:
Network risk is an analytical indicator only.
Human investigation remains required.

NEXT:
Advanced Graph + Customer Risk Integration
and AML Typology Detection.
"""
)