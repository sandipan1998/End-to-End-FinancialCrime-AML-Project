# =============================================================================
# PHASE 6 - SHAP EXPLAINABILITY
# US FINANCIAL CRIME / AML TRANSACTION MONITORING PROJECT
# =============================================================================
#
# Purpose:
#   Explain WHY the ML model considers a transaction suspicious.
#
# Supports:
#   1. XGBoost
#   2. CatBoost
#   3. Logistic Regression
#   4. Logistic Regression inside sklearn Pipeline
#
# Important:
#   SHAP explanations show model reasoning.
#   They do NOT prove that a customer committed money laundering or fraud.
#
# =============================================================================

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "aml_ml_features.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "models"
)

SHAP_REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "shap"
)

os.makedirs(SHAP_REPORT_DIR, exist_ok=True)


# =============================================================================
# 2. PRINT HEADER
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 6 - SHAP EXPLAINABILITY")
print("=" * 80)


# =============================================================================
# 3. LOAD DATASET
# =============================================================================

print("\n" + "=" * 80)
print("3. LOADING ML DATASET")
print("=" * 80)

print(f"Loading ML dataset:")
print(DATA_FILE)

if not os.path.exists(DATA_FILE):
    print("\nERROR: ML dataset not found.")
    print(DATA_FILE)
    sys.exit(1)

df = pd.read_csv(DATA_FILE)

print(f"\nDataset shape: {df.shape}")


# =============================================================================
# 4. CLEAN DATA
# =============================================================================

print("\n" + "=" * 80)
print("4. CLEANING DATA")
print("=" * 80)

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

df = df.replace([np.inf, -np.inf], np.nan)


# =============================================================================
# 5. LOCATE ACTUAL BEST MODEL
# =============================================================================

print("\n" + "=" * 80)
print("5. LOCATING TRAINED MODEL")
print("=" * 80)


def detect_best_model():

    comparison_file = os.path.join(
        MODEL_REPORT_DIR,
        "model_comparison.csv"
    )

    # -------------------------------------------------------------------------
    # First choice:
    # Read Phase 5 model comparison and select the model with highest PR-AUC.
    # -------------------------------------------------------------------------

    if os.path.exists(comparison_file):

        try:

            comparison = pd.read_csv(comparison_file)

            print("\nModel comparison found:")
            print(comparison.to_string(index=False))

            # Try several possible column names
            pr_auc_column = None

            for column in [
                "pr_auc",
                "PR-AUC",
                "PR_AUC",
                "test_pr_auc",
                "Test_PR_AUC"
            ]:

                if column in comparison.columns:
                    pr_auc_column = column
                    break

            model_column = None

            for column in [
                "model",
                "Model",
                "model_name",
                "Model_Name"
            ]:

                if column in comparison.columns:
                    model_column = column
                    break

            if pr_auc_column is not None and model_column is not None:

                comparison[pr_auc_column] = pd.to_numeric(
                    comparison[pr_auc_column],
                    errors="coerce"
                )

                comparison = comparison.dropna(
                    subset=[pr_auc_column]
                )

                if len(comparison) > 0:

                    best_row = comparison.loc[
                        comparison[pr_auc_column].idxmax()
                    ]

                    best_model_name = str(
                        best_row[model_column]
                    ).strip()

                    print(
                        f"\nBest model according to PR-AUC: "
                        f"{best_model_name}"
                    )

                    if "xgb" in best_model_name.lower():
                        model_file = os.path.join(
                            MODEL_DIR,
                            "aml_xgboost.json"
                        )

                        if os.path.exists(model_file):
                            return "XGBoost", model_file

                    if "cat" in best_model_name.lower():
                        model_file = os.path.join(
                            MODEL_DIR,
                            "aml_catboost.cbm"
                        )

                        if os.path.exists(model_file):
                            return "CatBoost", model_file

                    if "logistic" in best_model_name.lower():
                        model_file = os.path.join(
                            MODEL_DIR,
                            "aml_logistic_regression.joblib"
                        )

                        if os.path.exists(model_file):
                            return "Logistic Regression", model_file

        except Exception as e:

            print(
                "\nWARNING: Could not use model_comparison.csv."
            )

            print(
                f"Reason: {e}"
            )


    # -------------------------------------------------------------------------
    # Fallback model detection
    # -------------------------------------------------------------------------

    model_candidates = [

        (
            "XGBoost",
            os.path.join(
                MODEL_DIR,
                "aml_xgboost.json"
            )
        ),

        (
            "CatBoost",
            os.path.join(
                MODEL_DIR,
                "aml_catboost.cbm"
            )
        ),

        (
            "Logistic Regression",
            os.path.join(
                MODEL_DIR,
                "aml_logistic_regression.joblib"
            )
        )
    ]

    for model_name, model_file in model_candidates:

        if os.path.exists(model_file):

            return model_name, model_file


    return None, None


model_name, model_file = detect_best_model()


if model_name is None:

    print("\nERROR: No trained ML model was found.")

    print("\nExpected model files:")
    print(
        os.path.join(
            MODEL_DIR,
            "aml_xgboost.json"
        )
    )
    print(
        os.path.join(
            MODEL_DIR,
            "aml_catboost.cbm"
        )
    )
    print(
        os.path.join(
            MODEL_DIR,
            "aml_logistic_regression.joblib"
        )
    )

    sys.exit(1)


print(f"\nModel detected: {model_name}")
print(f"Model file: {model_file}")


# =============================================================================
# 6. PREPARE MODEL FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("6. PREPARING MODEL FEATURES")
print("=" * 80)


# Columns that should NOT be used as ML model features
EXCLUDED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "sender_account",
    "receiver_account",
    "account_id",
    "timestamp",
    "scenario",
    "known_suspicious",
    "target"
]


# Direct rule-engine outputs are excluded because the first ML experiment
# was intentionally designed to learn from transaction/customer behavior
# independently from the final rule score.
DIRECT_RULE_COLUMNS = [
    "rule_alert",
    "rule_risk_score",
    "red_flag_count",
    "customer_average_rule_score",
    "customer_max_rule_score",
    "high_risk_rule_indicator",
    "customer_average_red_flags",
    "customer_max_red_flags"
]


excluded = set(
    EXCLUDED_COLUMNS + DIRECT_RULE_COLUMNS
)


candidate_features = [
    column
    for column in df.columns
    if column not in excluded
]


# Keep numeric features only
feature_columns = []

for column in candidate_features:

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):
        feature_columns.append(column)


if len(feature_columns) == 0:

    print(
        "\nERROR: No numeric ML features were found."
    )

    sys.exit(1)


X = df[feature_columns].copy()


# Clean values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


print(
    f"\nNumber of model features: "
    f"{len(feature_columns)}"
)

print(
    f"Feature matrix shape: "
    f"{X.shape}"
)


# =============================================================================
# 7. LOAD MODEL
# =============================================================================

print("\n" + "=" * 80)
print("7. LOADING MODEL")
print("=" * 80)


model = None

if model_name == "XGBoost":

    try:

        from xgboost import XGBClassifier

        model = XGBClassifier()

        model.load_model(model_file)

        print(
            "XGBoost model loaded successfully."
        )

    except Exception as e:

        print(
            "\nERROR loading XGBoost model:"
        )

        print(e)

        sys.exit(1)


elif model_name == "CatBoost":

    try:

        from catboost import CatBoostClassifier

        model = CatBoostClassifier()

        model.load_model(model_file)

        print(
            "CatBoost model loaded successfully."
        )

    except Exception as e:

        print(
            "\nERROR loading CatBoost model:"
        )

        print(e)

        sys.exit(1)


elif model_name == "Logistic Regression":

    try:

        model = joblib.load(model_file)

        print(
            "Logistic Regression model loaded successfully."
        )

        print(
            f"Loaded model type: {type(model)}"
        )

    except Exception as e:

        print(
            "\nERROR loading Logistic Regression model:"
        )

        print(e)

        sys.exit(1)


# =============================================================================
# 8. PREPARE DATA FOR SHAP
# =============================================================================

print("\n" + "=" * 80)
print("8. PREPARING DATA FOR SHAP")
print("=" * 80)


# Maximum rows used for SHAP to keep execution practical
MAX_SHAP_ROWS = 5000


if len(X) > MAX_SHAP_ROWS:

    shap_sample = X.sample(
        n=MAX_SHAP_ROWS,
        random_state=42
    )

else:

    shap_sample = X.copy()


print(
    f"Rows selected for SHAP: "
    f"{len(shap_sample)}"
)


# =============================================================================
# 9. GENERATE MODEL PREDICTIONS
# =============================================================================

print("\n" + "=" * 80)
print("9. GENERATING MODEL PREDICTIONS")
print("=" * 80)


try:

    probabilities = model.predict_proba(X)[:, 1]

except Exception as e:

    print(
        "\nERROR generating model predictions:"
    )

    print(e)

    sys.exit(1)


def probability_to_risk_level(probability):

    if probability >= 0.75:
        return "CRITICAL"

    elif probability >= 0.50:
        return "HIGH"

    elif probability >= 0.25:
        return "MEDIUM"

    else:
        return "LOW"


ml_risk_level = [
    probability_to_risk_level(
        probability
    )
    for probability in probabilities
]


prediction_df = pd.DataFrame({

    "ml_risk_probability": probabilities,

    "ml_risk_score":
        probabilities * 100,

    "ml_risk_level":
        ml_risk_level

})


print("\nPrediction distribution:")

print(
    prediction_df[
        "ml_risk_level"
    ].value_counts()
)


# =============================================================================
# 10. CREATE SHAP EXPLAINER
# =============================================================================

print("\n" + "=" * 80)
print("10. CREATING SHAP EXPLAINER")
print("=" * 80)


explainer = None
shap_values = None
shap_data = None


# =============================================================================
# 10A. XGBOOST
# =============================================================================

if model_name == "XGBoost":

    print(
        "\nUsing SHAP TreeExplainer for XGBoost."
    )

    try:

        shap_data = shap_sample.copy()

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            shap_data
        )

    except Exception as e:

        print(
            "\nERROR creating XGBoost SHAP explainer:"
        )

        print(e)

        sys.exit(1)


# =============================================================================
# 10B. CATBOOST
# =============================================================================

elif model_name == "CatBoost":

    print(
        "\nUsing SHAP TreeExplainer for CatBoost."
    )

    try:

        shap_data = shap_sample.copy()

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            shap_data
        )

    except Exception as e:

        print(
            "\nERROR creating CatBoost SHAP explainer:"
        )

        print(e)

        sys.exit(1)


# =============================================================================
# 10C. LOGISTIC REGRESSION
# =============================================================================

elif model_name == "Logistic Regression":

    print(
        "\nPreparing Logistic Regression SHAP explanation."
    )


    # -------------------------------------------------------------------------
    # IMPORTANT FIX:
    #
    # The saved Logistic Regression model may be:
    #
    #   LogisticRegression
    #
    # OR:
    #
    #   Pipeline(
    #       preprocessing,
    #       LogisticRegression
    #   )
    #
    # SHAP LinearExplainer does not directly accept the Pipeline in the
    # installed SHAP version.
    #
    # Therefore:
    #
    #   1. Extract preprocessing part.
    #   2. Transform X.
    #   3. Extract final LogisticRegression estimator.
    #   4. Give transformed data to SHAP.
    # -------------------------------------------------------------------------

    try:

        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        if isinstance(
            model,
            Pipeline
        ):

            print(
                "\nDetected sklearn Pipeline."
            )

            print(
                f"Pipeline steps: "
                f"{model.named_steps}"
            )


            # -------------------------------------------------------------
            # Extract final estimator
            # -------------------------------------------------------------

            final_estimator = model.steps[-1][1]

            print(
                "\nFinal estimator:"
            )

            print(
                type(final_estimator)
            )


            if not isinstance(
                final_estimator,
                LogisticRegression
            ):

                print(
                    "\nWARNING: Final estimator is not "
                    "LogisticRegression."
                )

                print(
                    "Attempting generic linear SHAP explanation."
                )


            # -------------------------------------------------------------
            # Transform sample through preprocessing
            # -------------------------------------------------------------

            preprocessing_pipeline = Pipeline(
                model.steps[:-1]
            )

            shap_transformed = (
                preprocessing_pipeline.transform(
                    shap_sample
                )
            )


            # -------------------------------------------------------------
            # Convert sparse matrix to dense if required
            # -------------------------------------------------------------

            if hasattr(
                shap_transformed,
                "toarray"
            ):

                shap_transformed = (
                    shap_transformed.toarray()
                )


            shap_transformed = np.asarray(
                shap_transformed,
                dtype=float
            )


            print(
                "\nOriginal SHAP data shape:"
            )

            print(
                shap_sample.shape
            )


            print(
                "\nTransformed SHAP data shape:"
            )

            print(
                shap_transformed.shape
            )


            # -------------------------------------------------------------
            # Create SHAP feature names
            #
            # For a standard scaler / numeric preprocessing pipeline,
            # transformed columns preserve the same feature order.
            # -------------------------------------------------------------

            transformed_feature_names = (
                feature_columns.copy()
            )


            # -------------------------------------------------------------
            # Safety check
            # -------------------------------------------------------------

            if (
                shap_transformed.shape[1]
                != len(transformed_feature_names)
            ):

                print(
                    "\nWARNING:"
                )

                print(
                    "Number of transformed features does not match "
                    "original feature count."
                )

                print(
                    f"Transformed features: "
                    f"{shap_transformed.shape[1]}"
                )

                print(
                    f"Original features: "
                    f"{len(transformed_feature_names)}"
                )


                # Try obtaining names from preprocessing
                try:

                    if hasattr(
                        preprocessing_pipeline,
                        "get_feature_names_out"
                    ):

                        transformed_feature_names = list(
                            preprocessing_pipeline.get_feature_names_out(
                                feature_columns
                            )
                        )

                        print(
                            "\nFeature names obtained "
                            "from preprocessing pipeline."
                        )

                except Exception:

                    transformed_feature_names = [
                        f"feature_{i}"
                        for i in range(
                            shap_transformed.shape[1]
                        )
                    ]


            # -------------------------------------------------------------
            # Create background data
            # -------------------------------------------------------------

            background_size = min(
                100,
                len(shap_transformed)
            )

            background_data = (
                shap_transformed[
                    :background_size
                ]
            )


            # -------------------------------------------------------------
            # SHAP LinearExplainer
            # -------------------------------------------------------------

            print(
                "\nCreating SHAP LinearExplainer "
                "using final LogisticRegression estimator..."
            )

            explainer = shap.LinearExplainer(
                final_estimator,
                background_data
            )


            shap_values = explainer.shap_values(
                shap_transformed
            )


            # SHAP data must use transformed features
            shap_data = pd.DataFrame(
                shap_transformed,
                columns=transformed_feature_names
            )


        else:

            # -------------------------------------------------------------
            # Model is direct LogisticRegression
            # -------------------------------------------------------------

            print(
                "\nNo Pipeline detected."
            )

            print(
                "Using direct LogisticRegression model."
            )


            shap_data = shap_sample.copy()


            background_size = min(
                100,
                len(shap_data)
            )


            background_data = (
                shap_data.iloc[
                    :background_size
                ]
            )


            explainer = shap.LinearExplainer(
                model,
                background_data
            )


            shap_values = explainer.shap_values(
                shap_data
            )


    except Exception as e:

        print(
            "\nERROR creating Logistic Regression SHAP explainer:"
        )

        print(
            repr(e)
        )

        print(
            "\nModel type:"
        )

        print(
            type(model)
        )

        if hasattr(
            model,
            "steps"
        ):

            print(
                "\nPipeline steps:"
            )

            print(
                model.steps
            )

        sys.exit(1)


# =============================================================================
# 11. NORMALIZE SHAP OUTPUT
# =============================================================================

print("\n" + "=" * 80)
print("11. PROCESSING SHAP VALUES")
print("=" * 80)


shap_values = np.asarray(
    shap_values
)


print(
    f"Raw SHAP shape: "
    f"{shap_values.shape}"
)


# Some SHAP/model combinations can return 3D arrays.
#
# Expected binary classification format:
#   rows x features
#
# If:
#   rows x features x classes
#
# use class 1.

if shap_values.ndim == 3:

    print(
        "\n3D SHAP output detected."
    )

    if shap_values.shape[-1] >= 2:

        shap_values = (
            shap_values[:, :, 1]
        )

    else:

        shap_values = (
            shap_values[:, :, 0]
        )


# Some newer SHAP APIs can return lists
if isinstance(
    shap_values,
    list
):

    if len(shap_values) >= 2:

        shap_values = np.asarray(
            shap_values[1]
        )

    else:

        shap_values = np.asarray(
            shap_values[0]
        )


print(
    f"Processed SHAP shape: "
    f"{shap_values.shape}"
)


# =============================================================================
# 12. ALIGN FEATURE NAMES
# =============================================================================

if shap_values.shape[1] != shap_data.shape[1]:

    print(
        "\nWARNING: SHAP feature count differs from data feature count."
    )

    print(
        f"SHAP features: {shap_values.shape[1]}"
    )

    print(
        f"Data features: {shap_data.shape[1]}"
    )

    # Last-resort feature names
    shap_data = shap_data.iloc[
        :,
        :shap_values.shape[1]
    ]


feature_names = list(
    shap_data.columns
)


# =============================================================================
# 13. GLOBAL SHAP FEATURE IMPORTANCE
# =============================================================================

print("\n" + "=" * 80)
print("13. CALCULATING GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 80)


mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)


shap_importance = pd.DataFrame({

    "feature":
        feature_names,

    "mean_absolute_shap":
        mean_abs_shap

})


shap_importance = (
    shap_importance
    .sort_values(
        "mean_absolute_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


shap_importance[
    "rank"
] = (
    np.arange(
        1,
        len(shap_importance) + 1
    )
)


# Save
shap_importance_file = os.path.join(
    SHAP_REPORT_DIR,
    "shap_feature_importance.csv"
)

shap_importance.to_csv(
    shap_importance_file,
    index=False
)


print(
    "\nTop 20 SHAP features:"
)

print(
    shap_importance.head(20).to_string(
        index=False
    )
)


# =============================================================================
# 14. TRANSACTION-LEVEL SHAP EXPLANATIONS
# =============================================================================

print("\n" + "=" * 80)
print("14. CREATING TRANSACTION-LEVEL SHAP EXPLANATIONS")
print("=" * 80)


sample_indices = shap_sample.index.to_numpy()


transaction_ids = []

if "transaction_id" in df.columns:

    transaction_ids = (
        df.loc[
            sample_indices,
            "transaction_id"
        ]
        .astype(str)
        .tolist()
    )

else:

    transaction_ids = [
        str(index)
        for index in sample_indices
    ]


customer_ids = []

if "customer_id" in df.columns:

    customer_ids = (
        df.loc[
            sample_indices,
            "customer_id"
        ]
        .astype(str)
        .tolist()
    )

else:

    customer_ids = [
        ""
        for _ in sample_indices
    ]


amounts = []

if "amount_usd" in df.columns:

    amounts = (
        pd.to_numeric(
            df.loc[
                sample_indices,
                "amount_usd"
            ],
            errors="coerce"
        )
        .fillna(0)
        .tolist()
    )

else:

    amounts = [
        0
        for _ in sample_indices
    ]


# Create transaction explanation dataframe
transaction_explanations = pd.DataFrame({

    "transaction_id":
        transaction_ids,

    "customer_id":
        customer_ids,

    "amount_usd":
        amounts,

    "ml_risk_probability":
        probabilities[
            sample_indices
        ],

    "ml_risk_score":
        probabilities[
            sample_indices
        ] * 100,

    "ml_risk_level":
        np.array(
            ml_risk_level
        )[
            sample_indices
        ]

})


# =============================================================================
# 15. TOP POSITIVE AND NEGATIVE SHAP DRIVERS
# =============================================================================

print("\n" + "=" * 80)
print("15. IDENTIFYING TOP RISK DRIVERS")
print("=" * 80)


top_positive_features = []
top_positive_values = []

top_negative_features = []
top_negative_values = []


for row_number in range(
    len(shap_values)
):

    row_shap = shap_values[
        row_number
    ]

    # -------------------------------------------------------------
    # Positive SHAP = pushes prediction toward suspicious
    # -------------------------------------------------------------

    positive_indices = np.where(
        row_shap > 0
    )[0]


    if len(positive_indices) > 0:

        positive_sorted = (
            positive_indices[
                np.argsort(
                    row_shap[
                        positive_indices
                    ]
                )[::-1]
            ]
        )

        top_positive = (
            positive_sorted[:3]
        )

        positive_features = [
            feature_names[i]
            for i in top_positive
        ]

        positive_values = [
            float(row_shap[i])
            for i in top_positive
        ]

    else:

        positive_features = []
        positive_values = []


    # -------------------------------------------------------------
    # Negative SHAP = pushes prediction toward normal
    # -------------------------------------------------------------

    negative_indices = np.where(
        row_shap < 0
    )[0]


    if len(negative_indices) > 0:

        negative_sorted = (
            negative_indices[
                np.argsort(
                    np.abs(
                        row_shap[
                            negative_indices
                        ]
                    )
                )[::-1]
            ]
        )

        top_negative = (
            negative_sorted[:3]
        )

        negative_features = [
            feature_names[i]
            for i in top_negative
        ]

        negative_values = [
            float(row_shap[i])
            for i in top_negative
        ]

    else:

        negative_features = []
        negative_values = []


    top_positive_features.append(
        " | ".join(
            positive_features
        )
    )

    top_positive_values.append(
        " | ".join(
            [
                f"{value:.6f}"
                for value in positive_values
            ]
        )
    )


    top_negative_features.append(
        " | ".join(
            negative_features
        )
    )

    top_negative_values.append(
        " | ".join(
            [
                f"{value:.6f}"
                for value in negative_values
            ]
        )
    )


transaction_explanations[
    "top_positive_shap_features"
] = top_positive_features


transaction_explanations[
    "top_positive_shap_values"
] = top_positive_values


transaction_explanations[
    "top_negative_shap_features"
] = top_negative_features


transaction_explanations[
    "top_negative_shap_values"
] = top_negative_values


# =============================================================================
# 16. SAVE TRANSACTION SHAP EXPLANATIONS
# =============================================================================

transaction_explanations_file = os.path.join(
    SHAP_REPORT_DIR,
    "transaction_shap_explanations.csv"
)

transaction_explanations.to_csv(
    transaction_explanations_file,
    index=False
)


print(
    f"\nSaved:"
)

print(
    transaction_explanations_file
)


# =============================================================================
# 17. TOP RISK TRANSACTIONS
# =============================================================================

print("\n" + "=" * 80)
print("17. IDENTIFYING TOP RISK TRANSACTIONS")
print("=" * 80)


top_risk_transactions = (
    transaction_explanations
    .sort_values(
        "ml_risk_probability",
        ascending=False
    )
    .head(100)
    .copy()
)


top_risk_file = os.path.join(
    SHAP_REPORT_DIR,
    "top_risk_transactions.csv"
)


top_risk_transactions.to_csv(
    top_risk_file,
    index=False
)


print(
    f"Saved:"
)

print(
    top_risk_file
)


# =============================================================================
# 18. INDIVIDUAL TRANSACTION EXPLANATION
# =============================================================================

print("\n" + "=" * 80)
print("18. CREATING INDIVIDUAL TRANSACTION EXPLANATION")
print("=" * 80)


# Select highest-risk transaction in SHAP sample
highest_risk_position = int(
    np.argmax(
        probabilities[
            sample_indices
        ]
    )
)


highest_risk_global_index = (
    sample_indices[
        highest_risk_position
    ]
)


highest_risk_probability = (
    probabilities[
        highest_risk_global_index
    ]
)


highest_risk_shap = (
    shap_values[
        highest_risk_position
    ]
)


individual_rows = []


for feature_index, feature_name in enumerate(
    feature_names
):

    individual_rows.append({

        "transaction_id":
            str(
                df.loc[
                    highest_risk_global_index,
                    "transaction_id"
                ]
            )
            if "transaction_id" in df.columns
            else str(
                highest_risk_global_index
            ),

        "customer_id":
            str(
                df.loc[
                    highest_risk_global_index,
                    "customer_id"
                ]
            )
            if "customer_id" in df.columns
            else "",

        "feature":
            feature_name,

        "feature_value":
            shap_data.iloc[
                highest_risk_position,
                feature_index
            ],

        "shap_value":
            highest_risk_shap[
                feature_index
            ],

        "impact":
            (
                "INCREASES_SUSPICION"
                if highest_risk_shap[
                    feature_index
                ] > 0
                else "DECREASES_SUSPICION"
            )

    })


individual_explanation = pd.DataFrame(
    individual_rows
)


individual_explanation[
    "absolute_shap_value"
] = (
    individual_explanation[
        "shap_value"
    ].abs()
)


individual_explanation = (
    individual_explanation
    .sort_values(
        "absolute_shap_value",
        ascending=False
    )
)


individual_file = os.path.join(
    SHAP_REPORT_DIR,
    "individual_transaction_explanation.csv"
)


individual_explanation.to_csv(
    individual_file,
    index=False
)


print(
    f"Saved:"
)

print(
    individual_file
)


# =============================================================================
# 19. SHAP SUMMARY PLOT
# =============================================================================

print("\n" + "=" * 80)
print("19. CREATING SHAP SUMMARY PLOT")
print("=" * 80)


try:

    plt.figure(
        figsize=(12, 8)
    )

    shap.summary_plot(
        shap_values,
        shap_data,
        show=False,
        max_display=20
    )

    plt.tight_layout()

    summary_plot_file = os.path.join(
        SHAP_REPORT_DIR,
        "shap_summary.png"
    )

    plt.savefig(
        summary_plot_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:"
    )

    print(
        summary_plot_file
    )

except Exception as e:

    print(
        "\nWARNING: Could not create SHAP summary plot."
    )

    print(e)


# =============================================================================
# 20. SHAP BAR PLOT
# =============================================================================

print("\n" + "=" * 80)
print("20. CREATING SHAP BAR PLOT")
print("=" * 80)


try:

    plt.figure(
        figsize=(12, 8)
    )

    shap.summary_plot(
        shap_values,
        shap_data,
        plot_type="bar",
        show=False,
        max_display=20
    )

    plt.tight_layout()

    bar_plot_file = os.path.join(
        SHAP_REPORT_DIR,
        "shap_bar.png"
    )

    plt.savefig(
        bar_plot_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:"
    )

    print(
        bar_plot_file
    )

except Exception as e:

    print(
        "\nWARNING: Could not create SHAP bar plot."
    )

    print(e)


# =============================================================================
# 21. INDIVIDUAL TRANSACTION SHAP PLOT
# =============================================================================

print("\n" + "=" * 80)
print("21. CREATING INDIVIDUAL TRANSACTION SHAP PLOT")
print("=" * 80)


try:

    individual_plot_data = (
        individual_explanation
        .head(15)
        .sort_values(
            "shap_value"
        )
    )


    plt.figure(
        figsize=(10, 7)
    )


    plt.barh(
        individual_plot_data[
            "feature"
        ],
        individual_plot_data[
            "shap_value"
        ]
    )


    plt.axvline(
        x=0,
        linewidth=1
    )


    plt.xlabel(
        "SHAP Value"
    )


    plt.ylabel(
        "Feature"
    )


    plt.title(
        "Individual Transaction SHAP Explanation"
    )


    plt.tight_layout()


    individual_plot_file = os.path.join(
        SHAP_REPORT_DIR,
        "individual_transaction_shap.png"
    )


    plt.savefig(
        individual_plot_file,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Saved:"
    )

    print(
        individual_plot_file
    )


except Exception as e:

    print(
        "\nWARNING: Could not create individual SHAP plot."
    )

    print(e)


# =============================================================================
# 22. GLOBAL SHAP SUMMARY TEXT
# =============================================================================

print("\n" + "=" * 80)
print("22. CREATING SHAP SUMMARY REPORT")
print("=" * 80)


summary_file = os.path.join(
    SHAP_REPORT_DIR,
    "shap_global_summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PHASE 6 - SHAP GLOBAL EXPLAINABILITY SUMMARY\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Model: {model_name}\n"
    )

    file.write(
        f"Model file: {model_file}\n"
    )

    file.write(
        f"Original dataset rows: {len(df)}\n"
    )

    file.write(
        f"SHAP sample rows: {len(shap_sample)}\n"
    )

    file.write(
        f"Number of model features: {len(feature_names)}\n\n"
    )


    file.write(
        "TOP 20 GLOBAL SHAP FEATURES\n"
    )

    file.write(
        "-" * 70 + "\n"
    )


    for _, row in shap_importance.head(20).iterrows():

        file.write(
            f"{int(row['rank'])}. "
            f"{row['feature']} "
            f"-> mean |SHAP| = "
            f"{row['mean_absolute_shap']:.6f}\n"
        )


    file.write(
        "\n\nINTERPRETATION\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        "Positive SHAP values push the model toward a suspicious "
        "prediction.\n"
    )

    file.write(
        "Negative SHAP values push the model toward a normal "
        "prediction.\n"
    )

    file.write(
        "Mean absolute SHAP measures the average importance of "
        "a feature.\n"
    )

    file.write(
        "SHAP explanations describe model behavior and should "
        "not be interpreted as proof of criminal activity.\n"
    )


print(
    f"Saved:"
)

print(
    summary_file
)


# =============================================================================
# 23. PRINT HIGHEST-RISK TRANSACTION
# =============================================================================

print("\n" + "=" * 80)
print("23. HIGHEST-RISK TRANSACTION EXPLANATION")
print("=" * 80)


highest_risk_transaction = (
    transaction_explanations
    .iloc[
        highest_risk_position
    ]
)


print(
    "\nTransaction ID:"
)

print(
    highest_risk_transaction[
        "transaction_id"
    ]
)


print(
    "\nCustomer ID:"
)

print(
    highest_risk_transaction[
        "customer_id"
    ]
)


print(
    "\nAmount:"
)

print(
    f"${float(highest_risk_transaction['amount_usd']):,.2f}"
)


print(
    "\nML Risk Probability:"
)

print(
    f"{float(highest_risk_transaction['ml_risk_probability']) * 100:.2f}%"
)


print(
    "\nML Risk Level:"
)

print(
    highest_risk_transaction[
        "ml_risk_level"
    ]
)


print(
    "\nTop Positive SHAP Drivers:"
)

print(
    highest_risk_transaction[
        "top_positive_shap_features"
    ]
)


print(
    "\nTop Negative SHAP Drivers:"
)

print(
    highest_risk_transaction[
        "top_negative_shap_features"
    ]
)


# =============================================================================
# 24. FINAL OUTPUT SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 6 OUTPUT FILES")
print("=" * 80)


print(
    "\n1. SHAP Feature Importance:"
)

print(
    shap_importance_file
)


print(
    "\n2. Transaction SHAP Explanations:"
)

print(
    transaction_explanations_file
)


print(
    "\n3. Top Risk Transactions:"
)

print(
    top_risk_file
)


print(
    "\n4. Individual Transaction Explanation:"
)

print(
    individual_file
)


print(
    "\n5. SHAP Summary Plot:"
)

print(
    os.path.join(
        SHAP_REPORT_DIR,
        "shap_summary.png"
    )
)


print(
    "\n6. SHAP Bar Plot:"
)

print(
    os.path.join(
        SHAP_REPORT_DIR,
        "shap_bar.png"
    )
)


print(
    "\n7. Individual Transaction SHAP Plot:"
)

print(
    os.path.join(
        SHAP_REPORT_DIR,
        "individual_transaction_shap.png"
    )
)


print(
    "\n8. SHAP Global Summary:"
)

print(
    summary_file
)


# =============================================================================
# 25. COMPLETION
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 6 COMPLETED SUCCESSFULLY")
print("=" * 80)

print(
    "\nSHAP explainability has been generated successfully."
)

print(
    "\nThe model can now provide:"
)

print(
    "  - Global AML risk drivers"
)

print(
    "  - Transaction-level risk drivers"
)

print(
    "  - Positive SHAP factors"
)

print(
    "  - Negative SHAP factors"
)

print(
    "  - Highest-risk transaction explanations"
)

print(
    "  - Analyst-friendly model explanations"
)

print(
    "\nNext phase:"
)

print(
    "PHASE 7 - AML RISK SCORING & ALERT PRIORITIZATION"
)

print(
    "=" * 80
)