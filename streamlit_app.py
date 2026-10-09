from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

st.set_page_config(page_title="Credit Score Classification", page_icon="💳", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = BASE_DIR / "streamlit_deployment"
MODELS_DIR = DEPLOY_DIR / "models"

CLASS_LABELS = {0: "Poor", 1: "Standard", 2: "Good", "0": "Poor", "1": "Standard", "2": "Good"}

# These names are used only if metadata is absent. The deployment export cell
# writes the exact feature list from X_model into metadata.joblib.
FALLBACK_FEATURE_COLUMNS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance", "Total_Loans_from_Type",
    "Credit_History_Age_Months", "Month", "Occupation",
    "Payment_of_Min_Amount", "Spent", "Value_Payments",
]

# Gradient Boosting is intentionally not listed: the previous deployment's
# saved model failed to load because of sklearn's private _loss module.
FALLBACK_MODEL_FILES = {
    "Decision Tree": "decision_tree.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "KNN": "knn.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "LightGBM": "lightgbm.joblib",
    "Bagging": "bagging.joblib",
    "AdaBoost": "adaboost.joblib",
    "Extra Trees": "extra_trees.joblib",
}


def label(value):
    text = str(value).strip()
    return CLASS_LABELS.get(value, CLASS_LABELS.get(text, text))


@st.cache_resource
def load_artifacts():
    metadata_path = DEPLOY_DIR / "metadata.joblib"
    scaler_path = DEPLOY_DIR / "minmax_scaler.joblib"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run the notebook's Streamlit Deployment Export cell."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing {scaler_path}. Run the notebook's Streamlit Deployment Export cell."
        )

    metadata = joblib.load(metadata_path)
    scaler = joblib.load(scaler_path)
    return metadata, scaler


def get_available_models(metadata):
    registered = metadata.get("models", {})
    available = []
    missing_files = []

    for model_name, fallback_file in FALLBACK_MODEL_FILES.items():
        model_file = registered.get(model_name, fallback_file)
        if not model_file:
            continue
        if (MODELS_DIR / model_file).is_file():
            available.append(model_name)
        else:
            missing_files.append(f"{model_name}: {model_file}")

    return available, missing_files


@st.cache_resource
def load_model(model_name):
    metadata, _ = load_artifacts()
    model_file = metadata.get("models", {}).get(model_name, FALLBACK_MODEL_FILES.get(model_name))
    if not model_file:
        raise FileNotFoundError(f"No exported model file is registered for {model_name}.")
    model_path = MODELS_DIR / model_file
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file missing: {model_path}. Re-run the notebook export cell and commit the models folder."
        )
    return joblib.load(model_path)


def load_metrics():
    path = DEPLOY_DIR / "model_metrics.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def prepare_input(uploaded_df, feature_columns):
    df = uploaded_df.copy()
    target_column = next((c for c in ["Credit_Score", "Target", "target"] if c in df.columns), None)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "The uploaded CSV is missing required columns: " + ", ".join(missing)
            + f". The app expects {len(feature_columns)} exact final encoded/binned model features."
        )

    X = df[feature_columns].copy()
    bad_columns = []
    for col in feature_columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.isna().any() and not X[col].isna().all():
            bad_columns.append(col)
        X[col] = converted

    if bad_columns:
        raise ValueError(
            "These columns contain non-numeric values. Upload the final encoded/binned "
            "feature values produced by the notebook: " + ", ".join(bad_columns)
        )

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError("Missing values in input columns: " + ", ".join(bad))

    y = None
    if target_column:
        raw = df[target_column]
        if raw.dtype == object:
            mapping = {"Poor": 0, "Standard": 1, "Good": 2}
            y = raw.astype(str).str.strip().map(mapping)
        else:
            y = pd.to_numeric(raw, errors="coerce")
        if y.isna().any() or not set(y.astype(int).unique()).issubset({0, 1, 2}):
            raise ValueError("Credit_Score must contain Poor/Standard/Good or numeric codes 0/1/2.")
        y = y.astype(int)

    return X, y


st.title("💳 Credit Score Classification")
st.markdown(
    "Upload customer data and choose a trained model to predict **Poor, Standard, or Good**."
)

try:
    metadata, scaler = load_artifacts()
    feature_columns = metadata.get("feature_columns", FALLBACK_FEATURE_COLUMNS)
except Exception as exc:
    st.error(str(exc))
    st.stop()

available_models, missing_models = get_available_models(metadata)
if not available_models:
    st.error(
        "No model files were found. Run the Streamlit Deployment Export cell in the notebook "
        "and commit the generated streamlit_deployment folder, including its models subfolder."
    )
    st.stop()

with st.sidebar:
    st.header("🤖 Model Selection")
    selected_model = st.selectbox("Select Model", available_models)
    metrics_df = load_metrics()
    st.markdown("---")
    st.subheader("Notebook Performance")

    selected_metrics = pd.DataFrame()
    if not metrics_df.empty and "Model" in metrics_df.columns:
        selected_metrics = metrics_df[metrics_df["Model"].astype(str) == selected_model]

    if not selected_metrics.empty:
        row = selected_metrics.iloc[-1]
        metric_specs = [
            ("Test Accuracy", "Test Accuracy"),
            ("Test Precision (Macro)", "Test Precision"),
            ("Test Recall (Macro)", "Test Recall"),
            ("Test F1 (Macro)", "Test F1"),
        ]
        for display_name, column in metric_specs:
            value = row.get(column)
            if pd.notna(value):
                st.metric(display_name, f"{float(value) * 100:.2f}%")
    else:
        st.info("Metrics will appear after the notebook export cell is run.")
    st.caption("These are held-out test metrics reported by the notebook.")

if missing_models:
    with st.expander("Models not available in this deployment"):
        st.write("These models are hidden because their exported files are missing:")
        st.write(missing_models)

st.info(
    f"This app expects **{len(feature_columns)} numeric final model features** in the same order/representation "
    "used by the notebook. It does not perform raw customer-data cleaning or categorical encoding."
)

st.subheader("1. Upload Customer CSV")
sample_path = DEPLOY_DIR / "streamlit_input_sample.csv"
if sample_path.is_file():
    st.download_button(
        "⬇️ Download Input CSV Sample",
        data=sample_path.read_bytes(),
        file_name="streamlit_input_sample.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader(
    f"Upload a CSV with the {len(feature_columns)} final model features (optionally include Credit_Score for evaluation)",
    type=["csv"],
)
if uploaded_file is None:
    st.write("Upload a CSV to begin prediction.")
    with st.expander("View required columns"):
        st.write(feature_columns)
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file)
    X_input, y_true = prepare_input(uploaded_df, feature_columns)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.write(f"Uploaded rows: **{len(uploaded_df):,}**")
with st.expander("Preview input data"):
    st.dataframe(X_input.head(20), use_container_width=True, hide_index=True)

st.subheader("2. Predict")
if st.button("🔍 Predict Credit Score", type="primary", use_container_width=True):
    try:
        model = load_model(selected_model)
        X_scaled = scaler.transform(X_input)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X_input.index)

        predictions = model.predict(X_scaled)
        prediction_labels = [label(p) for p in predictions]

        result_df = uploaded_df.copy()
        result_df["Predicted_Credit_Score"] = prediction_labels
        result_df["Predicted_Class_Code"] = predictions

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)
            classes = list(getattr(model, "classes_", range(probabilities.shape[1])))
            for idx, cls in enumerate(classes):
                result_df[f"Probability_{label(cls)}"] = probabilities[:, idx]

        st.success(f"Prediction completed using **{selected_model}**.")
        counts = pd.Series(prediction_labels).value_counts().reindex(
            ["Good", "Standard", "Poor"], fill_value=0
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Good", int(counts["Good"]))
        c2.metric("Standard", int(counts["Standard"]))
        c3.metric("Poor", int(counts["Poor"]))

        if y_true is not None:
            st.subheader("Evaluation on Uploaded Data")
            accuracy = accuracy_score(y_true, predictions)
            precision = precision_score(y_true, predictions, average="macro", zero_division=0)
            recall = recall_score(y_true, predictions, average="macro", zero_division=0)
            f1 = f1_score(y_true, predictions, average="macro", zero_division=0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
            m2.metric("Macro Precision", f"{precision * 100:.2f}%")
            m3.metric("Macro Recall", f"{recall * 100:.2f}%")
            m4.metric("Macro F1", f"{f1 * 100:.2f}%")

            st.write("**Classification Report**")
            report = classification_report(
                y_true, predictions, labels=[0, 1, 2],
                target_names=["Poor", "Standard", "Good"],
                output_dict=True, zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_true, predictions, labels=[0, 1, 2])
            st.dataframe(
                pd.DataFrame(
                    cm,
                    index=["Actual Poor", "Actual Standard", "Actual Good"],
                    columns=["Pred Poor", "Pred Standard", "Pred Good"],
                ),
                use_container_width=True,
            )

        st.subheader("Prediction Results")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Prediction Results",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name=f"credit_score_predictions_{selected_model.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as exc:
        st.error("Prediction failed. Check that the exported model and Streamlit environment use compatible package versions.")
        st.exception(exc)

st.markdown("---")
st.caption("BITS Pilani Capstone • Credit Score Classification using Machine Learning")
