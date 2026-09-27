import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

st.set_page_config(
    page_title="Credit Score Classification",
    page_icon="💳",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = BASE_DIR / "streamlit_deployment"
MODELS_DIR = DEPLOY_DIR / "models"

CLASS_LABELS = {0: "Poor", 1: "Standard", 2: "Good", "0": "Poor", "1": "Standard", "2": "Good"}

# These are read from metadata exported by the notebook.
DEFAULT_FEATURE_COLUMNS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance", "Total_Loans_from_Type",
    "Credit_History_Age_Months", "Month", "Occupation",
    "Payment_of_Min_Amount", "Spent", "Value_Payments",
]

MODEL_ORDER = [
    "Decision Tree",
    "Logistic Regression",
    "KNN",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "Gradient Boosting",
    "Bagging",
    "AdaBoost",
    "Extra Trees",
]

MODEL_FILES = {
    "Decision Tree": "decision_tree.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "KNN": "knn.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "LightGBM": "lightgbm.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
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
            "metadata.joblib is missing. Run the deployment-export cell in the notebook first."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            "minmax_scaler.joblib is missing. Run the deployment-export cell in the notebook first."
        )

    metadata = joblib.load(metadata_path)
    scaler = joblib.load(scaler_path)
    return metadata, scaler


@st.cache_resource
def load_model(model_name):
    metadata, _ = load_artifacts()
    model_file = metadata.get("models", {}).get(model_name, MODEL_FILES.get(model_name))
    if not model_file:
        raise FileNotFoundError(f"No exported model file is registered for {model_name}.")

    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def load_metrics():
    metrics_path = DEPLOY_DIR / "model_metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    return pd.read_csv(metrics_path)


def show_prediction(prediction, probabilities=None):
    result = label(prediction)

    if result == "Good":
        st.success("### Predicted Credit Score: GOOD")
    elif result == "Standard":
        st.warning("### Predicted Credit Score: STANDARD")
    elif result == "Poor":
        st.error("### Predicted Credit Score: POOR")
    else:
        st.info(f"### Predicted Class: {result}")

    if probabilities is not None:
        probabilities = np.asarray(probabilities).reshape(-1)
        st.metric("Model confidence", f"{probabilities.max() * 100:.2f}%")

        # Match probability columns to the model's class order.
        st.subheader("Class probabilities")
        classes = list(getattr(st.session_state.get("active_model", None), "classes_", range(len(probabilities))))
        probability_df = pd.DataFrame({
            "Class": [label(c) for c in classes],
            "Probability (%)": probabilities * 100,
        })
        probability_df["Probability (%)"] = probability_df["Probability (%)"].round(2)
        st.dataframe(probability_df, use_container_width=True, hide_index=True)
        st.bar_chart(probability_df.set_index("Class"))


def prepare_input(uploaded_df, feature_columns):
    """Validate and return the exact 41-feature unscaled model input."""
    df = uploaded_df.copy()

    # Optional target column is allowed for evaluation, but never sent to a model.
    target_column = None
    for candidate in ["Credit_Score", "Target", "target"]:
        if candidate in df.columns:
            target_column = candidate
            break

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "The uploaded file is missing required model-input columns: "
            + ", ".join(missing)
        )

    X = df[feature_columns].copy()

    # The notebook's final X_model is numeric after encoding/binning.
    non_numeric = []
    for col in feature_columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.isna().any() and not X[col].isna().all():
            non_numeric.append(col)
        X[col] = converted

    if non_numeric:
        raise ValueError(
            "These columns contain non-numeric values. The uploaded file must contain "
            "the final encoded/binned X_model values from the notebook: "
            + ", ".join(non_numeric)
        )

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Missing values were found in: " + ", ".join(bad) + ". "
            "Use the Streamlit input sample exported by the notebook, or fill all 41 model-input fields."
        )

    y = None
    if target_column is not None:
        raw_y = df[target_column].copy()
        if raw_y.dtype == object:
            mapping = {"Poor": 0, "Standard": 1, "Good": 2}
            y = raw_y.astype(str).str.strip().map(mapping)
        else:
            y = pd.to_numeric(raw_y, errors="coerce")
        if y.isna().any() or not set(y.astype(int).unique()).issubset({0, 1, 2}):
            raise ValueError("Credit_Score must contain Poor/Standard/Good or numeric 0/1/2.")
        y = y.astype(int)

    return X, y


# ------------------------------------------------------------
# Page header
# ------------------------------------------------------------
st.title("💳 Credit Score Classification")
st.markdown(
    "### Upload customer data, select one of the **10 trained models**, and predict **Poor, Standard or Good**."
)
st.info(
    "The application uses the final 41-feature modelling representation from the uploaded notebook. "
    "Upload the CSV exported from the notebook's `streamlit_deployment` section or another CSV with the same 41 numeric columns."
)

# ------------------------------------------------------------
# Sidebar — model selection
# ------------------------------------------------------------
st.sidebar.header("🤖 Model Selection")

try:
    metadata, scaler = load_artifacts()
    feature_columns = metadata.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
except Exception as exc:
    st.sidebar.error("Deployment artifacts are not available.")
    st.error(str(exc))
    st.stop()

selected_model = st.sidebar.selectbox(
    "Select Model",
    MODEL_ORDER,
)

metrics_df = load_metrics()

# Display the selected model's notebook metrics.
selected_metrics = pd.DataFrame()
if not metrics_df.empty and "Model" in metrics_df.columns:
    selected_metrics = metrics_df[metrics_df["Model"].astype(str) == selected_model]

st.sidebar.markdown("---")
st.sidebar.subheader("Notebook Performance")

if not selected_metrics.empty:
    row = selected_metrics.iloc[-1]
    accuracy = row.get("Test Accuracy")
    f1 = row.get("Test F1")
    if pd.notna(accuracy):
        st.sidebar.metric("Test Accuracy", f"{float(accuracy) * 100:.2f}%")
    if pd.notna(f1):
        st.sidebar.metric("Test Macro F1", f"{float(f1) * 100:.2f}%")
else:
    st.sidebar.info("Metrics are shown after the notebook export has been run.")

st.sidebar.caption("These are the notebook's reported test metrics, not a guarantee of performance on new data.")

# ------------------------------------------------------------
# Main input
# ------------------------------------------------------------
st.subheader("1. Upload Customer CSV")

sample_path = DEPLOY_DIR / "streamlit_input_sample.csv"
if sample_path.exists():
    with open(sample_path, "rb") as f:
        st.download_button(
            "⬇️ Download Input CSV Sample",
            data=f.read(),
            file_name="streamlit_input_sample.csv",
            mime="text/csv",
        )

uploaded_file = st.file_uploader(
    "Upload a CSV file containing the 41 final model-input features",
    type=["csv"],
)

if uploaded_file is None:
    st.write("Upload the CSV to begin prediction.")
    st.caption(f"Expected columns: {len(feature_columns)} final encoded/binned features")
    with st.expander("View required columns"):
        st.write(feature_columns)
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

st.write(f"Uploaded rows: **{len(uploaded_df):,}**")

try:
    X_input, y_true = prepare_input(uploaded_df, feature_columns)
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Preview uploaded data"):
    st.dataframe(X_input.head(20), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
st.subheader("2. Predict")

if st.button("🔍 Predict Credit Score", type="primary", use_container_width=True):
    try:
        model = load_model(selected_model)
        st.session_state.active_model = model

        # The notebook fits the scaler on unscaled X_model and then trains
        # the classifiers on the scaled representation.
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

        # Summary
        st.subheader("Prediction Summary")
        counts = pd.Series(prediction_labels).value_counts().reindex(
            ["Good", "Standard", "Poor"], fill_value=0
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Good", int(counts["Good"]))
        c2.metric("Standard", int(counts["Standard"]))
        c3.metric("Poor", int(counts["Poor"]))

        # Evaluation when a target column is present.
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
                y_true,
                predictions,
                labels=[0, 1, 2],
                target_names=["Poor", "Standard", "Good"],
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_true, predictions, labels=[0, 1, 2])
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Poor", "Actual Standard", "Actual Good"],
                columns=["Pred Poor", "Pred Standard", "Pred Good"],
            )
            st.dataframe(cm_df, use_container_width=True)

        st.subheader("Prediction Results")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Prediction Results",
            data=csv_bytes,
            file_name=f"credit_score_predictions_{selected_model.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as exc:
        st.error("Prediction failed.")
        st.exception(exc)

st.markdown("---")
st.caption("BITS Pilani Capstone • Credit Score Classification using Machine Learning")
