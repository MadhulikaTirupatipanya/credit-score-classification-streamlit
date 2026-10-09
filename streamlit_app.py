"""Credit Score Classification — BITS Pilani capstone deployment.

Accepts the 24-column CLEANED predictor dataset used immediately before
encoding/binning in the training notebook. It reproduces the saved encoding,
binning, feature order and scaling before calling the exported classifiers.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Score Classification", page_icon="💳", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = BASE_DIR / "streamlit_deployment"
MODELS_DIR = DEPLOY_DIR / "models"

EXPECTED_RAW_COLUMNS = [
    "Month", "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Payment_of_Min_Amount", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance", "Total_Loans_from_Type",
    "Spent", "Value_Payments", "Credit_History_Age_Months",
]
TARGET_MAP = {0: "Poor", 1: "Standard", 2: "Good", "0": "Poor", "1": "Standard", "2": "Good"}

@st.cache_resource
def load_artifacts():
    required = [DEPLOY_DIR / "metadata.joblib", DEPLOY_DIR / "minmax_scaler.joblib",
                DEPLOY_DIR / "preprocessing.joblib"]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing deployment artifact(s): " + ", ".join(missing))
    metadata = joblib.load(required[0])
    scaler = joblib.load(required[1])
    preprocessing = joblib.load(required[2])
    return metadata, scaler, preprocessing

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def normalise_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def get_preproc_value(prep, *names, default=None):
    for name in names:
        if isinstance(prep, dict) and name in prep:
            return prep[name]
    return default

def make_label_mappings(raw_df, prep):
    """Load saved category mappings, with verified training-category fallback.

    The fallback mirrors sorted-category encoding from the training notebook for
    the five known categorical predictors. It prevents an older/stale preprocessing
    artifact from stopping the app at the mapping check.
    """
    mappings = get_preproc_value(prep, "label_mappings", "categorical_mappings", default={}) or {}
    if isinstance(mappings, dict) and mappings:
        return mappings
    bundled = DEPLOY_DIR / "training_category_values.joblib"
    if bundled.exists():
        candidate = joblib.load(bundled)
        if isinstance(candidate, dict) and candidate:
            return candidate
    # Exact sorted category order used by LabelEncoder-style mapping in the export.
    # These categories were verified from the corrected preprocessing artifact.
    return {
        "Month": {v: i for i, v in enumerate(["April", "August", "February", "January", "July", "June", "March", "May"])},
        "Credit_Mix": {v: i for i, v in enumerate(["Bad", "Good", "Standard"])},
        "Payment_of_Min_Amount": {v: i for i, v in enumerate(["NM", "No", "Yes"])},
        "Spent": {v: i for i, v in enumerate(["High_spent", "Low_spent", "Unknown"])},
        "Value_Payments": {v: i for i, v in enumerate(["Large_value_payments", "Small_value_payments", "Unknown"])},
    }

def prepare_model_matrix(raw_df, metadata, prep):
    raw_df = normalise_columns(raw_df)
    target_col = get_preproc_value(prep, "target_column", default="Credit_Score")
    if target_col in raw_df.columns:
        raw_df = raw_df.drop(columns=[target_col])

    expected = get_preproc_value(prep, "raw_columns", "input_columns", default=EXPECTED_RAW_COLUMNS)
    expected = [c for c in expected if c != target_col]
    missing = [c for c in expected if c not in raw_df.columns]
    extra = [c for c in raw_df.columns if c not in expected]
    if missing:
        raise ValueError(
            f"This CSV is missing {len(missing)} required cleaned input column(s): {missing}. "
            "Upload the cleaned 24-column dataset used by the notebook, not the original 27-column Kaggle file."
        )
    if extra:
        st.warning("Ignoring extra column(s): " + ", ".join(extra))
    raw_df = raw_df.loc[:, expected].copy()

    occupation_map = get_preproc_value(prep, "occupation_mapping", "occupation_mapping_dict", default={}) or {}
    global_mean = get_preproc_value(prep, "occupation_global_mean", "global_mean", default=None)
    label_mappings = make_label_mappings(raw_df, prep)
    numeric_columns = get_preproc_value(prep, "numeric_columns", "numeric_cols", default=[])
    bin_edges = get_preproc_value(prep, "bin_edges", "bins", default={}) or {}
    bin_feature_columns = get_preproc_value(prep, "bin_feature_columns", "bin_columns", default=[])
    feature_columns = metadata.get("feature_columns", [])
    if not feature_columns:
        raise ValueError("metadata.joblib does not contain feature_columns.")

    encoded = raw_df.copy()
    if "Occupation" in encoded.columns:
        if not occupation_map or global_mean is None:
            raise ValueError("Occupation target-encoding mapping is missing from preprocessing.joblib.")
        encoded["Occupation"] = encoded["Occupation"].astype(str).map(occupation_map).fillna(float(global_mean))

    for col, mapping in label_mappings.items():
        if col in encoded.columns and col != "Occupation":
            encoded[col] = encoded[col].astype(str).map(mapping).fillna(-1)

    # The model notebook uses numeric columns from the cleaned training frame,
    # then adds one four-bin feature per numeric column using pd.cut.
    for col in numeric_columns:
        if col not in encoded.columns:
            continue
        edges = bin_edges.get(col) if isinstance(bin_edges, dict) else None
        if edges is None or len(edges) < 2:
            raise ValueError(f"Saved bin edges are missing for numeric feature '{col}'.")
        values = pd.to_numeric(encoded[col], errors="coerce")
        values = values.fillna(float(values.median()) if values.notna().any() else 0.0)
        # Values outside the training range are assigned to the nearest edge bin.
        values_for_bins = values.clip(lower=float(edges[0]), upper=float(edges[-1]))
        cats = pd.cut(values_for_bins, bins=np.asarray(edges, dtype=float), precision=1,
                      include_lowest=True, right=True)
        encoded[col + "Bin"] = cats.cat.codes.astype(float)

    # Ensure the exact column order used when training the exported model.
    absent = [c for c in feature_columns if c not in encoded.columns]
    if absent:
        raise ValueError(
            "Preprocessing did not recreate all model features. Missing: " + str(absent) +
            ". Check that preprocessing.joblib was exported from the same notebook run as the models."
        )
    X = encoded.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing values remain in model features: {bad}")
    return X

def display_class(value):
    return TARGET_MAP.get(value, TARGET_MAP.get(str(value), str(value)))

st.title("💳 Credit Score Classification")
st.markdown("Predict a customer's credit-score category using the trained capstone models.")
st.caption("Input: cleaned customer data • Pipeline: saved encoding → binning → scaling → classifier")

try:
    metadata, scaler, preprocessing = load_artifacts()
except Exception as exc:
    st.error(f"Could not load deployment artifacts: {exc}")
    st.stop()

model_files = metadata.get("models", {})
# The metadata may list more models than were packaged; show only files that exist.
available = {}
for model_name, filename in model_files.items():
    path = MODELS_DIR / filename
    if path.exists():
        available[model_name] = path
# Also support the known deployment filenames if metadata's names differ.
known_files = {
    "AdaBoost": "adaboost.joblib", "Bagging": "bagging.joblib",
    "Decision Tree": "decision_tree.joblib", "KNN": "knn.joblib",
    "LightGBM": "lightgbm.joblib", "Logistic Regression": "logistic_regression.joblib",
    "XGBoost": "xgboost.joblib",
}
for name, filename in known_files.items():
    p = MODELS_DIR / filename
    if p.exists():
        available.setdefault(name, p)

with st.sidebar:
    st.header("Model selection")
    if not available:
        st.error(f"No model files found in {MODELS_DIR}")
        st.stop()
    selected_name = st.selectbox("Choose a trained model", list(available.keys()))
    st.caption(f"Expected model features: {len(metadata.get('feature_columns', []))}")
    if st.button("Show model metrics"):
        metrics_path = DEPLOY_DIR / "model_metrics.csv"
        if metrics_path.exists():
            st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)
        else:
            st.info("model_metrics.csv was not included in the deployment folder.")

st.subheader("Upload cleaned customer data")
st.write("Upload a CSV containing the **24 cleaned predictor columns** used immediately before encoding and binning in the training notebook. Do not upload the original raw Kaggle file with ID/Name/SSN/Type_of_Loan/Payment_Behaviour fields.")
with st.expander("Required columns"):
    st.code(", ".join(EXPECTED_RAW_COLUMNS), language="text")

sample_df = pd.DataFrame([{c: "" for c in EXPECTED_RAW_COLUMNS}])
st.download_button("Download 24-column CSV template", sample_df.to_csv(index=False).encode("utf-8"),
                    file_name="credit_score_cleaned_input_template.csv", mime="text/csv")
upload = st.file_uploader("Choose CSV", type=["csv"])

if upload is not None:
    try:
        uploaded = pd.read_csv(upload)
        uploaded = normalise_columns(uploaded)
        st.write(f"Uploaded dataset: **{len(uploaded):,} rows × {len(uploaded.columns)} columns**")
        st.dataframe(uploaded.head(5), use_container_width=True)
        if st.button("Predict credit score", type="primary"):
            with st.spinner("Preparing features and running prediction..."):
                X = prepare_model_matrix(uploaded, metadata, preprocessing)
                X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
                model = load_model(str(available[selected_name]))
                predictions = model.predict(X_scaled)
                output = uploaded.copy()
                output["Predicted_Credit_Score"] = [display_class(p) for p in predictions]
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(X_scaled)
                        classes = list(getattr(model, "classes_", range(probs.shape[1])))
                        for j, cls in enumerate(classes):
                            output[f"Probability_{display_class(cls)}"] = np.round(probs[:, j], 5)
                    except Exception:
                        pass
                st.success(f"Prediction completed using {selected_name}.")
                st.dataframe(output.head(100), use_container_width=True)
                st.download_button("Download predictions CSV", output.to_csv(index=False).encode("utf-8"),
                                   file_name="credit_score_predictions.csv", mime="text/csv")
                if "Credit_Score" in uploaded.columns:
                    st.info("The uploaded target column is not used as an input. This app predicts labels; it does not report test accuracy unless you run a separate evaluation against known labels.")
    except Exception as exc:
        st.error(f"Could not process this file: {exc}")

st.divider()
st.caption("BITS Pilani Capstone • Credit Score Classification using Machine Learning")
