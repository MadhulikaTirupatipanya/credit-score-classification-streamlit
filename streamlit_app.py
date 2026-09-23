
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Score Classification",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models_export",
    "credit_score_pipeline.joblib"
)


@st.cache_resource
def load_package():
    return joblib.load(MODEL_PATH)


def label(value):
    mapping = {
        0: "Poor",
        1: "Standard",
        2: "Good",
        "0": "Poor",
        "1": "Standard",
        "2": "Good"
    }

    return mapping.get(value, mapping.get(str(value), str(value)))


# ------------------------------------------------------------
# LOAD SAVED MODEL PACKAGE
# ------------------------------------------------------------

try:
    package = load_package()

    pipeline = package["pipeline"]
    feature_columns = package["feature_columns"]
    category_mappings = package["category_mappings"]

except Exception as e:
    st.error("Could not load the trained model.")
    st.exception(e)
    st.stop()


# ------------------------------------------------------------
# PAGE
# ------------------------------------------------------------

st.title("💳 Credit Score Classification")

st.markdown(
    "### Enter customer details and predict **Good, Standard or Poor**."
)

st.info(
    "This application uses the trained machine-learning model "
    "from the Credit Score Classification capstone project."
)


# ------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------

st.header("Customer Credit Information")

with st.form("credit_form"):

    st.subheader("Personal & Income")

    c1, c2, c3 = st.columns(3)

    age = c1.number_input(
        "Age",
        min_value=18.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

    annual_income = c2.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    monthly_salary = c3.number_input(
        "Monthly Inhand Salary",
        min_value=0.0,
        value=4000.0,
        step=100.0
    )


    st.subheader("Accounts & Loans")

    c1, c2, c3, c4 = st.columns(4)

    bank_accounts = c1.number_input(
        "Number of Bank Accounts",
        min_value=0.0,
        value=3.0,
        step=1.0
    )

    credit_cards = c2.number_input(
        "Number of Credit Cards",
        min_value=0.0,
        value=4.0,
        step=1.0
    )

    interest_rate = c3.number_input(
        "Interest Rate",
        min_value=0.0,
        value=10.0,
        step=0.5
    )

    number_of_loans = c4.number_input(
        "Number of Loans",
        min_value=0.0,
        value=2.0,
        step=1.0
    )


    c1, c2, c3, c4 = st.columns(4)

    total_loans = c1.number_input(
        "Total Loan Types",
        min_value=0.0,
        value=2.0,
        step=1.0
    )

    delayed_payment = c2.number_input(
        "Number of Delayed Payments",
        min_value=0.0,
        value=5.0,
        step=1.0
    )

    delay_due_date = c3.number_input(
        "Delay From Due Date",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    inquiries = c4.number_input(
        "Credit Inquiries",
        min_value=0.0,
        value=4.0,
        step=1.0
    )


    st.subheader("Debt, Utilization & Payments")

    c1, c2, c3 = st.columns(3)

    changed_limit = c1.number_input(
        "Changed Credit Limit",
        value=5.0,
        step=0.5
    )

    debt = c2.number_input(
        "Outstanding Debt",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    utilization = c3.number_input(
        "Credit Utilization Ratio",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )


    c1, c2, c3 = st.columns(3)

    emi = c1.number_input(
        "Total EMI Per Month",
        min_value=0.0,
        value=200.0,
        step=25.0
    )

    invested = c2.number_input(
        "Amount Invested Monthly",
        min_value=0.0,
        value=100.0,
        step=25.0
    )

    balance = c3.number_input(
        "Monthly Balance",
        min_value=0.0,
        value=2000.0,
        step=100.0
    )


    st.subheader("Credit History & Behaviour")

    c1, c2, c3 = st.columns(3)

    history_months = c1.number_input(
        "Credit History Age (Months)",
        min_value=0.0,
        value=60.0,
        step=1.0
    )

    month = c2.selectbox(
        "Month",
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August"
        ]
    )

    occupation = c3.selectbox(
        "Occupation",
        [
            "Accountant",
            "Architect",
            "Developer",
            "Doctor",
            "Engineer",
            "Entrepreneur",
            "Journalist",
            "Lawyer",
            "Manager",
            "Mechanic",
            "Media_Manager",
            "Musician",
            "Scientist",
            "Teacher",
            "Writer",
            "Other"
        ]
    )


    c1, c2, c3 = st.columns(3)

    min_payment = c1.selectbox(
        "Payment of Minimum Amount",
        ["Yes", "No", "NM"]
    )

    spent = c2.selectbox(
        "Spending Behaviour",
        [
            "High_spent",
            "Low_spent",
            "Unknown"
        ]
    )

    value_payments = c3.selectbox(
        "Value of Payments",
        [
            "Small_value_payments",
            "Large_value_payments",
            "Unknown"
        ]
    )


    submitted = st.form_submit_button(
        "🔍 Calculate Credit Score",
        use_container_width=True
    )


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

if submitted:

    # Raw user input
    X = pd.DataFrame([{
        "Age": age,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": bank_accounts,
        "Num_Credit_Card": credit_cards,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": number_of_loans,
        "Delay_from_due_date": delay_due_date,
        "Num_of_Delayed_Payment": delayed_payment,
        "Changed_Credit_Limit": changed_limit,
        "Num_Credit_Inquiries": inquiries,
        "Outstanding_Debt": debt,
        "Credit_Utilization_Ratio": utilization,
        "Total_EMI_per_month": emi,
        "Amount_invested_monthly": invested,
        "Monthly_Balance": balance,
        "Total_Loans_from_Type": total_loans,
        "Credit_History_Age_Months": history_months,
        "Month": month,
        "Occupation": occupation,
        "Payment_of_Min_Amount": min_payment,
        "Spent": spent,
        "Value_Payments": value_payments
    }])


    # --------------------------------------------------------
    # APPLY SAME CATEGORICAL ENCODING USED DURING TRAINING
    # --------------------------------------------------------

    for col, mapping in category_mappings.items():

        if col in X.columns:

            X[col] = (
                X[col]
                .astype(str)
                .map(mapping)
                .fillna(-1)
            )


    # Ensure exact feature order
    X = X.reindex(columns=feature_columns)

    # Convert everything to numeric
    X = X.astype(float)


    st.subheader("Customer Input")

    st.dataframe(
        X,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        pred = pipeline.predict(X)[0]

        probabilities = None

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X)[0]


        result = label(pred)

        st.divider()

        if result == "Good":

            st.success(
                "### Credit Score Classification: GOOD"
            )

        elif result == "Poor":

            st.error(
                "### Credit Score Classification: POOR"
            )

        elif result == "Standard":

            st.warning(
                "### Credit Score Classification: STANDARD"
            )

        else:

            st.info(
                f"### Predicted Class: {result}"
            )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        if probabilities is not None:

            classes = list(
                getattr(
                    pipeline,
                    "classes_",
                    range(len(probabilities))
                )
            )

            result_df = pd.DataFrame({
                "Class": [label(c) for c in classes],
                "Model Probability (%)":
                    probabilities * 100
            })

            result_df["Model Probability (%)"] = (
                result_df["Model Probability (%)"]
                .round(2)
            )

            st.metric(
                "Model confidence",
                f"{probabilities.max() * 100:.2f}%"
            )

            st.subheader("Class Probabilities")

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                result_df.set_index("Class")
            )

            st.caption(
                "These percentages are model probabilities, "
                "not a traditional 300–900 credit score."
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


st.markdown("---")

st.caption(
    "BITS Pilani Capstone • "
    "Credit Score Classification using Machine Learning"
)
