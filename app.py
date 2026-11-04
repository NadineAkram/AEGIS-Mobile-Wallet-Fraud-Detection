
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Mobile Wallet Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# 2. Load Saved Model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_pipeline.joblib"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
default_threshold = float(bundle["threshold"])
features = bundle["features"]

# -----------------------------
# 3. Application Title
# -----------------------------
st.title("💳 Mobile Wallet Fraud Detection System")

st.write(
    "An AI-powered system for detecting potentially "
    "fraudulent mobile wallet transactions."
)

st.divider()

# -----------------------------
# 4. Threshold Slider
# -----------------------------
st.sidebar.header("Model Settings")

threshold = st.sidebar.slider(
    "Fraud Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=default_threshold,
    step=0.05
)

st.sidebar.info(
    f"Current Threshold: {threshold:.2f}"
)

# -----------------------------
# 5. Single Transaction Prediction
# -----------------------------
st.header("1. Analyze a Single Transaction")

with st.form("transaction_form"):

    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox(
            "Transaction Type",
            ["CASH_OUT", "TRANSFER", "CASH_IN",
             "PAYMENT", "DEBIT"]
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=10000.0
        )

        oldbalance_org = st.number_input(
            "Sender's Old Balance",
            min_value=0.0,
            value=20000.0
        )

    with col2:
        step = st.number_input(
            "Transaction Time Step",
            min_value=1,
            value=100
        )

        oldbalance_dest = st.number_input(
            "Receiver's Old Balance",
            min_value=0.0,
            value=5000.0
        )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        use_container_width=True
    )

if submitted:

    transaction = pd.DataFrame(
        [[
            step,
            transaction_type,
            amount,
            oldbalance_org,
            oldbalance_dest
        ]],
        columns=features
    )

    probability = model.predict_proba(
        transaction
    )[0, 1]

    prediction = int(probability >= threshold)

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

    with col2:
        if prediction == 1:
            st.error("⚠️ Potential Fraud Detected")
        else:
            st.success("✅ Transaction Classified as Normal")

    st.progress(float(probability))

# -----------------------------
# 6. CSV Batch Prediction
# -----------------------------
st.divider()

st.header("2. Analyze Multiple Transactions")

st.write(
    "Upload a CSV file containing the required transaction features."
)

uploaded_file = st.file_uploader(
    "Upload Transactions CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        uploaded_df = pd.read_csv(uploaded_file)

        missing_features = [
            col for col in features
            if col not in uploaded_df.columns
        ]

        if missing_features:
            st.error(
                f"Missing required columns: {missing_features}"
            )

        else:
            batch_data = uploaded_df[features].copy()

            numeric_features = [
                "step",
                "amount",
                "oldbalanceOrg",
                "oldbalanceDest"
            ]

            for col in numeric_features:
                batch_data[col] = pd.to_numeric(
                    batch_data[col],
                    errors="coerce"
                )

            if batch_data.isnull().any().any():
                st.error(
                    "The uploaded file contains missing "
                    "or invalid feature values."
                )

            elif (batch_data[numeric_features] < 0).any().any():
                st.error(
                    "Numeric transaction features cannot be negative."
                )

            else:
                probabilities = model.predict_proba(
                    batch_data
                )[:, 1]

                predictions = (
                    probabilities >= threshold
                ).astype(int)

                results = uploaded_df.copy()

                results["Fraud_Probability"] = probabilities
                results["Prediction"] = [
                    "Fraud" if p == 1 else "Normal"
                    for p in predictions
                ]

                st.success("Batch analysis completed!")

                st.metric(
                    "Total Transactions",
                    len(results)
                )

                st.metric(
                    "Potential Fraud Cases",
                    int(predictions.sum())
                )

                st.dataframe(
                    results.head(100),
                    use_container_width=True
                )

                csv_output = results.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    "Download Prediction Results",
                    data=csv_output,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )

    except Exception as error:
        st.error(f"Error processing file: {error}")
