import streamlit as st
import pandas as pd
import joblib
import altair as alt
from pathlib import Path

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AEGIS | Mobile Wallet Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# 2. Design Tokens + Global CSS
# -----------------------------
NAVY = "#0B1220"
PANEL = "#111B2E"
BORDER = "#1E2A44"
BLUE = "#5B8DEF"
TEAL = "#2DD4BF"
RED = "#F25F5C"
WHITE = "#F1F5F9"
GRAY = "#94A3B8"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', 'Segoe UI', Tahoma, sans-serif;
}
.stApp { background: #0B1220; color: #F1F5F9; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0E1729;
    border-right: 1px solid #1E2A44;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #F1F5F9; font-size: 1.05rem; }
section[data-testid="stSidebar"] label p { color: #94A3B8; }

/* Headings & text */
h1, h2, h3, h4 { color: #F1F5F9; letter-spacing: -0.01em; }
p, li, label, span { color: inherit; }
hr { border-color: #1E2A44 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #1E2A44;
}
.stTabs [data-baseweb="tab"] {
    height: 46px;
    padding: 0 18px;
    color: #94A3B8;
    background: transparent;
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] { color: #F1F5F9 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #5B8DEF; }

/* Form panel */
div[data-testid="stForm"] {
    background: #111B2E;
    border: 1px solid #1E2A44;
    border-radius: 14px;
    padding: 1.4rem 1.4rem 1rem 1.4rem;
}

/* Inputs */
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
    background-color: #0B1220 !important;
    border-color: #1E2A44 !important;
    border-radius: 8px !important;
}
input { color: #F1F5F9 !important; }
.stNumberInput button { background: #14203A; color: #94A3B8; }

/* Buttons */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: #5B8DEF;
    color: #0B1220;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    background: #7AA3F5;
    color: #0B1220;
    border: none;
}

/* File uploader */
div[data-testid="stFileUploader"] section {
    background: #111B2E;
    border: 1px dashed #2A3A5E;
    border-radius: 14px;
}
div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] span { color: #94A3B8; }

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid #1E2A44;
    border-radius: 12px;
    overflow: hidden;
}

/* Alerts */
div[data-testid="stAlert"] { border-radius: 10px; }

/* Custom components */
.ae-header {
    display: flex; align-items: center; gap: 16px;
    padding: 18px 22px;
    background: linear-gradient(135deg, #111B2E 0%, #0E1729 100%);
    border: 1px solid #1E2A44;
    border-radius: 16px;
    margin-bottom: 1.2rem;
}
.ae-logo {
    width: 52px; height: 52px; flex: none;
    display: flex; align-items: center; justify-content: center;
    background: #14203A; border: 1px solid #2A3A5E; border-radius: 12px;
}
.ae-title { font-size: 1.6rem; font-weight: 700; color: #F1F5F9; line-height: 1.1; }
.ae-tag { color: #94A3B8; font-size: 0.95rem; margin-top: 4px; }
.ae-pill {
    margin-left: auto;
    display: flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 999px;
    background: rgba(45,212,191,0.10); color: #2DD4BF;
    border: 1px solid rgba(45,212,191,0.35);
    font-size: 0.85rem; font-weight: 500; white-space: nowrap;
}
.ae-dot { width: 8px; height: 8px; border-radius: 50%; background: #2DD4BF; }

.ae-section { font-size: 1.15rem; font-weight: 600; color: #F1F5F9; margin: 0.4rem 0 0.2rem 0; }
.ae-sub { color: #94A3B8; font-size: 0.92rem; margin-bottom: 1rem; }

.ae-kpi {
    background: #111B2E; border: 1px solid #1E2A44; border-radius: 14px;
    padding: 18px 20px; height: 100%;
    border-left-width: 3px;
}
.ae-kpi.blue { border-left-color: #5B8DEF; }
.ae-kpi.teal { border-left-color: #2DD4BF; }
.ae-kpi.red  { border-left-color: #F25F5C; }
.ae-kpi.gray { border-left-color: #3A4A6E; }
.ae-kpi-label { color: #94A3B8; font-size: 0.85rem; }
.ae-kpi-value { color: #F1F5F9; font-size: 1.9rem; font-weight: 700; margin-top: 4px; line-height: 1.15; }
.ae-kpi-note { color: #94A3B8; font-size: 0.8rem; margin-top: 4px; }

.ae-card {
    background: #111B2E; border: 1px solid #1E2A44; border-radius: 14px;
    padding: 20px 22px;
}
.ae-card-title { font-size: 1rem; font-weight: 600; color: #F1F5F9; margin-bottom: 10px; }

.ae-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.ae-step { background: #0E1729; border: 1px solid #1E2A44; border-radius: 12px; padding: 16px; }
.ae-step b { color: #F1F5F9; display: block; margin-bottom: 4px; }
.ae-step span { color: #94A3B8; font-size: 0.88rem; }

.ae-result {
    border-radius: 14px; padding: 22px 24px;
    border: 1px solid; display: flex; align-items: center; gap: 18px;
}
.ae-result.fraud { background: rgba(242,95,92,0.10); border-color: rgba(242,95,92,0.5); }
.ae-result.safe  { background: rgba(45,212,191,0.08); border-color: rgba(45,212,191,0.45); }
.ae-result-icon { font-size: 2rem; }
.ae-result-title { font-size: 1.25rem; font-weight: 700; }
.ae-result.fraud .ae-result-title { color: #F25F5C; }
.ae-result.safe .ae-result-title { color: #2DD4BF; }
.ae-result-sub { color: #94A3B8; font-size: 0.9rem; margin-top: 2px; }

.ae-gauge { margin-top: 18px; }
.ae-gauge-track {
    position: relative; height: 12px; border-radius: 999px;
    background: #1A2640; overflow: visible;
}
.ae-gauge-fill { height: 100%; border-radius: 999px; }
.ae-gauge-mark {
    position: absolute; top: -5px; width: 2px; height: 22px; background: #F1F5F9;
}
.ae-gauge-labels {
    display: flex; justify-content: space-between;
    color: #94A3B8; font-size: 0.78rem; margin-top: 8px;
}

.ae-empty {
    background: #111B2E; border: 1px dashed #2A3A5E; border-radius: 14px;
    padding: 28px; text-align: center; color: #94A3B8;
}
.ae-footer { text-align: center; color: #5E6E8F; font-size: 0.8rem; margin-top: 2.5rem; }

@media (max-width: 800px) {
    .ae-steps { grid-template-columns: 1fr; }
    .ae-header { flex-wrap: wrap; }
    .ae-pill { margin-left: 0; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(block: str):
    """Render an HTML block (strips indentation so Markdown doesn't treat it as code)."""
    cleaned = "\n".join(l.strip() for l in block.splitlines() if l.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "", tone: str = "blue"):
    return (
        f'<div class="ae-kpi {tone}">'
        f'<div class="ae-kpi-label">{label}</div>'
        f'<div class="ae-kpi-value">{value}</div>'
        f'<div class="ae-kpi-note">{note}</div>'
        f'</div>'
    )


def style_chart(chart, height=280):
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=GRAY, titleColor=GRAY,
            gridColor=BORDER, domainColor=BORDER, tickColor=BORDER
        )
        .configure_legend(labelColor=GRAY, titleColor=GRAY)
    )


# -----------------------------
# 3. Load Saved Model (unchanged)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_pipeline.joblib"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
default_threshold = float(bundle["threshold"])
features = bundle["features"]

numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest"
]
categorical_features = [c for c in features if c not in numeric_features]

# -----------------------------
# 4. Professional Header
# -----------------------------
html("""
<div class="ae-header">
  <div class="ae-logo">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2.5 4.5 5.5v5.8c0 4.5 3.1 8.6 7.5 10.2 4.4-1.6 7.5-5.7 7.5-10.2V5.5L12 2.5Z" fill="#5B8DEF" fill-opacity="0.18" stroke="#5B8DEF" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="m8.6 12.2 2.4 2.4 4.4-4.8" stroke="#2DD4BF" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
  <div>
    <div class="ae-title">AEGIS</div>
    <div class="ae-tag">Intelligent protection for mobile wallet transactions</div>
  </div>
  <div class="ae-pill"><span class="ae-dot"></span>Model active</div>
</div>
""")

# -----------------------------
# 5. Threshold Slider (unchanged logic)
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

st.sidebar.caption(
    "Lower values flag more transactions (higher recall). "
    "Higher values flag fewer (higher precision)."
)

# -----------------------------
# 6. Tabs
# -----------------------------
tab_overview, tab_single, tab_batch, tab_analytics = st.tabs(
    ["Overview", "Single Transaction", "Batch Prediction", "Visual Analytics"]
)

# ======================================================
# Single Transaction (logic unchanged)
# ======================================================
with tab_single:
    html("""
    <div class="ae-section">Analyze a single transaction</div>
    <div class="ae-sub">Enter the transaction details and AEGIS will estimate the fraud probability.</div>
    """)

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

        st.write("")
        if prediction == 1:
            tone, icon, title = "fraud", "⚠️", "Potential Fraud Detected"
            sub = "This transaction scored above the active threshold. Review before approving."
            bar_color = RED
        else:
            tone, icon, title = "safe", "✅", "Transaction Classified as Normal"
            sub = "This transaction scored below the active threshold."
            bar_color = TEAL

        html(f"""
        <div class="ae-result {tone}">
          <div class="ae-result-icon">{icon}</div>
          <div>
            <div class="ae-result-title">{title}</div>
            <div class="ae-result-sub">{sub}</div>
          </div>
        </div>
        """)

        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.markdown(
            kpi("Fraud Probability", f"{probability:.2%}",
                "Model confidence", "red" if prediction == 1 else "teal"),
            unsafe_allow_html=True
        )
        m2.markdown(
            kpi("Active Threshold", f"{threshold:.2f}",
                "Adjust in the sidebar", "blue"),
            unsafe_allow_html=True
        )
        m3.markdown(
            kpi("Decision", "Fraud" if prediction == 1 else "Normal",
                "Probability vs. threshold",
                "red" if prediction == 1 else "teal"),
            unsafe_allow_html=True
        )

        html(f"""
        <div class="ae-card ae-gauge" style="margin-top:16px;">
          <div class="ae-card-title">Risk level</div>
          <div class="ae-gauge-track">
            <div class="ae-gauge-fill" style="width:{float(probability) * 100:.1f}%; background:{bar_color};"></div>
            <div class="ae-gauge-mark" style="left:{threshold * 100:.1f}%;"></div>
          </div>
          <div class="ae-gauge-labels">
            <span>0%</span>
            <span>Threshold {threshold:.0%}</span>
            <span>100%</span>
          </div>
        </div>
        """)

# ======================================================
# Batch Prediction (logic unchanged; results kept for Overview/Analytics)
# ======================================================
results = None
predictions = None

with tab_batch:
    html("""
    <div class="ae-section">Analyze multiple transactions</div>
    <div class="ae-sub">Upload a CSV file containing the required transaction features.</div>
    """)

    html(f"""
    <div class="ae-card" style="margin-bottom:14px;">
      <div class="ae-card-title">Required columns</div>
      <div class="ae-sub" style="margin-bottom:0;">{", ".join(features)}</div>
    </div>
    """)

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

                    b1, b2, b3 = st.columns(3)
                    b1.markdown(
                        kpi("Total Transactions", f"{len(results):,}",
                            "Rows analyzed", "blue"),
                        unsafe_allow_html=True
                    )
                    b2.markdown(
                        kpi("Potential Fraud Cases", f"{int(predictions.sum()):,}",
                            "Above threshold", "red"),
                        unsafe_allow_html=True
                    )
                    b3.markdown(
                        kpi("Normal Transactions",
                            f"{len(results) - int(predictions.sum()):,}",
                            "Below threshold", "teal"),
                        unsafe_allow_html=True
                    )

                    st.write("")
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

# ======================================================
# Dashboard Overview
# ======================================================
with tab_overview:
    html("""
    <div class="ae-section">Dashboard overview</div>
    <div class="ae-sub">A quick summary of the model and your latest batch analysis.</div>
    """)

    if results is None:
        o1, o2, o3 = st.columns(3)
        o1.markdown(kpi("Model status", "Ready", "Pipeline loaded", "teal"),
                    unsafe_allow_html=True)
        o2.markdown(kpi("Active threshold", f"{threshold:.2f}",
                        f"Default {default_threshold:.2f}", "blue"),
                    unsafe_allow_html=True)
        o3.markdown(kpi("Input features", f"{len(features)}",
                        "Used by the model", "gray"),
                    unsafe_allow_html=True)

        st.write("")
        html("""
        <div class="ae-card">
          <div class="ae-card-title">How AEGIS works</div>
          <div class="ae-steps">
            <div class="ae-step"><b>Enter or upload</b><span>Type a single transaction, or upload a CSV with many.</span></div>
            <div class="ae-step"><b>Score</b><span>The model estimates the probability that each transaction is fraud.</span></div>
            <div class="ae-step"><b>Decide</b><span>Transactions above your threshold are flagged for review.</span></div>
          </div>
        </div>
        """)
    else:
        total = len(results)
        fraud_n = int(predictions.sum())
        fraud_rate = fraud_n / total if total else 0
        avg_prob = float(results["Fraud_Probability"].mean())

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi("Total transactions", f"{total:,}",
                        "Latest batch", "blue"), unsafe_allow_html=True)
        k2.markdown(kpi("Flagged as fraud", f"{fraud_n:,}",
                        "Above threshold", "red"), unsafe_allow_html=True)
        k3.markdown(kpi("Fraud rate", f"{fraud_rate:.2%}",
                        "Flagged / total", "red" if fraud_n else "teal"),
                    unsafe_allow_html=True)
        k4.markdown(kpi("Average risk", f"{avg_prob:.2%}",
                        "Mean fraud probability", "gray"),
                    unsafe_allow_html=True)

        st.write("")
        html('<div class="ae-section">Highest-risk transactions</div>')
        top_risk = results.sort_values(
            "Fraud_Probability", ascending=False
        ).head(5)
        st.dataframe(top_risk, use_container_width=True)

# ======================================================
# Visual Analytics
# ======================================================
with tab_analytics:
    html("""
    <div class="ae-section">Visual analytics</div>
    <div class="ae-sub">How the latest batch breaks down by outcome, risk score and transaction type.</div>
    """)

    if results is None:
        html("""
        <div class="ae-empty">
          No batch results yet. Upload a CSV in the Batch Prediction tab to see charts here.
        </div>
        """)
    else:
        chart_col1, chart_col2 = st.columns(2)

        # Outcome split (donut)
        with chart_col1:
            html('<div class="ae-card-title">Outcome split</div>')
            split = (
                results["Prediction"].value_counts()
                .reindex(["Normal", "Fraud"], fill_value=0)
                .rename_axis("Prediction").reset_index(name="Count")
            )
            donut = alt.Chart(split).mark_arc(innerRadius=70, outerRadius=115).encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color(
                    "Prediction:N",
                    scale=alt.Scale(domain=["Normal", "Fraud"], range=[TEAL, RED]),
                    legend=alt.Legend(title=None, orient="bottom")
                ),
                tooltip=["Prediction", "Count"]
            )
            st.altair_chart(style_chart(donut, 300),
                            use_container_width=True, theme=None)

        # Probability distribution
        with chart_col2:
            html('<div class="ae-card-title">Fraud probability distribution</div>')
            hist = alt.Chart(results[["Fraud_Probability"]]).mark_bar(
                color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3
            ).encode(
                x=alt.X("Fraud_Probability:Q",
                        bin=alt.Bin(maxbins=20, extent=[0, 1]),
                        title="Fraud probability"),
                y=alt.Y("count()", title="Transactions"),
                tooltip=[alt.Tooltip("count()", title="Transactions")]
            )
            rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(
                color=RED, strokeDash=[5, 4], size=2
            ).encode(x=alt.X("t:Q", scale=alt.Scale(domain=[0, 1])))
            st.altair_chart(style_chart(hist + rule, 300),
                            use_container_width=True, theme=None)

        # Fraud by transaction type
        if categorical_features:
            type_col = categorical_features[0]
            st.write("")
            html(f'<div class="ae-card-title">Flagged transactions by {type_col}</div>')
            by_type = (
                results.assign(Flagged=predictions)
                .groupby(type_col)
                .agg(Transactions=("Flagged", "size"), Flagged=("Flagged", "sum"))
                .reset_index()
            )
            by_type["Normal"] = by_type["Transactions"] - by_type["Flagged"]
            long_df = by_type.melt(
                id_vars=type_col, value_vars=["Normal", "Flagged"],
                var_name="Outcome", value_name="Count"
            )
            bars = alt.Chart(long_df).mark_bar(
                cornerRadiusTopLeft=3, cornerRadiusTopRight=3
            ).encode(
                x=alt.X(f"{type_col}:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Count:Q", title="Transactions"),
                color=alt.Color(
                    "Outcome:N",
                    scale=alt.Scale(domain=["Normal", "Flagged"], range=[TEAL, RED]),
                    legend=alt.Legend(title=None, orient="bottom")
                ),
                tooltip=[type_col, "Outcome", "Count"]
            )
            st.altair_chart(style_chart(bars, 320),
                            use_container_width=True, theme=None)

html('<div class="ae-footer">AEGIS · Mobile Wallet Fraud Detection</div>')
