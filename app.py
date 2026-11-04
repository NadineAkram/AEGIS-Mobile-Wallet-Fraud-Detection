import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
 
# =====================================================
# 1. Page configuration
# =====================================================
st.set_page_config(
    page_title="AEGIS | Mobile Wallet Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# =====================================================
# 2. Styling (custom CSS only - no JavaScript)
# =====================================================
# Palette (light, enterprise FinTech):
#   Background #FFFFFF | Surface #F8FAFC | Border #E2E8F0
#   Navy text #0F172A  | Muted #64748B
#   Blue #2563EB | Teal #14B8A6 | Red #EF4444 | Green #16A34A
CUSTOM_CSS = """
<style>
:root { color-scheme: light; }
.stApp, [data-testid="stAppViewContainer"] { background-color: #FFFFFF; color: #0F172A; }
header[data-testid="stHeader"] { background: rgba(255,255,255,0.85); }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
 
h1, h2, h3, h4 { color: #0F172A !important; letter-spacing: -0.01em; }
[data-testid="stMarkdownContainer"] h3 { font-size: 1.25rem; font-weight: 650; }
[data-testid="stWidgetLabel"] p { color: #334155; font-weight: 500; }
[data-testid="stCaptionContainer"] { color: #64748B; }
 
/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}
section[data-testid="stSidebar"] * { color: #0F172A; }
.aegis-brand { display: flex; align-items: center; gap: 12px; }
.aegis-brand-name { font-size: 1.4rem; font-weight: 700; line-height: 1.1; color: #0F172A; }
.aegis-brand-sub { color: #64748B; font-size: 0.8rem; }
.aegis-side-desc { color: #64748B; font-size: 0.85rem; margin: 12px 0 18px 0; line-height: 1.45; }
.aegis-side-label { color: #64748B; font-size: 0.78rem; font-weight: 600; margin: 4px 0 6px 4px; }
.aegis-status {
    display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #475569;
    padding: 10px 12px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
}
.aegis-dot { width: 8px; height: 8px; border-radius: 50%; background: #16A34A; flex: none; }
 
/* Sidebar navigation buttons */
[class*="st-key-nav_"] button {
    justify-content: flex-start; text-align: left; border-radius: 10px;
    padding: 0.6rem 0.85rem; font-weight: 500; min-height: 2.7rem;
}
[class*="st-key-nav_"] button[kind="secondary"],
[class*="st-key-nav_"] button[data-testid="stBaseButton-secondary"] {
    background: transparent; border: 1px solid transparent; color: #334155;
}
[class*="st-key-nav_"] button[kind="secondary"]:hover,
[class*="st-key-nav_"] button[data-testid="stBaseButton-secondary"]:hover {
    background: #EEF2F7; border-color: #E2E8F0; color: #0F172A;
}
[class*="st-key-nav_"] button[kind="primary"],
[class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] {
    background: #EAF1FF; border: 1px solid #BFD3FB; color: #1D4ED8; font-weight: 600;
}
 
/* ---------- Cards (containers with key="card_*") ---------- */
[class*="st-key-card"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
[class*="st-key-card"][class*="_blue_"]  { border-top: 3px solid #2563EB; }
[class*="st-key-card"][class*="_teal_"]  { border-top: 3px solid #14B8A6; }
[class*="st-key-card"][class*="_red_"]   { border-top: 3px solid #EF4444; }
[class*="st-key-card"][class*="_green_"] { border-top: 3px solid #16A34A; }
[class*="st-key-card"][class*="_gray_"]  { border-top: 3px solid #CBD5E1; }
.st-key-card_hero {
    background: #F5F9FF; border-color: #DBEAFE;
    padding: 1.6rem 1.8rem; box-shadow: none;
}
[data-testid="stMetricLabel"] p { color: #64748B !important; font-size: 0.85rem; }
[data-testid="stMetricValue"] { color: #0F172A; font-weight: 700; }
 
/* ---------- Form ---------- */
[data-testid="stForm"] {
    background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
    padding: 1.4rem; box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {
    background-color: #FFFFFF !important; border-color: #CBD5E1 !important; border-radius: 8px !important;
}
input { color: #0F172A !important; }
 
/* ---------- Buttons ---------- */
button[kind="primary"], button[kind="primaryFormSubmit"],
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
    background-color: #2563EB; color: #FFFFFF; border: 1px solid #2563EB;
    border-radius: 10px; font-weight: 600; padding: 0.6rem 1.2rem;
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover,
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
    background-color: #1D4ED8; border-color: #1D4ED8; color: #FFFFFF;
}
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
    background-color: #FFFFFF; color: #0F172A; border: 1px solid #CBD5E1; border-radius: 10px; font-weight: 500;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
    border-color: #2563EB; color: #2563EB;
}
 
/* ---------- Uploader / tables / tabs ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: #F8FAFC; border: 1.5px dashed #93C5FD; border-radius: 12px;
}
[data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 12px; overflow: hidden; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] { color: #64748B; padding: 0 14px; height: 44px; }
.stTabs [aria-selected="true"] { color: #1D4ED8; }
[data-testid="stExpander"] { border: 1px solid #E2E8F0; border-radius: 12px; background: #FFFFFF; }
 
/* ---------- Page header ---------- */
.aegis-page-title { font-size: 1.9rem; font-weight: 700; color: #0F172A; line-height: 1.2; }
.aegis-page-sub { color: #64748B; font-size: 1rem; margin: 6px 0 18px 0; max-width: 760px; }
.aegis-hero-title { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin: 10px 0 0 0; line-height: 1.15; }
.aegis-hero-sub { color: #475569; font-size: 1.05rem; margin: 10px 0 18px 0; max-width: 720px; line-height: 1.55; }
.aegis-pill {
    display: inline-flex; align-items: center; gap: 8px; padding: 4px 12px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 500; background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0;
}
.aegis-muted { color: #64748B; font-size: 0.93rem; line-height: 1.5; }
.aegis-step-num {
    width: 30px; height: 30px; border-radius: 50%; background: #EAF1FF; color: #1D4ED8;
    display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 10px;
}
.aegis-card-title { font-weight: 600; color: #0F172A; margin-bottom: 4px; }
 
/* ---------- Result card ---------- */
.aegis-result { border-radius: 14px; padding: 1.3rem 1.5rem; border: 1px solid; margin: 0.4rem 0 1rem 0; }
.aegis-result.fraud { background: #FEF2F2; border-color: #FCA5A5; }
.aegis-result.legit { background: #F0FDF4; border-color: #86EFAC; }
.aegis-result-badge {
    display: inline-block; font-size: 0.78rem; font-weight: 700; padding: 3px 10px;
    border-radius: 999px; color: #FFFFFF; margin-bottom: 8px;
}
.aegis-result.fraud .aegis-result-badge { background: #DC2626; }
.aegis-result.legit .aegis-result-badge { background: #16A34A; }
.aegis-result-title { font-size: 1.35rem; font-weight: 700; color: #0F172A; }
.aegis-result-text { color: #475569; margin-top: 6px; font-size: 0.97rem; line-height: 1.55; }
 
/* Probability meter */
.aegis-meter-track { position: relative; height: 12px; border-radius: 999px; background: #E2E8F0; margin-top: 14px; }
.aegis-meter-fill { height: 100%; border-radius: 999px; }
.aegis-meter-mark { position: absolute; top: -6px; width: 2px; height: 24px; background: #0F172A; }
.aegis-meter-labels { display: flex; justify-content: space-between; color: #64748B; font-size: 0.78rem; margin-top: 10px; }
 
@media (max-width: 800px) {
    .aegis-page-title { font-size: 1.55rem; }
    .aegis-hero-title { font-size: 1.75rem; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
 
LOGO_SVG = """
<svg width="38" height="38" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2.5 4.5 5.5v5.8c0 4.5 3.1 8.6 7.5 10.2 4.4-1.6 7.5-5.7 7.5-10.2V5.5L12 2.5Z"
 fill="#2563EB" fill-opacity="0.12" stroke="#2563EB" stroke-width="1.6" stroke-linejoin="round"/>
<path d="m8.6 12.2 2.4 2.4 4.4-4.8" stroke="#14B8A6" stroke-width="1.9"
 stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""
 
# =====================================================
# 3. Load saved model (path and bundle keys unchanged)
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_pipeline.joblib"
 
 
@st.cache_resource(show_spinner="Loading fraud detection model...")
def load_bundle(path: Path):
    return joblib.load(path)
 
 
try:
    bundle = load_bundle(MODEL_PATH)
    model = bundle["model"]
    default_threshold = float(bundle["threshold"])
    features = bundle["features"]
except FileNotFoundError:
    st.error(
        "Model file not found. Make sure "
        "`models/fraud_detection_pipeline.joblib` exists next to `app.py`."
    )
    st.stop()
except Exception as error:
    st.error(f"The model could not be loaded: {error}")
    st.stop()
 
# Same numeric columns the original app coerced for batch input
NUMERIC_FEATURES = [
    c for c in ["step", "amount", "oldbalanceOrg", "oldbalanceDest"] if c in features
]
TRANSACTION_TYPES = ["CASH_OUT", "TRANSFER", "CASH_IN", "PAYMENT", "DEBIT"]
 
# Default form values (also used to build the sample CSV template)
DEFAULTS = {
    "step": 100,
    "type": "CASH_OUT",
    "amount": 10000.0,
    "oldbalanceOrg": 20000.0,
    "oldbalanceDest": 5000.0,
}
 
FEATURE_INFO = {
    "step": "Time step of the transaction in the dataset's timeline",
    "type": "Transaction type (CASH_OUT, TRANSFER, CASH_IN, PAYMENT, DEBIT)",
    "amount": "Amount of the transaction",
    "oldbalanceOrg": "Origin (sender) account balance before the transaction",
    "oldbalanceDest": "Destination (receiver) account balance before the transaction",
}
 
# Project-reported evaluation results (provided by the project author)
REPORTED_METRICS = [
    ("Precision", "89.39%", "blue"),
    ("Recall", "60.50%", "teal"),
    ("F1-score", "72.05%", "blue"),
    ("Average Precision", "0.7424", "teal"),
]
 
PAGES = [
    ("Dashboard", ":material/dashboard:"),
    ("Single Transaction Analysis", ":material/receipt_long:"),
    ("Batch CSV Analysis", ":material/upload_file:"),
    ("About the Model", ":material/info:"),
]
PAGE_NAMES = [name for name, _ in PAGES]
 
if "nav" not in st.session_state:
    st.session_state["nav"] = PAGE_NAMES[0]
 
 
def go_to(page: str):
    """Button callback used to switch pages."""
    st.session_state["nav"] = page
 
 
# =====================================================
# 4. Sidebar: brand, navigation, threshold, status
# =====================================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="aegis-brand">{LOGO_SVG}
        <div><div class="aegis-brand-name">AEGIS</div>
        <div class="aegis-brand-sub">Mobile Wallet Fraud Detection</div></div></div>
        <div class="aegis-side-desc">Machine-learning protection that scores mobile wallet
        transactions for fraud risk.</div>
        <div class="aegis-side-label">Navigation</div>
        """,
        unsafe_allow_html=True,
    )
 
    for i, (page_name, page_icon) in enumerate(PAGES):
        st.button(
            page_name,
            key=f"nav_{i}",
            icon=page_icon,
            type="primary" if st.session_state["nav"] == page_name else "secondary",
            on_click=go_to,
            args=(page_name,),
            width="stretch",
        )
 
    st.divider()
 
    with st.expander("Decision threshold", expanded=False):
        threshold = st.slider(
            "Fraud Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=default_threshold,
            step=0.05,
            help="A transaction is flagged as fraud when its probability "
                 "is greater than or equal to this value.",
        )
        st.caption(
            f"Model default: {default_threshold:.2f}. "
            "Higher values flag fewer transactions; lower values flag more."
        )
 
    st.markdown(
        f'<div class="aegis-status"><span class="aegis-dot"></span>'
        f"Model loaded - threshold {threshold:.2f}</div>",
        unsafe_allow_html=True,
    )
 
 
# =====================================================
# 5. Shared UI helpers
# =====================================================
def page_header(title: str, subtitle: str):
    st.markdown(
        f'<div class="aegis-page-title">{title}</div>'
        f'<div class="aegis-page-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )
 
 
def metric_card(key: str, label: str, value, accent: str = "blue", help_text: str | None = None):
    """A bordered card around a native st.metric (call inside a column)."""
    with st.container(key=f"card_{accent}_{key}"):
        st.metric(label, value, help=help_text)
 
 
def text_card(key: str, title: str, body: str, accent: str = "gray", number: int | None = None):
    with st.container(key=f"card_{accent}_{key}"):
        num = f'<div class="aegis-step-num">{number}</div>' if number else ""
        st.markdown(
            f'{num}<div class="aegis-card-title">{title}</div>'
            f'<div class="aegis-muted">{body}</div>',
            unsafe_allow_html=True,
        )
 
 
def result_card(is_fraud: bool, probability: float):
    """Result card that clearly distinguishes FRAUD from LEGITIMATE."""
    if is_fraud:
        css, badge, title = "fraud", "FRAUD", "Potential fraudulent transaction"
        text = (
            f"The model estimates a {probability:.2%} probability of fraud, which is at or above "
            f"the decision threshold of {threshold:.2f}. Review this transaction "
            "before approving it."
        )
    else:
        css, badge, title = "legit", "LEGITIMATE", "Transaction looks normal"
        text = (
            f"The model estimates a {probability:.2%} probability of fraud, which is below "
            f"the decision threshold of {threshold:.2f}. No strong fraud pattern was found. "
            "A prediction is an estimate, not a guarantee."
        )
    st.markdown(
        f'<div class="aegis-result {css}"><span class="aegis-result-badge">{badge}</span>'
        f'<div class="aegis-result-title">{title}</div>'
        f'<div class="aegis-result-text">{text}</div></div>',
        unsafe_allow_html=True,
    )
 
 
def probability_meter(probability: float, is_fraud: bool):
    """Horizontal meter with the decision threshold marked."""
    color = "#DC2626" if is_fraud else "#16A34A"
    st.markdown(
        f'<div class="aegis-card-title">Fraud probability vs. threshold</div>'
        f'<div class="aegis-meter-track">'
        f'<div class="aegis-meter-fill" style="width:{probability * 100:.1f}%; background:{color};"></div>'
        f'<div class="aegis-meter-mark" style="left:{threshold * 100:.1f}%;"></div></div>'
        f'<div class="aegis-meter-labels"><span>0%</span>'
        f"<span>Threshold {threshold:.0%}</span><span>100%</span></div>",
        unsafe_allow_html=True,
    )
 
 
PROBABILITY_COLUMN = {
    "Fraud_Probability": st.column_config.ProgressColumn(
        "Fraud probability", min_value=0.0, max_value=1.0, format="%.3f"
    )
}
 
 
# =====================================================
# 6. Pages
# =====================================================
def render_dashboard():
    with st.container(key="card_hero"):
        st.markdown(
            '<span class="aegis-pill"><span class="aegis-dot"></span>Model loaded and ready</span>'
            '<div class="aegis-hero-title">Welcome to AEGIS</div>'
            '<div class="aegis-hero-sub">AEGIS analyzes mobile wallet transactions with a trained '
            "Random Forest model and estimates the probability that each one is fraudulent. "
            "Check a single transaction or screen an entire CSV file in seconds.</div>",
            unsafe_allow_html=True,
        )
        b1, b2, _ = st.columns([1.2, 1.2, 2])
        b1.button("Analyze a transaction", type="primary", icon=":material/bolt:",
                  on_click=go_to, args=("Single Transaction Analysis",), width="stretch")
        b2.button("Upload a CSV file", icon=":material/upload_file:",
                  on_click=go_to, args=("Batch CSV Analysis",), width="stretch")
 
    st.write("")
    st.markdown("### System overview")
 
    summary = st.session_state.get("batch_summary")
 
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("m1", "Model status", "Active", "green")
    with c2:
        metric_card("m2", "Decision threshold", f"{threshold:.2f}", "blue",
                    "Probability at or above this value is flagged as fraud.")
    with c3:
        metric_card("m3", "Input features", len(features), "blue")
    with c4:
        metric_card("m4", "Analyzed this session",
                    f"{summary['total']:,}" if summary else "None yet", "gray",
                    "Total transactions in the latest batch you analyzed.")
 
    st.write("")
    if summary:
        st.markdown("### Latest batch result")
        st.caption(f"File: {summary['file']}  |  Threshold used: {summary['threshold']:.2f}")
        d1, d2, d3, d4 = st.columns(4)
        rate = summary["fraud"] / summary["total"] if summary["total"] else 0
        with d1:
            metric_card("b1", "Total transactions", f"{summary['total']:,}", "blue")
        with d2:
            metric_card("b2", "Predicted fraud", f"{summary['fraud']:,}", "red")
        with d3:
            metric_card("b3", "Legitimate", f"{summary['legit']:,}", "green")
        with d4:
            metric_card("b4", "Flagged share", f"{rate:.2%}", "gray")
    else:
        st.info("Batch statistics appear here after you analyze a CSV file. "
                "No figures are shown until real predictions exist.", icon=":material/info:")
 
    st.write("")
    st.markdown("### How detection works")
    s1, s2, s3 = st.columns(3)
    with s1:
        text_card("s1", "Provide transaction data",
                  "Enter one transaction manually or upload a CSV file with the required columns.",
                  "blue", 1)
    with s2:
        text_card("s2", "The model scores it",
                  "A Random Forest pipeline prepares the inputs and estimates a fraud probability "
                  "between 0% and 100%.", "blue", 2)
    with s3:
        text_card("s3", "The threshold decides",
                  "If the probability reaches the decision threshold, the transaction is flagged "
                  "as fraud for review.", "blue", 3)
 
 
def render_single():
    page_header("Single Transaction Analysis",
                "Enter the transaction details to estimate its fraud probability.")
 
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
 
        with col1:
            transaction_type = st.selectbox(
                "Transaction Type", TRANSACTION_TYPES,
                help="The kind of wallet operation being performed.")
            amount = st.number_input(
                "Transaction Amount", min_value=0.0, value=10000.0,
                help="Amount transferred in this transaction.")
            oldbalance_org = st.number_input(
                "Origin Account Balance Before Transaction", min_value=0.0, value=20000.0,
                help="Sender's balance before the transaction.")
 
        with col2:
            step = st.number_input(
                "Transaction Step", min_value=1, value=100, step=1,
                help="Time step of the transaction in the dataset's timeline.")
            oldbalance_dest = st.number_input(
                "Destination Account Balance Before Transaction", min_value=0.0, value=5000.0,
                help="Receiver's balance before the transaction.")
 
        submitted = st.form_submit_button(
            "Predict Transaction", type="primary", icon=":material/shield:", width="stretch"
        )
 
    if not submitted:
        st.caption("Fill in the fields above and select Predict Transaction to see the result.")
        return
 
    try:
        with st.spinner("Analyzing transaction..."):
            # Column order must match the order used during training
            transaction = pd.DataFrame(
                [[step, transaction_type, amount, oldbalance_org, oldbalance_dest]],
                columns=features,
            )
            probability = float(model.predict_proba(transaction)[0, 1])
            prediction = int(probability >= threshold)
    except Exception as error:
        st.error(f"The prediction could not be completed: {error}")
        return
 
    is_fraud = prediction == 1
 
    st.write("")
    st.markdown("### Prediction result")
    result_card(is_fraud, probability)
 
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("r1", "Fraud probability", f"{probability:.2%}", "red" if is_fraud else "green")
    with m2:
        metric_card("r2", "Decision threshold", f"{threshold:.2f}", "blue")
    with m3:
        metric_card("r3", "Classification", "FRAUD" if is_fraud else "LEGITIMATE",
                    "red" if is_fraud else "green")
 
    st.write("")
    with st.container(key="card_gray_meter"):
        probability_meter(min(max(probability, 0.0), 1.0), is_fraud)
 
    with st.expander("Submitted transaction details"):
        st.dataframe(transaction, hide_index=True, width="stretch")
 
 
def render_batch():
    page_header("Batch CSV Analysis",
                "Screen many transactions at once and download the predictions.")
 
    # Step overview + template
    left, right = st.columns([2, 1])
    with left:
        with st.container(key="card_blue_req"):
            st.markdown('<div class="aegis-card-title">Required columns</div>', unsafe_allow_html=True)
            st.code(", ".join(features), language=None)
            st.caption("Extra columns are allowed and are kept in the downloadable results.")
    with right:
        with st.container(key="card_gray_tpl"):
            st.markdown('<div class="aegis-card-title">Need a starting point?</div>',
                        unsafe_allow_html=True)
            st.caption("Download a one-row example file with the correct columns.")
            template = pd.DataFrame([{c: DEFAULTS.get(c, "") for c in features}])
            st.download_button(
                "Download sample template",
                data=template.to_csv(index=False).encode("utf-8"),
                file_name="aegis_sample_template.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )
 
    st.write("")
    uploaded_file = st.file_uploader("Upload Transactions CSV", type=["csv"])
 
    if uploaded_file is None:
        return
 
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
        return
    except Exception as error:
        st.error(f"The file could not be read as a CSV: {error}")
        return
 
    if uploaded_df.empty:
        st.error("The uploaded file has headers but no transaction rows.")
        return
 
    with st.expander(
        f"Preview of uploaded data ({len(uploaded_df):,} rows, {len(uploaded_df.columns)} columns)",
        expanded=True,
    ):
        st.dataframe(uploaded_df.head(10), width="stretch")
        st.caption("Showing the first 10 rows.")
 
    # ---- Validation (same rules as the original app) ----
    missing_features = [c for c in features if c not in uploaded_df.columns]
    if missing_features:
        st.error(f"Missing required columns: {missing_features}")
        return
 
    batch_data = uploaded_df[features].copy()
    for col in NUMERIC_FEATURES:
        batch_data[col] = pd.to_numeric(batch_data[col], errors="coerce")
 
    if batch_data.isnull().any().any():
        st.error("The uploaded file contains missing or invalid feature values.")
        return
 
    if (batch_data[NUMERIC_FEATURES] < 0).any().any():
        st.error("Numeric transaction features cannot be negative.")
        return
 
    # ---- Prediction ----
    try:
        with st.spinner("Analyzing transactions..."):
            probabilities = model.predict_proba(batch_data)[:, 1]
            predictions = (probabilities >= threshold).astype(int)
    except Exception as error:
        st.error(f"The predictions could not be completed: {error}")
        return
 
    results = uploaded_df.copy()
    results["Fraud_Probability"] = probabilities
    results["Prediction"] = ["Fraud" if p == 1 else "Normal" for p in predictions]
 
    total = len(results)
    fraud_count = int(predictions.sum())
    st.session_state["batch_summary"] = {
        "file": uploaded_file.name,
        "total": total,
        "fraud": fraud_count,
        "legit": total - fraud_count,
        "threshold": threshold,
    }
 
    st.success("Batch analysis completed!", icon=":material/check_circle:")
 
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("t1", "Total transactions", f"{total:,}", "blue")
    with m2:
        metric_card("t2", "Predicted fraud cases", f"{fraud_count:,}", "red")
    with m3:
        metric_card("t3", "Legitimate cases", f"{total - fraud_count:,}", "green")
    with m4:
        metric_card("t4", "Flagged share", f"{fraud_count / total:.2%}", "gray")
 
    st.write("")
    st.markdown("### Prediction results")
    tab_all, tab_flagged, tab_top = st.tabs(
        ["All results", f"Flagged as fraud ({fraud_count:,})", "Top 10 by risk"]
    )
 
    with tab_all:
        st.caption(f"Showing up to 100 of {total:,} rows. The download contains all rows.")
        st.dataframe(results.head(100), column_config=PROBABILITY_COLUMN, width="stretch")
 
    with tab_flagged:
        flagged = results[results["Prediction"] == "Fraud"]
        if flagged.empty:
            st.info("No transactions were flagged at the current threshold.")
        else:
            st.caption(f"Showing up to 100 of {len(flagged):,} flagged rows.")
            st.dataframe(flagged.head(100), column_config=PROBABILITY_COLUMN, width="stretch")
 
    with tab_top:
        st.caption("The ten transactions with the highest fraud probability.")
        st.dataframe(
            results.sort_values("Fraud_Probability", ascending=False).head(10),
            column_config=PROBABILITY_COLUMN,
            width="stretch",
        )
 
    csv_output = results.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download Prediction Results",
        data=csv_output,
        file_name="fraud_detection_results.csv",
        mime="text/csv",
        type="primary",
        icon=":material/download:",
    )
 
 
def render_about():
    page_header("About the Model",
                "How AEGIS makes its decisions and how well it performed.")
 
    c1, c2 = st.columns(2)
    with c1:
        text_card("a1", "Algorithm",
                  "Random Forest Classifier inside a preprocessing pipeline.", "blue")
    with c2:
        text_card("a2", "Problem type",
                  "Imbalanced binary classification: fraud is rare compared with legitimate transactions.",
                  "teal")
 
    st.write("")
    st.markdown("### Reported evaluation metrics")
    st.caption("Results reported by the project; they are not recalculated by this app.")
    cols = st.columns(4)
    for col, (name, value, accent), idx in zip(cols, REPORTED_METRICS, range(4)):
        with col:
            metric_card(f"met{idx}", name, value, accent)
 
    st.write("")
    st.markdown("### What the metrics mean for fraud detection")
    p, r = st.columns(2)
    with p:
        text_card("prec", "Precision: how trustworthy an alert is",
                  "Of all transactions the model flags as fraud, precision is the share that really "
                  "are fraud. A precision of 89.39% means roughly 89 of every 100 alerts are genuine, "
                  "so few legitimate customers are disturbed by false alarms.", "blue")
    with r:
        text_card("rec", "Recall: how much fraud is caught",
                  "Of all real fraud cases, recall is the share the model detects. A recall of 60.50% "
                  "means roughly 60 of every 100 fraudulent transactions are caught, and the rest "
                  "are missed.", "teal")
 
    st.write("")
    st.info(
        "F1-score balances precision and recall. Average Precision summarizes the "
        "precision-recall trade-off across all thresholds. The strict decision threshold "
        f"(default {default_threshold:.2f}) favors precision: fewer false alarms, at the cost of "
        "missing some fraud. Lowering it in the sidebar catches more fraud but raises false alarms.",
        icon=":material/lightbulb:",
    )
 
    st.write("")
    st.markdown("### Model inputs")
    st.caption("The model expects these features, in this order.")
    feature_table = pd.DataFrame(
        {
            "Order": range(1, len(features) + 1),
            "Feature": features,
            "Kind": ["Numeric" if f in NUMERIC_FEATURES else "Categorical" for f in features],
            "Meaning": [FEATURE_INFO.get(f, "") for f in features],
        }
    )
    st.dataframe(feature_table, hide_index=True, width="stretch")
 
 
# =====================================================
# 7. Router
# =====================================================
PAGE_RENDERERS = {
    "Dashboard": render_dashboard,
    "Single Transaction Analysis": render_single,
    "Batch CSV Analysis": render_batch,
    "About the Model": render_about,
}
 
PAGE_RENDERERS[st.session_state["nav"]]()
 
