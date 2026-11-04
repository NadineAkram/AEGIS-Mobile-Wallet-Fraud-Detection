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
# Palette: Navy #0B1220 | Dark blue #111C30 | Blue #3B82F6 | Teal #14B8A6
#          Red #EF4444 | Green #22C55E | Text #F1F5F9 / #94A3B8
CUSTOM_CSS = """
<style>
.stApp { background-color: #0B1220; color: #F1F5F9; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111C30;
    border-right: 1px solid #1F2D4A;
}
.aegis-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.aegis-brand-name { font-size: 1.35rem; font-weight: 700; color: #F1F5F9; line-height: 1.1; }
.aegis-brand-sub { color: #94A3B8; font-size: 0.82rem; }
.aegis-desc { color: #94A3B8; font-size: 0.88rem; margin: 10px 0 18px 0; }

/* Cards: any container created with key="card_*" */
[class*="st-key-card"] {
    background-color: #111C30;
    border: 1px solid #1F2D4A;
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
}
[data-testid="stForm"] {
    background-color: #111C30;
    border: 1px solid #1F2D4A;
    border-radius: 14px;
    padding: 1.25rem;
}

/* Metrics */
[data-testid="stMetricLabel"] p { color: #94A3B8; font-size: 0.85rem; }
[data-testid="stMetricValue"] { color: #F1F5F9; font-weight: 700; }

/* Buttons */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background-color: #3B82F6; color: #FFFFFF; border: none;
    border-radius: 8px; font-weight: 600; padding: 0.55rem 1.1rem;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    background-color: #2563EB; color: #FFFFFF; border: none;
}

/* Result cards */
.aegis-result { border-radius: 14px; padding: 1.25rem 1.4rem; border: 1px solid; margin: 0.5rem 0 1rem 0; }
.aegis-result.fraud { background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.55); }
.aegis-result.legit { background: rgba(34,197,94,0.10); border-color: rgba(34,197,94,0.55); }
.aegis-result-title { font-size: 1.4rem; font-weight: 700; margin: 0; }
.aegis-result.fraud .aegis-result-title { color: #EF4444; }
.aegis-result.legit .aegis-result-title { color: #22C55E; }
.aegis-result-text { color: #CBD5E1; margin: 6px 0 0 0; font-size: 0.95rem; }

/* Typography helpers */
.aegis-hero-title { font-size: 2.2rem; font-weight: 700; margin: 0; color: #F1F5F9; }
.aegis-hero-sub { color: #94A3B8; font-size: 1.02rem; margin-top: 6px; }
.aegis-muted { color: #94A3B8; }
.aegis-pill {
    display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 0.8rem;
    background: rgba(20,184,166,0.12); color: #14B8A6; border: 1px solid rgba(20,184,166,0.4);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

LOGO_SVG = """
<svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2.5 4.5 5.5v5.8c0 4.5 3.1 8.6 7.5 10.2 4.4-1.6 7.5-5.7 7.5-10.2V5.5L12 2.5Z"
 fill="#3B82F6" fill-opacity="0.18" stroke="#3B82F6" stroke-width="1.6" stroke-linejoin="round"/>
<path d="m8.6 12.2 2.4 2.4 4.4-4.8" stroke="#14B8A6" stroke-width="1.8"
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

# Project-reported evaluation results (provided by the project author)
REPORTED_METRICS = {
    "Precision": "89.39%",
    "Recall": "60.50%",
    "F1-score": "72.05%",
    "Average Precision": "0.7424",
}

PAGES = [
    "Dashboard",
    "Single Transaction Analysis",
    "Batch CSV Analysis",
    "About the Model",
]

if "nav" not in st.session_state:
    st.session_state["nav"] = PAGES[0]


def go_to(page: str):
    """Button callback used to jump between pages."""
    st.session_state["nav"] = page


# =====================================================
# 4. Sidebar: brand, navigation, decision threshold
# =====================================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="aegis-brand">{LOGO_SVG}
        <div><div class="aegis-brand-name">AEGIS</div>
        <div class="aegis-brand-sub">Mobile Wallet Fraud Detection</div></div></div>
        <div class="aegis-desc">Machine-learning protection that scores mobile wallet
        transactions for fraud risk.</div>
        """,
        unsafe_allow_html=True,
    )

    st.radio(
        "Navigation",
        PAGES,
        key="nav",
        label_visibility="collapsed",
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


# =====================================================
# 5. Shared helpers
# =====================================================
def result_card(is_fraud: bool, probability: float):
    """Colored card that clearly distinguishes FRAUD from LEGITIMATE."""
    if is_fraud:
        css, title = "fraud", "FRAUD - Potential fraudulent transaction"
        text = (
            f"The model estimates a {probability:.2%} probability of fraud, which is at or above "
            f"the decision threshold of {threshold:.2f}. This transaction should be reviewed "
            "before it is approved."
        )
    else:
        css, title = "legit", "LEGITIMATE - Transaction looks normal"
        text = (
            f"The model estimates a {probability:.2%} probability of fraud, which is below "
            f"the decision threshold of {threshold:.2f}. No strong fraud pattern was found. "
            "A prediction is an estimate, not a guarantee."
        )
    st.markdown(
        f'<div class="aegis-result {css}"><p class="aegis-result-title">{title}</p>'
        f'<p class="aegis-result-text">{text}</p></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str):
    st.markdown(f'<p class="aegis-hero-title">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="aegis-hero-sub">{subtitle}</p>', unsafe_allow_html=True)
    st.write("")


# =====================================================
# 6. Pages
# =====================================================
def render_dashboard():
    with st.container(key="card_hero"):
        st.markdown('<span class="aegis-pill">Model loaded and ready</span>', unsafe_allow_html=True)
        st.markdown('<p class="aegis-hero-title" style="margin-top:10px;">Welcome to AEGIS</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p class="aegis-hero-sub">AEGIS analyzes mobile wallet transactions with a trained '
            "Random Forest model and estimates the probability that each one is fraudulent. "
            "Check a single transaction or screen an entire CSV file in seconds.</p>",
            unsafe_allow_html=True,
        )
        b1, b2, _ = st.columns([1, 1, 2])
        b1.button("Analyze a transaction", on_click=go_to,
                  args=("Single Transaction Analysis",), width="stretch")
        b2.button("Upload a CSV file", on_click=go_to,
                  args=("Batch CSV Analysis",), width="stretch")

    st.write("")
    st.subheader("System overview")

    summary = st.session_state.get("batch_summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1, st.container(key="card_m1"):
        st.metric("Model status", "Active")
    with c2, st.container(key="card_m2"):
        st.metric("Decision threshold", f"{threshold:.2f}")
    with c3, st.container(key="card_m3"):
        st.metric("Input features", len(features))
    with c4, st.container(key="card_m4"):
        st.metric("Transactions analyzed (this session)",
                  f"{summary['total']:,}" if summary else "None yet")

    if summary:
        st.write("")
        st.subheader("Latest batch result")
        st.caption(f"File: {summary['file']} - threshold used: {summary['threshold']:.2f}")
        d1, d2, d3, d4 = st.columns(4)
        with d1, st.container(key="card_b1"):
            st.metric("Total transactions", f"{summary['total']:,}")
        with d2, st.container(key="card_b2"):
            st.metric("Predicted fraud", f"{summary['fraud']:,}")
        with d3, st.container(key="card_b3"):
            st.metric("Legitimate", f"{summary['legit']:,}")
        with d4, st.container(key="card_b4"):
            rate = summary["fraud"] / summary["total"] if summary["total"] else 0
            st.metric("Flagged share", f"{rate:.2%}")
    else:
        st.caption("Batch statistics appear here after you analyze a CSV file. "
                   "No figures are shown until real predictions exist.")

    st.write("")
    st.subheader("How detection works")
    s1, s2, s3 = st.columns(3)
    steps = [
        ("1. Provide transaction data",
         "Enter one transaction manually or upload a CSV file containing the required columns."),
        ("2. The model scores it",
         "A Random Forest pipeline converts the inputs and estimates a fraud probability between 0% and 100%."),
        ("3. The threshold decides",
         "If the probability reaches the decision threshold, the transaction is flagged as fraud for review."),
    ]
    for col, (title, text), idx in zip((s1, s2, s3), steps, range(3)):
        with col, st.container(key=f"card_step{idx}"):
            st.markdown(f"**{title}**")
            st.markdown(f'<span class="aegis-muted">{text}</span>', unsafe_allow_html=True)


def render_single():
    page_header("Single Transaction Analysis",
                "Enter the transaction details to estimate its fraud probability.")

    with st.form("transaction_form"):
        col1, col2 = st.columns(2)

        with col1:
            transaction_type = st.selectbox("Transaction Type", TRANSACTION_TYPES)
            amount = st.number_input(
                "Transaction Amount", min_value=0.0, value=10000.0,
                help="Amount transferred in this transaction.")
            oldbalance_org = st.number_input(
                "Origin Account Balance Before Transaction", min_value=0.0, value=20000.0,
                help="Sender's balance before the transaction.")

        with col2:
            step = st.number_input(
                "Transaction Step", min_value=1, value=100, step=1,
                help="Time step (hour) of the transaction in the dataset's timeline.")
            oldbalance_dest = st.number_input(
                "Destination Account Balance Before Transaction", min_value=0.0, value=5000.0,
                help="Receiver's balance before the transaction.")

        submitted = st.form_submit_button("Predict Transaction", width="stretch")

    if not submitted:
        return

    try:
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

    st.subheader("Prediction result")
    result_card(prediction == 1, probability)

    m1, m2, m3 = st.columns(3)
    with m1, st.container(key="card_r1"):
        st.metric("Fraud probability", f"{probability:.2%}")
    with m2, st.container(key="card_r2"):
        st.metric("Decision threshold", f"{threshold:.2f}")
    with m3, st.container(key="card_r3"):
        st.metric("Classification", "FRAUD" if prediction == 1 else "LEGITIMATE")

    st.progress(min(max(probability, 0.0), 1.0))


def render_batch():
    page_header("Batch CSV Analysis",
                "Upload a CSV file to screen many transactions at once.")

    with st.container(key="card_req"):
        st.markdown("**Required columns**")
        st.code(", ".join(features), language=None)
        st.caption("Extra columns are allowed and are kept in the downloadable results.")

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

    st.subheader("Data preview")
    st.caption(f"{len(uploaded_df):,} rows, {len(uploaded_df.columns)} columns. Showing the first 10 rows.")
    st.dataframe(uploaded_df.head(10), width="stretch")

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

    st.success("Batch analysis completed!")

    m1, m2, m3 = st.columns(3)
    with m1, st.container(key="card_t1"):
        st.metric("Total transactions", f"{total:,}")
    with m2, st.container(key="card_t2"):
        st.metric("Predicted fraud cases", f"{fraud_count:,}")
    with m3, st.container(key="card_t3"):
        st.metric("Legitimate cases", f"{total - fraud_count:,}")

    st.write("")
    st.subheader("Prediction results")
    only_fraud = st.toggle("Show only flagged transactions", value=False)
    view = results[results["Prediction"] == "Fraud"] if only_fraud else results
    st.caption(f"Showing up to 100 of {len(view):,} rows. The download contains all rows.")
    st.dataframe(view.head(100), width="stretch")

    csv_output = results.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download Prediction Results",
        data=csv_output,
        file_name="fraud_detection_results.csv",
        mime="text/csv",
    )


def render_about():
    page_header("About the Model",
                "How AEGIS makes its decisions and how well it performed.")

    c1, c2 = st.columns(2)
    with c1, st.container(key="card_about1"):
        st.markdown("**Algorithm**")
        st.write("Random Forest Classifier inside a preprocessing pipeline.")
    with c2, st.container(key="card_about2"):
        st.markdown("**Problem type**")
        st.write("Imbalanced binary classification: fraud is rare compared with legitimate transactions.")

    st.write("")
    st.subheader("Reported evaluation metrics")
    st.caption("Results reported by the project; they are not recalculated by this app.")
    cols = st.columns(4)
    for col, (name, value), idx in zip(cols, REPORTED_METRICS.items(), range(4)):
        with col, st.container(key=f"card_metric{idx}"):
            st.metric(name, value)

    st.write("")
    st.subheader("What the metrics mean for fraud detection")
    p, r = st.columns(2)
    with p, st.container(key="card_prec"):
        st.markdown("**Precision: how trustworthy an alert is**")
        st.write(
            "Of all transactions the model flags as fraud, precision is the share that really "
            "are fraud. A precision of 89.39% means roughly 89 of every 100 alerts are genuine, "
            "so few legitimate customers are disturbed by false alarms."
        )
    with r, st.container(key="card_rec"):
        st.markdown("**Recall: how much fraud is caught**")
        st.write(
            "Of all real fraud cases, recall is the share the model detects. A recall of 60.50% "
            "means roughly 60 of every 100 fraudulent transactions are caught, and the rest are missed."
        )
    st.info(
        "F1-score balances precision and recall. Average Precision summarizes the "
        "precision-recall trade-off across all thresholds. The strict decision threshold "
        f"(default {default_threshold:.2f}) favors precision: fewer false alarms, at the cost of "
        "missing some fraud. Lowering it in the sidebar catches more fraud but raises false alarms."
    )

    st.write("")
    st.subheader("Model inputs")
    st.write("The model expects these features, in this order:")
    st.code(", ".join(features), language=None)


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
