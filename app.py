import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from html import escape
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

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

/* ---------- Page header & text helpers ---------- */
.aegis-page-title { font-size: 1.9rem; font-weight: 700; color: #0F172A; line-height: 1.2; }
.aegis-page-sub { color: #64748B; font-size: 1rem; margin: 6px 0 18px 0; max-width: 760px; }
.aegis-hero-title { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin: 10px 0 0 0; line-height: 1.15; }
.aegis-hero-sub { color: #475569; font-size: 1.05rem; margin: 10px 0 18px 0; max-width: 720px; line-height: 1.55; }
.aegis-pill {
    display: inline-flex; align-items: center; gap: 8px; padding: 4px 12px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 500; background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0;
}
.aegis-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 6px 0; }
.aegis-chip {
    display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.8rem;
    background: #F1F5F9; color: #334155; border: 1px solid #E2E8F0;
}
.aegis-chip.blue  { background: #EAF1FF; color: #1D4ED8; border-color: #BFD3FB; }
.aegis-chip.green { background: #ECFDF5; color: #047857; border-color: #A7F3D0; }
.aegis-chip.amber { background: #FFFBEB; color: #92400E; border-color: #FDE68A; }
.aegis-muted { color: #64748B; font-size: 0.93rem; line-height: 1.5; }
.aegis-step-num {
    width: 30px; height: 30px; border-radius: 50%; background: #EAF1FF; color: #1D4ED8;
    display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 10px;
}
.aegis-card-title { font-weight: 600; color: #0F172A; margin-bottom: 4px; }
.aegis-empty { text-align: center; padding: 1.6rem 1rem 0.6rem 1rem; }
.aegis-empty-title { font-size: 1.2rem; font-weight: 650; color: #0F172A; margin: 6px 0; }
.aegis-empty-text { color: #64748B; max-width: 560px; margin: 0 auto 12px auto; line-height: 1.55; }

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
CATEGORICAL_FEATURES = [c for c in features if c not in NUMERIC_FEATURES]
TYPE_COL = CATEGORICAL_FEATURES[0] if CATEGORICAL_FEATURES else None
STEP_COL = "step" if "step" in features else None
AMOUNT_COL = "amount" if "amount" in features else None
TRANSACTION_TYPES = ["CASH_OUT", "TRANSFER", "CASH_IN", "PAYMENT", "DEBIT"]

# Optional ground-truth column. Used ONLY for evaluation, never as a model input.
LABEL_COL = "isFraud"

# Risk bands (fixed, for review prioritisation only - NOT the classification threshold)
LOW_MAX = 0.30
MEDIUM_MAX = 0.70
RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]

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
# (name, numeric value, display text, accent)
REPORTED_METRICS = [
    ("Precision", 0.8939, "89.39%", "blue"),
    ("Recall", 0.6050, "60.50%", "teal"),
    ("F1-score", 0.7205, "72.05%", "blue"),
    ("Average Precision", 0.7424, "0.7424", "teal"),
]

PAGES = [
    ("Dashboard", ":material/dashboard:"),
    ("Single Transaction Analysis", ":material/receipt_long:"),
    ("Batch CSV Analysis", ":material/upload_file:"),
    ("Fraud Analytics", ":material/analytics:"),
    ("About the Model", ":material/info:"),
]
PAGE_NAMES = [name for name, _ in PAGES]

# Chart colors
BLUE, TEAL, RED, GREEN, SLATE = "#2563EB", "#14B8A6", "#EF4444", "#16A34A", "#64748B"
CLASS_COLORS = {"Fraud": RED, "Legitimate": GREEN}
RISK_COLORS = {"Low Risk": GREEN, "Medium Risk": BLUE, "High Risk": RED}
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

if "nav" not in st.session_state:
    st.session_state["nav"] = PAGE_NAMES[0]


def go_to(page: str):
    """Button callback used to switch pages."""
    st.session_state["nav"] = page


def clear_analysis():
    """Button callback: remove the stored analysis data from this session."""
    st.session_state.pop("analysis", None)


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


def empty_state(key: str, title: str, text: str, button_label: str | None = None,
                target_page: str | None = None):
    """Professional empty state card with an optional call-to-action button."""
    with st.container(key=f"card_gray_empty_{key}"):
        st.markdown(
            f'<div class="aegis-empty"><div style="font-size:2rem;">📂</div>'
            f'<div class="aegis-empty-title">{title}</div>'
            f'<div class="aegis-empty-text">{text}</div></div>',
            unsafe_allow_html=True,
        )
        if button_label and target_page:
            _, mid, _ = st.columns([1, 1, 1])
            mid.button(button_label, key=f"empty_btn_{key}", type="primary",
                       icon=":material/upload_file:", on_click=go_to,
                       args=(target_page,), width="stretch")


def chips(items: list[tuple[str, str]]):
    """Small informational pills: list of (text, color)."""
    body = "".join(f'<span class="aegis-chip {c}">{escape(t)}</span>' for t, c in items)
    st.markdown(f'<div class="aegis-chips">{body}</div>', unsafe_allow_html=True)


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
# 6. Analytics data layer (session state -> view)
# =====================================================
def build_view(analysis: dict, thr: float) -> pd.DataFrame:
    """
    Builds the analytics DataFrame from the stored model probabilities.
    Probabilities are the model's actual output; the classification uses the
    CURRENT decision threshold, so no re-prediction is needed when it changes.
    """
    v = analysis["frame"].copy()
    v["Predicted"] = (v["Fraud_Probability"] >= thr).astype(int)
    v["Classification"] = np.where(v["Predicted"] == 1, "Fraud", "Legitimate")
    v["Risk"] = pd.cut(
        v["Fraud_Probability"],
        bins=[-0.001, LOW_MAX, MEDIUM_MAX, 1.001],
        labels=RISK_ORDER,
        right=False,
    ).astype(str)
    return v


def type_stats(v: pd.DataFrame) -> pd.DataFrame:
    g = (
        v.groupby(TYPE_COL)
        .agg(Transactions=("Predicted", "size"), Fraud=("Predicted", "sum"))
        .reset_index()
    )
    g["Rate"] = g["Fraud"] / g["Transactions"]
    return g


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    """Consistent AEGIS look for every Plotly chart."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color="#0F172A"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=13),
    )
    fig.update_xaxes(gridcolor="#E2E8F0", zeroline=False, linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    return fig


def chart_card(key: str, title: str, fig: go.Figure, caption: str | None = None):
    with st.container(key=f"card_gray_{key}"):
        st.markdown(f'<div class="aegis-card-title">{title}</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG, key=f"plt_{key}")
        if caption:
            st.caption(caption)


def note_card(key: str, title: str, message: str):
    """Placeholder inside a chart slot when a chart cannot be drawn."""
    with st.container(key=f"card_gray_{key}"):
        st.markdown(f'<div class="aegis-card-title">{title}</div>', unsafe_allow_html=True)
        st.info(message)


# ---------- Individual figures ----------
def fig_outcome_donut(v: pd.DataFrame) -> go.Figure:
    fraud = int(v["Predicted"].sum())
    legit = len(v) - fraud
    fig = go.Figure(go.Pie(
        labels=["Fraud", "Legitimate"], values=[fraud, legit], hole=0.62, sort=False,
        marker=dict(colors=[RED, GREEN], line=dict(color="#FFFFFF", width=2)),
        textinfo="percent", hovertemplate="%{label}: %{value:,} transactions (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{len(v):,}</b><br>transactions", showarrow=False, font=dict(size=15))
    return style_fig(fig, 320)


def fig_cases_by_type(g: pd.DataFrame) -> go.Figure:
    g = g.sort_values("Fraud", ascending=False)
    fig = go.Figure(go.Bar(
        x=g[TYPE_COL], y=g["Fraud"], marker_color=RED, text=g["Fraud"], textposition="outside",
        cliponaxis=False, customdata=g["Transactions"],
        hovertemplate="%{x}<br>Predicted fraud: %{y:,}<br>Transactions: %{customdata:,}<extra></extra>",
    ))
    fig.update_xaxes(title="Transaction type")
    fig.update_yaxes(title="Predicted fraud cases", range=[0, max(int(g["Fraud"].max()), 1) * 1.18])
    return style_fig(fig, 320)


def fig_rate_by_type(g: pd.DataFrame) -> go.Figure:
    g = g.sort_values("Rate", ascending=False)
    pct = g["Rate"] * 100
    fig = go.Figure(go.Bar(
        x=g[TYPE_COL], y=pct, marker_color=TEAL, text=[f"{x:.2f}%" for x in pct],
        textposition="outside", cliponaxis=False,
        customdata=np.stack([g["Fraud"], g["Transactions"]], axis=-1),
        hovertemplate="%{x}<br>Fraud rate: %{y:.2f}%<br>Flagged: %{customdata[0]:,} of "
                      "%{customdata[1]:,}<extra></extra>",
    ))
    fig.update_xaxes(title="Transaction type")
    fig.update_yaxes(title="Fraud rate (%)", range=[0, max(float(pct.max()), 1.0) * 1.2])
    return style_fig(fig, 320)


def fig_avg_amount(v: pd.DataFrame) -> go.Figure:
    g = (
        v.groupby("Classification")[AMOUNT_COL]
        .agg(["mean", "size"])
        .reindex(["Fraud", "Legitimate"])
        .dropna()
        .reset_index()
    )
    fig = go.Figure(go.Bar(
        x=g["Classification"], y=g["mean"],
        marker_color=[CLASS_COLORS[c] for c in g["Classification"]],
        text=[f"{m:,.0f}" for m in g["mean"]], textposition="outside", cliponaxis=False,
        customdata=g["size"],
        hovertemplate="%{x}<br>Average amount: %{y:,.2f}<br>Transactions: %{customdata:,}<extra></extra>",
    ))
    fig.update_xaxes(title="Predicted class")
    fig.update_yaxes(title="Average transaction amount", range=[0, float(g["mean"].max()) * 1.2 or 1])
    return style_fig(fig, 320)


def fig_activity(v: pd.DataFrame) -> go.Figure:
    a = (
        v.groupby(STEP_COL)
        .agg(Transactions=("Predicted", "size"), Flagged=("Predicted", "sum"))
        .reset_index()
        .sort_values(STEP_COL)
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=a[STEP_COL], y=a["Transactions"], name="Transactions", mode="lines",
        line=dict(color=BLUE, width=2),
        hovertemplate="Step %{x}<br>Transactions: %{y:,}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=a[STEP_COL], y=a["Flagged"], name="Predicted fraud", mode="lines",
        line=dict(color=RED, width=2),
        hovertemplate="Step %{x}<br>Predicted fraud: %{y:,}<extra></extra>"), secondary_y=True)
    fig.update_xaxes(title="Transaction step")
    fig.update_yaxes(title_text="Transactions", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="Predicted fraud", secondary_y=True, rangemode="tozero", showgrid=False)
    return style_fig(fig, 360)


def fig_amount_hist(v: pd.DataFrame, log_scale: bool) -> go.Figure:
    d = v[[AMOUNT_COL, "Classification"]].copy()
    if log_scale:
        d = d[d[AMOUNT_COL] > 0]
        d["x"] = np.log10(d[AMOUNT_COL])
        x_title = "log10(Transaction amount)"
    else:
        d["x"] = d[AMOUNT_COL]
        x_title = "Transaction amount"
    fig = px.histogram(
        d, x="x", color="Classification", nbins=40, barmode="stack",
        color_discrete_map=CLASS_COLORS, category_orders={"Classification": ["Legitimate", "Fraud"]},
    )
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="Transactions")
    return style_fig(fig, 340)


def fig_prob_hist(v: pd.DataFrame, thr: float) -> go.Figure:
    fig = px.histogram(
        v, x="Fraud_Probability", color="Classification", barmode="stack",
        color_discrete_map=CLASS_COLORS, category_orders={"Classification": ["Legitimate", "Fraud"]},
    )
    fig.update_traces(xbins=dict(start=0, end=1, size=0.05))
    fig.add_vline(x=thr, line_dash="dash", line_color="#0F172A",
                  annotation_text=f"Threshold {thr:.2f}", annotation_position="top")
    fig.update_xaxes(title="Fraud probability", range=[0, 1])
    fig.update_yaxes(title="Transactions")
    return style_fig(fig, 340)


MAX_SCATTER_POINTS = 5000


def fig_scatter(v: pd.DataFrame, thr: float, log_scale: bool):
    d = v[v[AMOUNT_COL] > 0] if log_scale else v
    flagged = d[d["Predicted"] == 1].head(MAX_SCATTER_POINTS // 2)
    others = d[d["Predicted"] == 0]
    others = others.sample(min(len(others), MAX_SCATTER_POINTS - len(flagged)), random_state=0)
    pts = pd.concat([flagged, others])
    hover = [c for c in [TYPE_COL, STEP_COL] if c]
    fig = px.scatter(
        pts, x=AMOUNT_COL, y="Fraud_Probability", color="Classification",
        color_discrete_map=CLASS_COLORS, hover_data=hover, opacity=0.65, log_x=log_scale,
        category_orders={"Classification": ["Legitimate", "Fraud"]},
    )
    fig.add_hline(y=thr, line_dash="dash", line_color="#0F172A",
                  annotation_text=f"Threshold {thr:.2f}", annotation_position="top left")
    fig.update_xaxes(title="Transaction amount" + (" (log scale)" if log_scale else ""))
    fig.update_yaxes(title="Fraud probability", range=[-0.02, 1.02])
    return style_fig(fig, 380), len(pts), len(d)


def fig_risk_donut(v: pd.DataFrame) -> go.Figure:
    counts = v["Risk"].value_counts().reindex(RISK_ORDER, fill_value=0)
    fig = go.Figure(go.Pie(
        labels=RISK_ORDER, values=counts.values, hole=0.62, sort=False,
        marker=dict(colors=[RISK_COLORS[r] for r in RISK_ORDER], line=dict(color="#FFFFFF", width=2)),
        textinfo="percent", hovertemplate="%{label}: %{value:,} transactions (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{len(v):,}</b><br>transactions", showarrow=False, font=dict(size=15))
    return style_fig(fig, 320)


def threshold_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Predicted fraud counts (and, with labels, precision/recall) for a grid of thresholds."""
    p = frame["Fraud_Probability"].to_numpy()
    y = frame[LABEL_COL].to_numpy() if LABEL_COL in frame.columns else None
    rows = []
    for t in np.round(np.arange(0.05, 1.0001, 0.05), 2):
        pred = p >= t
        row = {"Threshold": float(t), "Predicted fraud": int(pred.sum()),
               "Flagged share": float(pred.mean())}
        if y is not None:
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            fn = int((~pred & (y == 1)).sum())
            row["Precision"] = tp / (tp + fp) if (tp + fp) else np.nan
            row["Recall"] = tp / (tp + fn) if (tp + fn) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================
# 7. Pages
# =====================================================
def render_dashboard():
    analysis = st.session_state.get("analysis")

    with st.container(key="card_hero"):
        st.markdown(
            '<span class="aegis-pill"><span class="aegis-dot"></span>Model loaded and ready</span>'
            '<div class="aegis-hero-title">Welcome to AEGIS</div>'
            '<div class="aegis-hero-sub">AEGIS analyzes mobile wallet transactions with a trained '
            "Random Forest model and estimates the probability that each one is fraudulent. "
            "Check a single transaction or screen an entire CSV file in seconds.</div>",
            unsafe_allow_html=True,
        )
        b1, b2, b3 = st.columns(3)
        b1.button("Analyze a transaction", type="primary", icon=":material/bolt:",
                  on_click=go_to, args=("Single Transaction Analysis",), width="stretch")
        b2.button("Upload a CSV file", icon=":material/upload_file:",
                  on_click=go_to, args=("Batch CSV Analysis",), width="stretch")
        b3.button("Open Fraud Analytics", icon=":material/analytics:",
                  on_click=go_to, args=("Fraud Analytics",), width="stretch")

    st.write("")
    st.markdown("### System overview")

    v = build_view(analysis, threshold) if analysis else None

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
                    f"{len(v):,}" if v is not None else "None yet", "gray",
                    "Total transactions in the latest CSV you analyzed.")

    st.write("")
    if v is not None:
        fraud = int(v["Predicted"].sum())
        st.markdown("### Latest batch result")
        st.caption(f"File: {analysis['file']}  |  Threshold: {threshold:.2f}")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            metric_card("b1", "Total transactions", f"{len(v):,}", "blue")
        with d2:
            metric_card("b2", "Predicted fraud", f"{fraud:,}", "red")
        with d3:
            metric_card("b3", "Legitimate", f"{len(v) - fraud:,}", "green")
        with d4:
            metric_card("b4", "Flagged share", f"{fraud / len(v):.2%}", "gray")
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

    left, right = st.columns([2, 1])
    with left:
        with st.container(key="card_blue_req"):
            st.markdown('<div class="aegis-card-title">Required columns</div>', unsafe_allow_html=True)
            st.code(", ".join(features), language=None)
            st.caption(
                "Extra columns are allowed and are kept in the downloadable results. "
                f"An optional `{LABEL_COL}` column (1 = fraud, 0 = legitimate) is used only to "
                "evaluate the model, never as an input."
            )
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

    # Any previous analysis is replaced by the newest upload (never show stale data)
    st.session_state.pop("analysis", None)

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

    # ---- Optional ground-truth labels (evaluation only) ----
    actual = None
    label_note = None
    if LABEL_COL in uploaded_df.columns and LABEL_COL not in features:
        raw = uploaded_df[LABEL_COL]
        lab = raw.astype(int) if pd.api.types.is_bool_dtype(raw) else pd.to_numeric(raw, errors="coerce")
        if lab.isnull().any() or not lab.isin([0, 1]).all():
            label_note = (f"The `{LABEL_COL}` column was ignored because it must contain only "
                          "0 and 1 values.")
        else:
            actual = lab.astype(int)

    # ---- Prediction (features only, original order) ----
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

    # ---- Store latest results for the Analytics / About pages ----
    frame = batch_data.copy()
    frame["Fraud_Probability"] = probabilities
    if actual is not None:
        frame[LABEL_COL] = actual.to_numpy()
    st.session_state["analysis"] = {
        "file": uploaded_file.name,
        "frame": frame,
        "has_labels": actual is not None,
    }

    st.success("Batch analysis completed!", icon=":material/check_circle:")
    if actual is not None:
        st.info(f"`{LABEL_COL}` detected. It is used only to evaluate the model on the "
                "About the Model page and is never passed to the model.", icon=":material/fact_check:")
    if label_note:
        st.warning(label_note)

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
    st.button("Open Fraud Analytics", icon=":material/analytics:", on_click=go_to,
              args=("Fraud Analytics",), key="open_analytics")

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


def render_analytics():
    page_header("Fraud Analytics",
                "Interactive insights from your latest uploaded CSV and the model's predictions.")

    analysis = st.session_state.get("analysis")
    if analysis is None:
        empty_state(
            "analytics",
            "No data to analyze yet",
            "Upload a transaction CSV on the Batch CSV Analysis page. Charts and metrics here are "
            "calculated only from your uploaded data and the model's actual predictions.",
            "Go to Batch CSV Analysis", "Batch CSV Analysis",
        )
        return

    v = build_view(analysis, threshold)
    total = len(v)
    fraud = int(v["Predicted"].sum())
    legit = total - fraud
    high_risk = int((v["Risk"] == "High Risk").sum())
    avg_prob = float(v["Fraud_Probability"].mean())

    # ---- Context bar ----
    top_l, top_r = st.columns([4, 1])
    with top_l:
        chips([
            (f"File: {analysis['file']}", "blue"),
            (f"{total:,} transactions", ""),
            (f"Decision threshold {threshold:.2f}", ""),
            ("Actual labels available" if analysis["has_labels"] else "No actual labels", 
             "green" if analysis["has_labels"] else "amber"),
        ])
    with top_r:
        st.button("Clear data", icon=":material/delete:", on_click=clear_analysis,
                  key="clear_analysis", width="stretch")

    # ---- KPI cards ----
    k1, k2, k3 = st.columns(3)
    with k1:
        metric_card("k1", "Total Transactions", f"{total:,}", "blue")
    with k2:
        metric_card("k2", "Predicted Fraud Cases", f"{fraud:,}", "red",
                    f"Probability at or above the decision threshold ({threshold:.2f}).")
    with k3:
        metric_card("k3", "Predicted Legitimate Cases", f"{legit:,}", "green")
    k4, k5, k6 = st.columns(3)
    with k4:
        metric_card("k4", "Fraud Detection Rate", f"{fraud / total:.2%}", "teal",
                    "Predicted fraud cases divided by total transactions. "
                    "This is the share flagged, not the model's recall.")
    with k5:
        metric_card("k5", "Average Fraud Probability", f"{avg_prob:.2%}", "blue",
                    "Mean model-estimated fraud probability across all transactions.")
    with k6:
        metric_card("k6", "High-Risk Transactions", f"{high_risk:,}", "red",
                    f"Transactions with a fraud probability of {MEDIUM_MAX:.0%} or higher. "
                    "This is a review band, separate from the Fraud classification.")

    st.write("")
    tab_over, tab_dist, tab_risk = st.tabs(["Overview", "Distributions", "Risk analysis"])

    # ================= Overview =================
    with tab_over:
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            chart_card(
                "donut", "Fraud vs legitimate transactions", fig_outcome_donut(v),
                f"{fraud:,} of {total:,} transactions ({fraud / total:.2%}) are classified as fraud "
                f"at the {threshold:.2f} threshold.")

        g = type_stats(v) if TYPE_COL else None
        with c2:
            if g is not None:
                if fraud:
                    top = g.sort_values("Fraud", ascending=False).iloc[0]
                    cap = (f"{top[TYPE_COL]} has the most predicted fraud cases "
                           f"({int(top['Fraud']):,}).")
                else:
                    cap = "No transactions were flagged at this threshold."
                chart_card("cases_type", "Fraud cases by transaction type", fig_cases_by_type(g), cap)
            else:
                note_card("cases_type", "Fraud cases by transaction type",
                          "The model has no transaction type feature.")

        c3, c4 = st.columns(2)
        with c3:
            if g is not None:
                if fraud:
                    top = g.sort_values("Rate", ascending=False).iloc[0]
                    cap = (f"{top[TYPE_COL]} has the highest flagged rate "
                           f"({top['Rate']:.2%} of its {int(top['Transactions']):,} transactions). "
                           "Rates use each type's own transaction count.")
                else:
                    cap = "No transactions were flagged at this threshold."
                chart_card("rate_type", "Fraud rate by transaction type", fig_rate_by_type(g), cap)
            else:
                note_card("rate_type", "Fraud rate by transaction type",
                          "The model has no transaction type feature.")
        with c4:
            if AMOUNT_COL and fraud and legit:
                fr_avg = v.loc[v["Predicted"] == 1, AMOUNT_COL].mean()
                lg_avg = v.loc[v["Predicted"] == 0, AMOUNT_COL].mean()
                chart_card(
                    "avg_amount", "Average amount: fraud vs legitimate", fig_avg_amount(v),
                    f"Flagged transactions average {fr_avg:,.2f} versus {lg_avg:,.2f} for "
                    "legitimate ones.")
            else:
                note_card("avg_amount", "Average amount: fraud vs legitimate",
                          "A comparison needs at least one flagged and one legitimate transaction "
                          "at the current threshold.")

        if STEP_COL:
            chart_card(
                "activity", "Transaction activity by step", fig_activity(v),
                "Blue shows how many transactions occurred at each step (left axis). Red shows how "
                "many were flagged as fraud (right axis).")

    # ================= Distributions =================
    with tab_dist:
        st.write("")
        log_scale = st.toggle(
            "Use log scale for amounts", value=False, key="log_amount",
            help="Helpful when a few very large amounts compress the chart. "
                 "Transactions with an amount of 0 are left out of log-scale charts.")
        zero_note = ""
        if log_scale and AMOUNT_COL:
            zeros = int((v[AMOUNT_COL] <= 0).sum())
            if zeros:
                zero_note = f" {zeros:,} transactions with amount 0 are not shown on the log scale."

        d1, d2 = st.columns(2)
        with d1:
            if AMOUNT_COL:
                chart_card(
                    "amt_hist", "Transaction amount distribution", fig_amount_hist(v, log_scale),
                    f"Median amount: {v[AMOUNT_COL].median():,.2f}. Maximum: "
                    f"{v[AMOUNT_COL].max():,.2f}." + zero_note)
            else:
                note_card("amt_hist", "Transaction amount distribution",
                          "The model has no amount feature.")
        with d2:
            chart_card(
                "prob_hist", "Fraud probability distribution", fig_prob_hist(v, threshold),
                f"{fraud:,} transactions are at or above the dashed threshold line. Most of the "
                "model's probabilities show where it is confident and where it is uncertain.")

        if AMOUNT_COL:
            fig_sc, shown, available = fig_scatter(v, threshold, log_scale)
            if shown < available:
                sc_cap = (f"Showing {shown:,} of {available:,} transactions for speed: flagged "
                          "transactions are prioritized and the rest are randomly sampled. "
                          "All statistics elsewhere use the full data.")
            else:
                sc_cap = f"Showing all {shown:,} transactions."
            chart_card("scatter", "Transaction amount vs fraud probability", fig_sc, sc_cap)

    # ================= Risk analysis =================
    with tab_risk:
        st.write("")
        st.markdown("### Risk categories")
        r1, r2, r3 = st.columns(3)
        with r1:
            text_card("rk1", "Low Risk", f"Fraud probability below {LOW_MAX:.0%}.", "green")
        with r2:
            text_card("rk2", "Medium Risk",
                      f"Fraud probability from {LOW_MAX:.0%} up to (not including) {MEDIUM_MAX:.0%}.",
                      "blue")
        with r3:
            text_card("rk3", "High Risk", f"Fraud probability of {MEDIUM_MAX:.0%} or higher.", "red")
        st.info(
            "Risk categories are fixed bands that help you prioritize manual review. They do not "
            "change the model's classification. A transaction is classified as Fraud only when its "
            f"probability reaches the decision threshold (currently {threshold:.2f}; model default "
            f"{default_threshold:.2f}), so a transaction can be High Risk without being flagged as Fraud.",
            icon=":material/info:",
        )

        counts = v["Risk"].value_counts().reindex(RISK_ORDER, fill_value=0)
        summary = pd.DataFrame({
            "Risk category": RISK_ORDER,
            "Probability range": [f"< {LOW_MAX:.0%}", f"{LOW_MAX:.0%} to < {MEDIUM_MAX:.0%}",
                                  f">= {MEDIUM_MAX:.0%}"],
            "Transactions": counts.values,
            "Share": [c / total for c in counts.values],
        })

        a1, a2 = st.columns(2)
        with a1:
            chart_card("risk_donut", "Risk distribution", fig_risk_donut(v),
                       f"{high_risk:,} transactions ({high_risk / total:.2%}) fall in the High Risk band.")
        with a2:
            with st.container(key="card_gray_risk_table"):
                st.markdown('<div class="aegis-card-title">Risk summary</div>', unsafe_allow_html=True)
                st.dataframe(
                    summary, hide_index=True, width="stretch",
                    column_config={"Share": st.column_config.NumberColumn(format="percent")},
                )
                st.caption("Counts are calculated from the model's predicted probabilities.")

        st.write("")
        st.markdown("### Top 10 highest-risk transactions")
        top10 = v.sort_values("Fraud_Probability", ascending=False).head(10).copy()
        top10.insert(0, "Row #", top10.index + 1)
        cols = ["Row #"] + features + ["Fraud_Probability", "Risk", "Classification"]
        st.dataframe(top10[cols], hide_index=True, width="stretch", column_config=PROBABILITY_COLUMN)
        st.caption("Row # is the transaction's position in the uploaded file (starting at 1).")


def render_about():
    page_header("About the Model",
                "How AEGIS makes its decisions and how well it performed.")

    tab_model, tab_threshold, tab_eval = st.tabs(
        ["Model overview", "Threshold analysis", "Evaluation on uploaded data"]
    )

    # ================= Model overview =================
    with tab_model:
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            text_card("a1", "Algorithm",
                      "Random Forest Classifier inside a preprocessing pipeline.", "blue")
        with c2:
            text_card("a2", "Problem type",
                      "Imbalanced binary classification: fraud is rare compared with legitimate "
                      "transactions.", "teal")

        st.write("")
        st.markdown("### Reported evaluation metrics")
        st.caption("Results reported by the project; they are not recalculated by this app.")
        cols = st.columns(4)
        for col, (name, _, text, accent), idx in zip(cols, REPORTED_METRICS, range(4)):
            with col:
                metric_card(f"met{idx}", name, text, accent)

        st.write("")
        fig = go.Figure(go.Bar(
            x=[m[0] for m in REPORTED_METRICS], y=[m[1] for m in REPORTED_METRICS],
            marker_color=[BLUE, TEAL, BLUE, TEAL], text=[m[2] for m in REPORTED_METRICS],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{x}: %{text}<extra></extra>",
        ))
        fig.update_yaxes(title="Score (0 to 1)", range=[0, 1.1])
        fig.update_xaxes(title="Metric")
        chart_card(
            "reported", "Reported metrics comparison", style_fig(fig, 340),
            "Reported project results, shown on a 0 to 1 scale (percentages are shown as decimals). "
            "Precision is higher than recall: the model's fraud alerts are mostly correct, but it "
            "misses a share of real fraud.")

        st.write("")
        st.markdown("### What the metrics mean for fraud detection")
        p, r = st.columns(2)
        with p:
            text_card("prec", "Precision: how trustworthy an alert is",
                      "Of all transactions the model flags as fraud, precision is the share that "
                      "really are fraud. A precision of 89.39% means roughly 89 of every 100 alerts "
                      "are genuine, so few legitimate customers are disturbed by false alarms.", "blue")
        with r:
            text_card("rec", "Recall: how much fraud is caught",
                      "Of all real fraud cases, recall is the share the model detects. A recall of "
                      "60.50% means roughly 60 of every 100 fraudulent transactions are caught, and "
                      "the rest are missed.", "teal")

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

    analysis = st.session_state.get("analysis")

    # ================= Threshold analysis =================
    with tab_threshold:
        st.write("")
        if analysis is None:
            empty_state(
                "thr", "Upload a CSV to explore thresholds",
                "Threshold analysis shows how many transactions would be flagged at each threshold, "
                "using the model's predictions on your uploaded dataset.",
                "Go to Batch CSV Analysis", "Batch CSV Analysis",
            )
        else:
            frame = analysis["frame"]
            table = threshold_table(frame)
            current = int((frame["Fraud_Probability"] >= threshold).sum())
            st.caption(f"Dataset: {analysis['file']} ({len(frame):,} transactions). "
                       "Calculated from the model's predicted probabilities; nothing is retrained.")

            fig = go.Figure(go.Scatter(
                x=table["Threshold"], y=table["Predicted fraud"], mode="lines+markers",
                line=dict(color=BLUE, width=2), marker=dict(size=7),
                hovertemplate="Threshold %{x:.2f}<br>Predicted fraud: %{y:,}<extra></extra>",
            ))
            fig.add_vline(x=threshold, line_dash="dash", line_color=RED,
                          annotation_text=f"Current {threshold:.2f}", annotation_position="top")
            fig.update_xaxes(title="Decision threshold", range=[0, 1.02])
            fig.update_yaxes(title="Predicted fraud cases", rangemode="tozero")
            chart_card(
                "thr_counts", "Predicted fraud cases by threshold", style_fig(fig, 360),
                f"At the current threshold ({threshold:.2f}), {current:,} transactions are flagged. "
                "Raising the threshold flags fewer transactions; lowering it flags more.")

            if analysis["has_labels"]:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=table["Threshold"], y=table["Precision"], name="Precision", mode="lines+markers",
                    line=dict(color=BLUE, width=2),
                    hovertemplate="Threshold %{x:.2f}<br>Precision: %{y:.2%}<extra></extra>"))
                fig2.add_trace(go.Scatter(
                    x=table["Threshold"], y=table["Recall"], name="Recall", mode="lines+markers",
                    line=dict(color=TEAL, width=2),
                    hovertemplate="Threshold %{x:.2f}<br>Recall: %{y:.2%}<extra></extra>"))
                fig2.add_vline(x=threshold, line_dash="dash", line_color=RED,
                               annotation_text=f"Current {threshold:.2f}", annotation_position="top")
                fig2.update_xaxes(title="Decision threshold", range=[0, 1.02])
                fig2.update_yaxes(title="Score", range=[0, 1.05], tickformat=".0%")
                st.write("")
                chart_card(
                    "thr_pr", "Precision and recall by threshold (uploaded labels)",
                    style_fig(fig2, 360),
                    f"Calculated with the `{LABEL_COL}` labels in your file. Gaps appear where no "
                    "transactions are flagged, so precision is undefined.")

            with st.expander("View threshold table"):
                show = table.rename(columns={"Flagged share": "Share flagged"})
                cfg = {"Share flagged": st.column_config.NumberColumn(format="percent")}
                if analysis["has_labels"]:
                    cfg["Precision"] = st.column_config.NumberColumn(format="percent")
                    cfg["Recall"] = st.column_config.NumberColumn(format="percent")
                st.dataframe(show, hide_index=True, width="stretch", column_config=cfg)

    # ================= Evaluation on uploaded data =================
    with tab_eval:
        st.write("")
        if analysis is None:
            empty_state(
                "eval", "Upload a CSV to evaluate",
                f"Evaluation uses a `{LABEL_COL}` column from your CSV as the true answer. "
                "Upload a file to get started.",
                "Go to Batch CSV Analysis", "Batch CSV Analysis",
            )
        elif not analysis["has_labels"]:
            with st.container(key="card_gray_nolabels"):
                st.markdown('<div class="aegis-card-title">Actual-performance metrics are hidden</div>',
                            unsafe_allow_html=True)
                st.write(
                    f"The confusion matrix, precision-recall curve and measured precision and recall "
                    f"need the true outcome of each transaction. Your file has no valid `{LABEL_COL}` "
                    "column, so these cannot be calculated and nothing is estimated in their place.")
                st.write(
                    f"To enable them, add an `{LABEL_COL}` column (1 = fraud, 0 = legitimate) to your "
                    "CSV and upload it again. The column is used only for evaluation and is never "
                    "given to the model.")
        else:
            frame = analysis["frame"]
            y = frame[LABEL_COL].to_numpy()
            pred = (frame["Fraud_Probability"].to_numpy() >= threshold).astype(int)
            actual_fraud = int((y == 1).sum())
            st.caption(
                f"Dataset: {analysis['file']} ({len(frame):,} transactions, {actual_fraud:,} actual "
                f"fraud). Calculated at threshold {threshold:.2f}. These figures describe this file "
                "only and are separate from the reported project metrics. If the file contains "
                "data the model was trained on, scores may look better than real-world performance.")

            cm = confusion_matrix(y, pred, labels=[0, 1])
            tn, fp, fn, tp = (int(x) for x in cm.ravel())
            precision = tp / (tp + fp) if (tp + fp) else None
            recall = tp / (tp + fn) if (tp + fn) else None
            f1 = (2 * precision * recall / (precision + recall)
                  if precision is not None and recall is not None and (precision + recall) else None)
            both_classes = len(np.unique(y)) == 2
            ap = float(average_precision_score(y, frame["Fraud_Probability"])) if both_classes else None

            def pct(x):
                return f"{x:.2%}" if x is not None else "n/a"

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                metric_card("e1", "Precision (this file)", pct(precision), "blue",
                            "Flagged transactions that are truly fraud. n/a if nothing is flagged.")
            with e2:
                metric_card("e2", "Recall (this file)", pct(recall), "teal",
                            "Actual fraud cases that were flagged. n/a if the file has no fraud.")
            with e3:
                metric_card("e3", "F1-score (this file)", pct(f1), "blue")
            with e4:
                metric_card("e4", "Average Precision (this file)",
                            f"{ap:.4f}" if ap is not None else "n/a", "teal",
                            "Needs both fraud and legitimate transactions in the file.")

            st.write("")
            g1, g2 = st.columns(2)
            with g1:
                fig_cm = px.imshow(
                    cm, x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"], text_auto=True,
                    color_continuous_scale=[[0, "#EFF6FF"], [1, BLUE]],
                    labels=dict(x="Predicted", y="Actual", color="Transactions"),
                )
                fig_cm.update_coloraxes(showscale=False)
                fig_cm.update_xaxes(side="bottom")
                chart_card(
                    "cm", "Confusion matrix", style_fig(fig_cm, 340),
                    f"True negatives {tn:,}, false positives {fp:,}, false negatives {fn:,}, "
                    f"true positives {tp:,}. False negatives are fraud cases the model missed.")
            with g2:
                if both_classes:
                    prec_c, rec_c, _ = precision_recall_curve(y, frame["Fraud_Probability"])
                    fig_pr = go.Figure(go.Scatter(
                        x=rec_c, y=prec_c, mode="lines", line=dict(color=BLUE, width=2),
                        name="Precision-recall curve",
                        hovertemplate="Recall %{x:.2%}<br>Precision %{y:.2%}<extra></extra>"))
                    if precision is not None and recall is not None:
                        fig_pr.add_trace(go.Scatter(
                            x=[recall], y=[precision], mode="markers", name=f"Threshold {threshold:.2f}",
                            marker=dict(color=RED, size=11),
                            hovertemplate="Current threshold<br>Recall %{x:.2%}<br>"
                                          "Precision %{y:.2%}<extra></extra>"))
                    fig_pr.update_xaxes(title="Recall", range=[0, 1.02], tickformat=".0%")
                    fig_pr.update_yaxes(title="Precision", range=[0, 1.05], tickformat=".0%")
                    chart_card(
                        "prc", "Precision-recall curve", style_fig(fig_pr, 340),
                        "Built from your file's labels and the model's probabilities. The red dot "
                        "marks the current decision threshold.")
                else:
                    note_card("prc", "Precision-recall curve",
                              "The curve needs both fraud and legitimate transactions in the labels. "
                              "Your file contains only one class.")


# =====================================================
# 8. Router
# =====================================================
PAGE_RENDERERS = {
    "Dashboard": render_dashboard,
    "Single Transaction Analysis": render_single,
    "Batch CSV Analysis": render_batch,
    "Fraud Analytics": render_analytics,
    "About the Model": render_about,
}

PAGE_RENDERERS[st.session_state["nav"]]()
