"""
Bank Marketing GraphSAGE - Streamlit Frontend

Single-file frontend.

Features:
- Dashboard
- Single customer prediction
- Batch CSV prediction
- Model information
- Backend health monitoring
- Prediction probability visualization
- Basic customer/business interpretation
"""

from __future__ import annotations

import io
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_API_URL = "http://127.0.0.1:8000"

API_URL = os.getenv(
    "BACKEND_API_URL",
    DEFAULT_API_URL,
).rstrip("/")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing GraphSAGE",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME TOKENS
# ============================================================
# Design language: "heterogeneous graph" — nodes + typed edges.
# Ink-navy surface, teal edges, amber nodes, coral signal.
# Display: Space Grotesk · Body: Inter · Data: JetBrains Mono

INK = "#0B1220"
PANEL = "#131B2E"
PANEL_LIGHT = "#1B2438"
EDGE = "#2DD4BF"
NODE = "#F5A623"
SIGNAL = "#FF6B6B"
MIST = "#C9D1D9"
WHITE = "#F8FAFC"


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        --ink: {INK};
        --panel: {PANEL};
        --panel-light: {PANEL_LIGHT};
        --edge: {EDGE};
        --node: {NODE};
        --signal: {SIGNAL};
        --mist: {MIST};
        --white: {WHITE};
        --border: rgba(45, 212, 191, 0.20);
    }}

    @media (prefers-reduced-motion: reduce) {{
        * {{ animation: none !important; transition: none !important; }}
    }}

    html, body, .stApp {{
        background: radial-gradient(120% 140% at 10% 0%, #101a30 0%, var(--ink) 45%) fixed;
        color: var(--mist);
        font-family: 'Inter', sans-serif;
    }}

    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--white) !important;
        letter-spacing: -0.01em;
    }}

    p, li, label, span, div {{
        font-family: 'Inter', sans-serif;
    }}

    .main {{
        padding-top: 1rem;
    }}

    .block-container {{
        max-width: 1400px;
        padding-top: 2rem;
    }}

    /* ---------- Hero ---------- */

    .hero {{
        position: relative;
        padding: 2.75rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 1.75rem;
        background:
            radial-gradient(circle at 8px 8px, rgba(45,212,191,0.35) 1.6px, transparent 1.6px),
            radial-gradient(circle at 68px 38px, rgba(245,166,35,0.30) 1.6px, transparent 1.6px),
            radial-gradient(circle at 38px 68px, rgba(45,212,191,0.22) 1.6px, transparent 1.6px),
            linear-gradient(135deg, #10192E 0%, #0D1526 60%, #0B1220 100%);
        background-size: 76px 76px, 76px 76px, 76px 76px, cover;
        border: 1px solid var(--border);
        overflow: hidden;
        animation: hero-in 700ms ease-out;
    }}

    .hero::after {{
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, transparent 40%, rgba(45,212,191,0.06) 55%, transparent 70%);
        pointer-events: none;
    }}

    @keyframes hero-in {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .hero-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--edge);
        margin-bottom: 0.6rem;
    }}

    .hero-title {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--white);
    }}

    .hero-subtitle {{
        font-size: 1.05rem;
        color: var(--mist);
        opacity: 0.85;
        max-width: 46rem;
    }}

    /* ---------- Metric cards ---------- */

    .metric-card {{
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: var(--panel);
        text-align: center;
    }}

    div[data-testid="stMetric"] {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        transition: border-color 180ms ease, transform 180ms ease;
    }}

    div[data-testid="stMetric"]:hover {{
        border-color: var(--edge);
        transform: translateY(-2px);
    }}

    div[data-testid="stMetricLabel"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--mist) !important;
        opacity: 0.8;
    }}

    div[data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--edge) !important;
        font-weight: 600 !important;
    }}

    /* ---------- Prediction box ---------- */

    .prediction-box {{
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border);
        background: var(--panel);
        margin-top: 1rem;
    }}

    .small-text {{
        font-size: 0.85rem;
        opacity: 0.7;
    }}

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0D1526 0%, #0A101F 100%);
        border-right: 1px solid var(--border);
    }}

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--white) !important;
    }}

    section[data-testid="stSidebar"] .stRadio label {{
        font-family: 'Inter', sans-serif;
        color: var(--mist);
    }}

    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
        padding: 0.35rem 0.5rem;
        border-radius: 8px;
        transition: background 150ms ease;
    }}

    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
        background: var(--panel-light);
    }}

    section[data-testid="stSidebar"] .stRadio input {{
        accent-color: var(--edge);
    }}

    /* ---------- Buttons ---------- */

    .stButton > button,
    .stFormSubmitButton > button,
    div[data-testid="stDownloadButton"] > button {{
        background: linear-gradient(135deg, var(--edge) 0%, #17A398 100%);
        color: #06110F;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        transition: transform 150ms ease, box-shadow 150ms ease;
        box-shadow: 0 0 0 rgba(45,212,191,0);
    }}

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(45,212,191,0.28);
        color: #06110F;
    }}

    /* ---------- Alerts ---------- */

    div[data-testid="stAlertContentInfo"],
    .stAlert:has(> div[data-testid="stAlertContentInfo"]) {{
        background: rgba(45,212,191,0.08) !important;
    }}

    div[data-testid="stNotification"] {{
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        background: var(--panel) !important;
    }}

    /* ---------- Inputs / Forms ---------- */

    div[data-testid="stForm"] {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
    }}

    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
        background: var(--panel-light) !important;
        color: var(--white) !important;
        border-radius: 8px !important;
        border: 1px solid var(--border) !important;
    }}

    /* ---------- Dataframes / code ---------- */

    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
        font-family: 'JetBrains Mono', monospace;
    }}

    .stCodeBlock, pre, code {{
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 10px !important;
    }}

    /* ---------- Divider ---------- */

    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 1.75rem 0;
    }}

    /* ---------- Caption / footer ---------- */

    .stCaption, [data-testid="stCaptionContainer"] {{
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.02em;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

def api_get(
    endpoint: str,
    timeout: int = 30,
):
    """
    GET request to FastAPI backend.
    """

    url = f"{API_URL}{endpoint}"

    response = requests.get(
        url,
        timeout=timeout,
    )

    response.raise_for_status()

    return response


def api_post_file(
    endpoint: str,
    file_bytes: bytes,
    filename: str,
    timeout: int = 300,
):
    """
    POST a CSV file to FastAPI.
    """

    url = f"{API_URL}{endpoint}"

    response = requests.post(
        url,
        files={
            "file": (
                filename,
                file_bytes,
                "text/csv",
            )
        },
        timeout=timeout,
    )

    return response


# ============================================================
# BACKEND STATUS
# ============================================================

def get_backend_status():

    try:

        response = api_get(
            "/health",
            timeout=10,
        )

        data = response.json()

        return (
            True,
            data,
        )

    except Exception as exc:

        return (
            False,
            str(exc),
        )


def get_model_info():

    try:

        response = api_get(
            "/model-info",
            timeout=10,
        )

        return response.json()

    except Exception:

        return None


# ============================================================
# PLOTLY THEME HELPER
# ============================================================

def _apply_plot_theme(figure: go.Figure) -> go.Figure:
    """
    Apply the app's visual theme to a Plotly figure without
    touching any underlying data values.
    """

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Inter, sans-serif",
            "color": MIST,
        },
        margin={"t": 60, "b": 40, "l": 40, "r": 40},
    )

    return figure


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-eyebrow">
            Heterogeneous Graph Neural Network &middot; Inference Console
        </div>

        <div class="hero-title">
            🏦 Bank Marketing GraphSAGE
        </div>

        <div class="hero-subtitle">
            Heterogeneous Graph Neural Network for
            Bank Term Deposit Subscription Prediction
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## Navigation"
    )

    page = st.radio(
        "Go to",
        [
            "🏠 Dashboard",
            "🔮 Single Prediction",
            "📁 Batch Prediction",
            "ℹ️ Model Information",
        ],
    )

    st.divider()

    st.markdown(
        "### Backend"
    )

    st.code(
        API_URL,
        language="text",
    )

    backend_ok, backend_data = (
        get_backend_status()
    )

    if backend_ok:

        st.success(
            "Backend Online"
        )

        st.caption(
            "FINAL_FROZEN_MODEL"
        )

    else:

        st.error(
            "Backend Offline"
        )

        st.caption(
            "Start FastAPI before making predictions."
        )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        "## Project Overview"
    )

    st.write(
        """
        This application uses a frozen heterogeneous
        GraphSAGE model to predict whether a bank customer
        is likely to subscribe to a term deposit.
        """
    )

    # --------------------------------------------------------
    # Backend metrics
    # --------------------------------------------------------

    model_info = get_model_info()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Model",
            (
                model_info["status"]
                if model_info
                else "Unavailable"
            ),
        )

    with col2:

        st.metric(
            "Hidden Dimension",
            (
                model_info["hidden_dim"]
                if model_info
                else "—"
            ),
        )

    with col3:

        st.metric(
            "Dropout",
            (
                model_info["dropout"]
                if model_info
                else "—"
            ),
        )

    with col4:

        st.metric(
            "Threshold",
            (
                model_info[
                    "classification_threshold"
                ]
                if model_info
                else "—"
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    st.markdown(
        "## Model Architecture"
    )

    architecture = pd.DataFrame(
        {
            "Stage": [
                "Customer Data",
                "Preprocessing",
                "Heterogeneous Graph",
                "GraphSAGE Layer 1",
                "GraphSAGE Layer 2",
                "Classifier",
            ],
            "Description": [
                "Bank Marketing customer attributes",
                "Training-fitted preprocessing pipeline",
                "Customer + categorical attribute nodes",
                "Typed GraphSAGE message passing",
                "Typed GraphSAGE message passing",
                "Binary subscription prediction",
            ],
        }
    )

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.markdown(
        "## Research Results"
    )

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    with result_col1:

        st.metric(
            "Test PR-AUC",
            "0.4354",
        )

    with result_col2:

        st.metric(
            "Test ROC-AUC",
            "0.7838",
        )

    with result_col3:

        st.metric(
            "Test F1",
            "0.4343",
        )

    st.info(
        """
        The model was frozen after the research and
        optimization phase. The frontend performs inference
        only and never retrains or modifies the model.
        """
    )


# ============================================================
# SINGLE CUSTOMER PREDICTION
# ============================================================

elif page == "🔮 Single Prediction":

    st.markdown(
        "## Customer Subscription Prediction"
    )

    st.write(
        """
        Enter customer information below to obtain a
        GraphSAGE prediction.
        """
    )

    with st.form(
        "single_prediction_form"
    ):

        st.markdown(
            "### Customer Information"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35,
            )

            job = st.selectbox(
                "Job",
                [
                    "admin.",
                    "blue-collar",
                    "entrepreneur",
                    "housemaid",
                    "management",
                    "retired",
                    "self-employed",
                    "services",
                    "student",
                    "technician",
                    "unemployed",
                    "unknown",
                ],
            )

            marital = st.selectbox(
                "Marital Status",
                [
                    "divorced",
                    "married",
                    "single",
                ],
            )

            education = st.selectbox(
                "Education",
                [
                    "primary",
                    "secondary",
                    "tertiary",
                    "unknown",
                ],
            )

            default = st.selectbox(
                "Credit Default",
                [
                    "no",
                    "yes",
                ],
            )

        with col2:

            balance = st.number_input(
                "Balance",
                value=1000,
                step=100,
            )

            housing = st.selectbox(
                "Housing Loan",
                [
                    "no",
                    "yes",
                ],
            )

            loan = st.selectbox(
                "Personal Loan",
                [
                    "no",
                    "yes",
                ],
            )

            contact = st.selectbox(
                "Contact",
                [
                    "cellular",
                    "telephone",
                    "unknown",
                ],
            )

            day = st.number_input(
                "Campaign Day",
                min_value=1,
                max_value=31,
                value=15,
            )

        with col3:

            month = st.selectbox(
                "Campaign Month",
                [
                    "jan",
                    "feb",
                    "mar",
                    "apr",
                    "may",
                    "jun",
                    "jul",
                    "aug",
                    "sep",
                    "oct",
                    "nov",
                    "dec",
                ],
            )

            duration = st.number_input(
                "Duration",
                min_value=0,
                value=0,
                help=(
                    "Retained for compatibility with the "
                    "original dataset. It is excluded from "
                    "the production model."
                ),
            )

            campaign = st.number_input(
                "Campaign Contacts",
                min_value=1,
                value=1,
            )

            pdays = st.number_input(
                "Days Since Previous Contact",
                value=-1,
            )

            previous = st.number_input(
                "Previous Contacts",
                min_value=0,
                value=0,
            )

            poutcome = st.selectbox(
                "Previous Outcome",
                [
                    "unknown",
                    "failure",
                    "other",
                    "success",
                ],
            )

        submitted = st.form_submit_button(
            "🔮 Predict Subscription",
            use_container_width=True,
        )

    if submitted:

        customer = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
        }

        dataframe = pd.DataFrame(
            [customer]
        )

        csv_buffer = io.StringIO()

        dataframe.to_csv(
            csv_buffer,
            index=False,
            sep=";",
        )

        with st.spinner(
            "Running GraphSAGE inference..."
        ):

            try:

                response = api_post_file(
                    "/batch/predict",
                    csv_buffer.getvalue().encode(
                        "utf-8"
                    ),
                    "single_customer.csv",
                    timeout=120,
                )

                if response.status_code != 200:

                    try:
                        detail = response.json().get(
                            "detail",
                            response.text,
                        )
                    except Exception:
                        detail = response.text

                    st.error(
                        f"Prediction failed: {detail}"
                    )

                else:

                    result = pd.read_csv(
                        io.BytesIO(
                            response.content
                        )
                    )

                    probability = float(
                        result.loc[
                            0,
                            "prediction_probability",
                        ]
                    )

                    prediction = int(
                        result.loc[
                            0,
                            "prediction",
                        ]
                    )

                    label = result.loc[
                        0,
                        "prediction_label",
                    ]

                    st.divider()

                    st.markdown(
                        "## Prediction Result"
                    )

                    col1, col2, col3 = (
                        st.columns(3)
                    )

                    with col1:

                        st.metric(
                            "Subscription Probability",
                            f"{probability:.2%}",
                        )

                    with col2:

                        st.metric(
                            "Prediction",
                            label.upper(),
                        )

                    with col3:

                        st.metric(
                            "Threshold",
                            "50%",
                        )

                    # ----------------------------------------
                    # Probability gauge
                    # ----------------------------------------

                    figure = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,
                            number={
                                "suffix": "%",
                                "font": {"color": WHITE},
                            },
                            title={
                                "text": (
                                    "Subscription Probability"
                                ),
                                "font": {"color": MIST},
                            },
                            gauge={
                                "axis": {
                                    "range": [
                                        0,
                                        100,
                                    ],
                                    "tickcolor": MIST,
                                },
                                "bar": {"color": EDGE},
                                "bgcolor": PANEL_LIGHT,
                                "bordercolor": "rgba(45,212,191,0.25)",
                                "steps": [
                                    {"range": [0, 50], "color": "rgba(255,107,107,0.14)"},
                                    {"range": [50, 100], "color": "rgba(45,212,191,0.14)"},
                                ],
                                "threshold": {
                                    "line": {"color": NODE, "width": 3},
                                    "thickness": 0.85,
                                    "value": 50,
                                },
                            },
                        )
                    )

                    figure.update_layout(
                        height=350,
                    )

                    figure = _apply_plot_theme(figure)

                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                    )

                    if prediction == 1:

                        st.success(
                            "The model predicts that "
                            "this customer is likely to "
                            "subscribe."
                        )

                    else:

                        st.warning(
                            "The model predicts that "
                            "this customer is unlikely to "
                            "subscribe."
                        )

                    st.caption(
                        "Prediction generated by the "
                        "FINAL_FROZEN_MODEL."
                    )

            except requests.RequestException as exc:

                st.error(
                    "Could not connect to FastAPI: "
                    f"{exc}"
                )


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "📁 Batch Prediction":

    st.markdown(
        "## Batch Customer Prediction"
    )

    st.write(
        """
        Upload the original Bank Marketing CSV.
        The backend will preprocess the records, construct
        inference graphs, and generate predictions.
        """
    )

    st.info(
        """
        The CSV does not need a `customer_index` column.
        The application generates customer indices automatically.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Bank Marketing CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        st.markdown(
            "### Uploaded File"
        )

        st.write(
            f"**Filename:** {uploaded_file.name}"
        )

        file_bytes = (
            uploaded_file.getvalue()
        )

        # ----------------------------------------------------
        # Preview
        # ----------------------------------------------------

        try:

            preview = pd.read_csv(
                io.BytesIO(
                    file_bytes
                ),
                sep=";",
            )

            st.write(
                f"Rows: **{len(preview):,}**"
            )

            st.write(
                f"Columns: **{len(preview.columns)}**"
            )

            st.dataframe(
                preview.head(10),
                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                f"Could not preview CSV: {exc}"
            )

        st.warning(
            """
            Batch inference currently processes customers
            through the graph inference pipeline individually.
            A large file may therefore take some time.
            """
        )

        if st.button(
            "🚀 Run Batch Prediction",
            use_container_width=True,
        ):

            progress_placeholder = (
                st.empty()
            )

            progress_placeholder.info(
                "Sending CSV to FastAPI..."
            )

            try:

                response = api_post_file(
                    "/batch/predict",
                    file_bytes,
                    uploaded_file.name,
                    timeout=1800,
                )

                if response.status_code != 200:

                    try:

                        detail = (
                            response.json()
                            .get(
                                "detail",
                                response.text,
                            )
                        )

                    except Exception:

                        detail = response.text

                    progress_placeholder.empty()

                    st.error(
                        f"Batch prediction failed: {detail}"
                    )

                else:

                    progress_placeholder.empty()

                    result = pd.read_csv(
                        io.BytesIO(
                            response.content
                        )
                    )

                    st.success(
                        "Batch prediction completed successfully."
                    )

                    # ----------------------------------------
                    # Summary
                    # ----------------------------------------

                    total = len(result)

                    yes_count = int(
                        (
                            result[
                                "prediction"
                            ]
                            == 1
                        ).sum()
                    )

                    no_count = int(
                        (
                            result[
                                "prediction"
                            ]
                            == 0
                        ).sum()
                    )

                    average_probability = float(
                        result[
                            "prediction_probability"
                        ].mean()
                    )

                    col1, col2, col3, col4 = (
                        st.columns(4)
                    )

                    with col1:

                        st.metric(
                            "Customers",
                            f"{total:,}",
                        )

                    with col2:

                        st.metric(
                            "Predicted Yes",
                            f"{yes_count:,}",
                        )

                    with col3:

                        st.metric(
                            "Predicted No",
                            f"{no_count:,}",
                        )

                    with col4:

                        st.metric(
                            "Average Probability",
                            f"{average_probability:.2%}",
                        )

                    # ----------------------------------------
                    # Results
                    # ----------------------------------------

                    st.markdown(
                        "### Prediction Results"
                    )

                    st.dataframe(
                        result,
                        use_container_width=True,
                    )

                    # ----------------------------------------
                    # Distribution chart
                    # ----------------------------------------

                    st.markdown(
                        "### Prediction Distribution"
                    )

                    chart_data = pd.DataFrame(
                        {
                            "Prediction": [
                                "Yes",
                                "No",
                            ],
                            "Customers": [
                                yes_count,
                                no_count,
                            ],
                        }
                    )

                    figure = go.Figure(
                        data=[
                            go.Bar(
                                x=chart_data[
                                    "Prediction"
                                ],
                                y=chart_data[
                                    "Customers"
                                ],
                                marker={
                                    "color": [EDGE, SIGNAL],
                                    "line": {"width": 0},
                                },
                            )
                        ]
                    )

                    figure.update_layout(
                        height=400,
                        xaxis_title="Prediction",
                        yaxis_title="Customers",
                        xaxis={"gridcolor": "rgba(201,209,217,0.08)"},
                        yaxis={"gridcolor": "rgba(201,209,217,0.08)"},
                    )

                    figure = _apply_plot_theme(figure)

                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                    )

                    # ----------------------------------------
                    # Download
                    # ----------------------------------------

                    output_csv = (
                        result.to_csv(
                            index=False
                        ).encode("utf-8")
                    )

                    st.download_button(
                        label=(
                            "⬇️ Download Predictions CSV"
                        ),
                        data=output_csv,
                        file_name=(
                            "bank_marketing_predictions.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True,
                    )

            except requests.Timeout:

                progress_placeholder.empty()

                st.error(
                    "The backend timed out while processing "
                    "the batch."
                )

            except requests.RequestException as exc:

                progress_placeholder.empty()

                st.error(
                    "Could not connect to FastAPI: "
                    f"{exc}"
                )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "ℹ️ Model Information":

    st.markdown(
        "## Frozen Model Information"
    )

    model_info = get_model_info()

    if model_info is None:

        st.error(
            "Could not retrieve model information "
            "from the backend."
        )

    else:

        st.success(
            "Model successfully loaded."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### Model Configuration"
            )

            config = pd.DataFrame(
                {
                    "Parameter": [
                        "Status",
                        "Hidden Dimension",
                        "Dropout",
                        "Learning Rate",
                        "Classification Threshold",
                        "Device",
                    ],
                    "Value": [
                        model_info[
                            "status"
                        ],
                        model_info[
                            "hidden_dim"
                        ],
                        model_info[
                            "dropout"
                        ],
                        model_info[
                            "learning_rate"
                        ],
                        model_info[
                            "classification_threshold"
                        ],
                        model_info[
                            "device"
                        ],
                    ],
                }
            )

            st.dataframe(
                config,
                hide_index=True,
                use_container_width=True,
            )

        with col2:

            st.markdown(
                "### Research Metrics"
            )

            metrics = pd.DataFrame(
                {
                    "Metric": [
                        "Test PR-AUC",
                        "Test ROC-AUC",
                        "Test F1",
                    ],
                    "Value": [
                        0.435405,
                        0.783800,
                        0.434307,
                    ],
                }
            )

            st.dataframe(
                metrics,
                hide_index=True,
                use_container_width=True,
            )

        st.divider()

        st.markdown(
            "### Graph Structure"
        )

        graph_structure = pd.DataFrame(
            {
                "Node Type": [
                    "customer",
                    "job",
                    "education",
                    "marital",
                    "contact",
                    "month",
                ],
                "Purpose": [
                    "Customer prediction nodes",
                    "Job category",
                    "Education category",
                    "Marital status category",
                    "Contact method category",
                    "Campaign month category",
                ],
            }
        )

        st.dataframe(
            graph_structure,
            hide_index=True,
            use_container_width=True,
        )

        st.markdown(
            "### Relations"
        )

        st.code(
            """
customer ──has_job──────────────→ job
job ──rev_has_job───────────────→ customer

customer ──has_education────────→ education
education ──rev_has_education───→ customer

customer ──has_marital_status───→ marital
marital ──rev_has_marital_status→ customer

customer ──contacted_via────────→ contact
contact ──rev_contacted_via─────→ customer

customer ──campaign_month───────→ month
month ──rev_campaign_month──────→ customer
            """,
            language="text",
        )

        st.divider()

        st.markdown(
            "### Prediction Policy"
        )

        st.info(
            """
            The production model follows a pre-contact
            prediction scenario.

            `duration` is excluded because it represents
            completed contact duration and would not be
            available before the interaction.

            The frontend never retrains the model.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Bank Marketing GraphSAGE • "
    "Heterogeneous Graph Neural Network • "
    "FINAL_FROZEN_MODEL"
)