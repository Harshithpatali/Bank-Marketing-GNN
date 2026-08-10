from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.services.prediction_service import PredictionService


# ============================================================
# VERIFIED MODEL INFORMATION
# ============================================================

MODEL_STATUS = "FINAL_FROZEN_MODEL"
TEST_PR_AUC = 0.435405
TEST_ROC_AUC = 0.783800
TEST_F1 = 0.434307
CLASSIFICATION_THRESHOLD = 0.5

JOB_VALUES = [
    "admin.", "blue-collar", "entrepreneur", "housemaid",
    "management", "retired", "self-employed", "services",
    "student", "technician", "unemployed", "unknown",
]

MARITAL_VALUES = ["divorced", "married", "single"]
EDUCATION_VALUES = ["primary", "secondary", "tertiary", "unknown"]
DEFAULT_VALUES = ["no", "yes"]
YES_NO = ["no", "yes"]
CONTACT_VALUES = ["cellular", "telephone", "unknown"]
MONTH_VALUES = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]
POUTCOME_VALUES = ["unknown", "failure", "other", "success"]


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Bank Marketing GraphSAGE",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME
# ============================================================

INK = "#0B1220"
PANEL = "#131B2E"
PANEL_LIGHT = "#1B2438"
EDGE = "#2DD4BF"
NODE = "#F5A623"
SIGNAL = "#FF6B6B"
MIST = "#C9D1D9"
WHITE = "#F8FAFC"

st.markdown(
    f"""
<style>
html, body, .stApp {{
    background:
        radial-gradient(
            120% 140% at 10% 0%,
            #101A30 0%,
            #0B1220 50%,
            #080E1A 100%
        ) fixed !important;
    color: {MIST} !important;
}}

.block-container {{
    max-width: 1400px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}}

h1, h2, h3, h4 {{
    color: {WHITE} !important;
}}

.hero {{
    padding: 2.5rem;
    margin-bottom: 2rem;
    border-radius: 22px;
    background:
        radial-gradient(
            circle at 8px 8px,
            rgba(45,212,191,.32) 1.5px,
            transparent 1.6px
        ),
        linear-gradient(135deg, #10192E, #0B1220);
    background-size: 76px 76px, cover;
    border: 1px solid rgba(45,212,191,.24);
}}

.eyebrow {{
    color: {EDGE};
    font-family: monospace;
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .14em;
}}

.hero-title {{
    color: {WHITE};
    font-size: 2.65rem;
    font-weight: 700;
    margin: .5rem 0;
}}

.hero-subtitle {{
    color: {MIST};
    font-size: 1.05rem;
    line-height: 1.6;
}}

div[data-testid="stMetric"] {{
    background: linear-gradient(145deg, #151E34, #11192C) !important;
    border: 1px solid rgba(45,212,191,.22) !important;
    border-radius: 15px !important;
    padding: 1rem 1.15rem !important;
}}

div[data-testid="stMetricLabel"] {{
    color: #AEB8C7 !important;
}}

div[data-testid="stMetricValue"] {{
    color: {EDGE} !important;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1526, #090F1C) !important;
}}

.stButton > button,
.stFormSubmitButton > button,
div[data-testid="stDownloadButton"] > button {{
    background: linear-gradient(135deg, #2DD4BF, #17A398) !important;
    color: #06110F !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 42px;
}}

div[data-testid="stForm"] {{
    background: #131B2E !important;
    border: 1px solid rgba(45,212,191,.20) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}}

.stTextInput input, .stNumberInput input,
div[data-baseweb="select"] > div {{
    background: #1B2438 !important;
    color: {WHITE} !important;
    border: 1px solid rgba(45,212,191,.20) !important;
}}

hr {{
    border: none !important;
    height: 1px !important;
    background: linear-gradient(
        90deg, transparent,
        rgba(45,212,191,.25),
        transparent
    ) !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# INFERENCE SERVICE
# ============================================================

@st.cache_resource(show_spinner="Loading frozen GraphSAGE model...")
def get_prediction_service() -> PredictionService:
    """
    Load the verified production inference service once.

    This does NOT train the model. It only loads:
    - preprocessor.joblib
    - bank_heterodata.pt
    - hetero_graphsage_final.pt
    """
    return PredictionService()


def run_single_prediction(customer: dict) -> pd.DataFrame:
    """Run one customer through the verified inference service."""
    service = get_prediction_service()
    return service.predict_batch_with_input(
        pd.DataFrame([customer])
    )


@st.cache_data(show_spinner=False)
def run_batch_prediction(
    csv_bytes: bytes,
    separator: str,
) -> pd.DataFrame:
    """Run a batch CSV through the verified inference service."""
    dataframe = pd.read_csv(
        io.BytesIO(csv_bytes),
        sep=separator,
    )

    service = get_prediction_service()

    return service.predict_batch_with_input(dataframe)


def get_model_info() -> dict:
    """Return metadata verified from the frozen checkpoint."""
    return {
        "status": MODEL_STATUS,
        "hidden_dim": 128,
        "dropout": 0.2,
        "learning_rate": 0.0005,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "device": "cpu",
    }


def detect_separator(file_bytes: bytes) -> str:
    """Detect the separator used by an uploaded CSV."""
    sample = file_bytes[:5000].decode(
        "utf-8",
        errors="replace",
    )

    return (
        ";"
        if sample.count(";") > sample.count(",")
        else ","
    )


def probability_gauge(probability: float) -> go.Figure:
    """Create the probability gauge."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={
                "suffix": "%",
                "font": {"color": WHITE},
            },
            title={
                "text": "Subscription Probability",
                "font": {"color": MIST},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": MIST,
                },
                "bar": {"color": EDGE},
                "bgcolor": PANEL_LIGHT,
                "bordercolor": "rgba(45,212,191,.25)",
                "steps": [
                    {
                        "range": [0, 50],
                        "color": "rgba(255,107,107,.14)",
                    },
                    {
                        "range": [50, 100],
                        "color": "rgba(45,212,191,.14)",
                    },
                ],
                "threshold": {
                    "line": {
                        "color": NODE,
                        "width": 3,
                    },
                    "thickness": 0.85,
                    "value": 50,
                },
            },
        )
    )

    figure.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Inter, sans-serif",
            "color": MIST,
        },
        margin={
            "t": 60,
            "b": 40,
            "l": 40,
            "r": 40,
        },
    )

    return figure


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="eyebrow">
        HETEROGENEOUS GRAPH NEURAL NETWORK · INFERENCE CONSOLE
    </div>
    <div class="hero-title">
        🏦 Bank Marketing GraphSAGE
    </div>
    <div class="hero-subtitle">
        Heterogeneous Graph Neural Network for Bank Term Deposit
        Subscription Prediction
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## Navigation")

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

    st.markdown("### Inference Engine")

    try:
        get_prediction_service()
        st.success("Model Online")
        st.caption(MODEL_STATUS)
    except Exception as exc:
        st.error("Model Load Failed")
        st.caption(str(exc))

    st.caption(
        "Standalone deployment — no FastAPI server required."
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown("## Project Overview")

    st.write(
        """
        This portfolio demo uses the frozen heterogeneous
        GraphSAGE model to predict whether a bank customer is
        likely to subscribe to a term deposit.

        The research and training phase is complete. This app
        performs inference only and never retrains or modifies
        the frozen model.
        """
    )

    model_info = get_model_info()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model", model_info["status"])

    with col2:
        st.metric("Hidden Dimension", model_info["hidden_dim"])

    with col3:
        st.metric("Dropout", model_info["dropout"])

    with col4:
        st.metric(
            "Threshold",
            f"{model_info['classification_threshold']:.0%}",
        )

    st.divider()

    st.markdown("## Research Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Test PR-AUC", f"{TEST_PR_AUC:.4f}")

    with col2:
        st.metric("Test ROC-AUC", f"{TEST_ROC_AUC:.4f}")

    with col3:
        st.metric("Test F1", f"{TEST_F1:.4f}")

    st.divider()

    st.markdown("## Model Architecture")

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

    st.info(
        """
        Live demo architecture:
        Streamlit → PredictionService → frozen GraphSAGE model.

        FastAPI is still included in the GitHub project as the
        production REST API implementation, but this public demo
        does not depend on a separately hosted backend.
        """
    )


# ============================================================
# SINGLE PREDICTION
# ============================================================

elif page == "🔮 Single Prediction":

    st.markdown("## Customer Subscription Prediction")

    st.write(
        "Enter customer information and run the frozen GraphSAGE model."
    )

    with st.form("single_prediction_form"):

        st.markdown("### Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35,
            )

            job = st.selectbox("Job", JOB_VALUES)

            marital = st.selectbox(
                "Marital Status",
                MARITAL_VALUES,
            )

            education = st.selectbox(
                "Education",
                EDUCATION_VALUES,
            )

            default = st.selectbox(
                "Credit Default",
                DEFAULT_VALUES,
            )

        with col2:
            balance = st.number_input(
                "Balance",
                value=1000,
                step=100,
            )

            housing = st.selectbox(
                "Housing Loan",
                YES_NO,
            )

            loan = st.selectbox(
                "Personal Loan",
                YES_NO,
            )

            contact = st.selectbox(
                "Contact",
                CONTACT_VALUES,
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
                MONTH_VALUES,
            )

            duration = st.number_input(
                "Duration",
                min_value=0,
                value=0,
                help=(
                    "Retained for input compatibility. "
                    "The frozen production model excludes duration."
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
                POUTCOME_VALUES,
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

        with st.spinner(
            "Running frozen GraphSAGE inference..."
        ):
            try:
                result = run_single_prediction(customer)

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

                label = str(
                    result.loc[
                        0,
                        "prediction_label",
                    ]
                )

                st.divider()
                st.markdown("## Prediction Result")

                col1, col2, col3 = st.columns(3)

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
                        f"{CLASSIFICATION_THRESHOLD:.0%}",
                    )

                st.plotly_chart(
                    probability_gauge(probability),
                    use_container_width=True,
                )

                if prediction == 1:
                    st.success(
                        "The model predicts that this customer "
                        "is likely to subscribe."
                    )
                else:
                    st.warning(
                        "The model predicts that this customer "
                        "is unlikely to subscribe."
                    )

                st.caption(
                    "Prediction generated by the FINAL_FROZEN_MODEL."
                )

            except Exception as exc:
                st.error(
                    f"Prediction failed: {exc}"
                )


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "📁 Batch Prediction":

    st.markdown("## Batch Customer Prediction")

    st.write(
        """
        Upload a Bank Marketing CSV. The frozen GraphSAGE
        inference pipeline will generate predictions locally.
        """
    )

    st.info(
        """
        The uploaded CSV does not need a `customer_index`
        column. The verified inference service creates the
        inference customer index automatically.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Bank Marketing CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        st.markdown("### Uploaded File")

        st.write(
            f"**Filename:** {uploaded_file.name}"
        )

        separator = detect_separator(file_bytes)

        try:
            preview = pd.read_csv(
                io.BytesIO(file_bytes),
                sep=separator,
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
            preview = None

        if st.button(
            "🚀 Run Batch Prediction",
            use_container_width=True,
        ):

            with st.spinner(
                "Running local GraphSAGE batch inference..."
            ):
                try:
                    result = run_batch_prediction(
                        file_bytes,
                        separator,
                    )

                    st.success(
                        "Batch prediction completed successfully."
                    )

                    total = len(result)

                    yes_count = int(
                        (result["prediction"] == 1).sum()
                    )

                    no_count = int(
                        (result["prediction"] == 0).sum()
                    )

                    average_probability = float(
                        result[
                            "prediction_probability"
                        ].mean()
                    )

                    col1, col2, col3, col4 = st.columns(4)

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

                    st.markdown("### Prediction Results")

                    st.dataframe(
                        result,
                        use_container_width=True,
                    )

                    chart_data = pd.DataFrame(
                        {
                            "Prediction": ["Yes", "No"],
                            "Customers": [
                                yes_count,
                                no_count,
                            ],
                        }
                    )

                    figure = go.Figure(
                        data=[
                            go.Bar(
                                x=chart_data["Prediction"],
                                y=chart_data["Customers"],
                                marker={
                                    "color": [
                                        EDGE,
                                        SIGNAL,
                                    ],
                                    "line": {
                                        "width": 0
                                    },
                                },
                            )
                        ]
                    )

                    figure.update_layout(
                        height=400,
                        xaxis_title="Prediction",
                        yaxis_title="Customers",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={
                            "family": "Inter, sans-serif",
                            "color": MIST,
                        },
                        xaxis={
                            "gridcolor":
                                "rgba(201,209,217,0.08)"
                        },
                        yaxis={
                            "gridcolor":
                                "rgba(201,209,217,0.08)"
                        },
                    )

                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                    )

                    output_csv = (
                        result
                        .to_csv(index=False)
                        .encode("utf-8")
                    )

                    st.download_button(
                        label="⬇️ Download Predictions CSV",
                        data=output_csv,
                        file_name=(
                            "bank_marketing_predictions.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True,
                    )

                except Exception as exc:
                    st.error(
                        f"Batch prediction failed: {exc}"
                    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "ℹ️ Model Information":

    st.markdown("## Frozen Model Information")

    try:
        get_prediction_service()
        st.success(
            "Frozen GraphSAGE inference service loaded successfully."
        )
    except Exception as exc:
        st.error(
            f"Could not load inference service: {exc}"
        )

    model_info = get_model_info()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Configuration")

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
                    model_info["status"],
                    model_info["hidden_dim"],
                    model_info["dropout"],
                    model_info["learning_rate"],
                    model_info["classification_threshold"],
                    model_info["device"],
                ],
            }
        )

        st.dataframe(
            config,
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.markdown("### Research Metrics")

        metrics = pd.DataFrame(
            {
                "Metric": [
                    "Test PR-AUC",
                    "Test ROC-AUC",
                    "Test F1",
                ],
                "Value": [
                    TEST_PR_AUC,
                    TEST_ROC_AUC,
                    TEST_F1,
                ],
            }
        )

        st.dataframe(
            metrics,
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    st.markdown("### Graph Structure")

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
                "Marital category",
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

    st.markdown("### Relations")

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

    st.markdown("### Prediction Policy")

    st.info(
        """
        The production model follows a pre-contact prediction
        scenario. `duration` is retained as an input for
        compatibility with the original dataset but is excluded
        from the frozen production feature set.

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
    "FINAL_FROZEN_MODEL • "
    "Standalone Streamlit Demo"
)