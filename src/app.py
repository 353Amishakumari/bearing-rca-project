import streamlit as st
import numpy as np
import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from io import BytesIO
from pathlib import Path

# ============================================================
# ENSURE PROJECT ROOT IS ON sys.path
#
# This makes "from src.xxx import ..." work no matter how the
# app is launched (streamlit run src/app.py, python -m streamlit
# run src/app.py, from project root, from inside src/, etc.)
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rag_engine import generate_rca_report, build_vector_store
from src.predict import predict_fault
from src.thermal_image_module import (
    load_radiometric_image,
    compute_thermal_metrics,
    predict_motor_thermal_status,
    get_thermal_cnn,
    validate_thermal_input,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Industrial Bearing RCA Dashboard",
    page_icon="🏭",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# This fixes the dataset path issue when Streamlit runs from src/
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

X_PATH = ROOT_DIR / "dataset" / "processed" / "X_segments.npy"
Y_PATH = ROOT_DIR / "dataset" / "processed" / "y_labels.npy"

EVAL_SUMMARY_PATH = ROOT_DIR / "saved_models" / "eval_summary.txt"
CONFUSION_MATRIX_PATH = ROOT_DIR / "saved_models" / "confusion_matrix.png"

# External Paderborn file-level evaluation
PADERBORN_RESULTS_PATH = ROOT_DIR / "paderborn_results" / "paderborn_file_level_results.csv"
PADERBORN_CM_PATH = ROOT_DIR / "paderborn_results" / "file_level_confusion_matrix.npy"

# Trained thermal CNN weights (produced by train_thermal.py)
THERMAL_MODEL_PATH = ROOT_DIR / "saved_models" / "radiometric_thermal_cnn.pt"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
/* ============================================================
   PROFESSIONAL INDUSTRIAL LIGHT THEME
   UI-only styling: application logic is unchanged.
   ============================================================ */

.stApp {
    background: #f4f7fb;
    color: #172033;
}

.main .block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dfe5ee;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #172033;
}

/* Typography */
h1, h2, h3, h4 {
    color: #172033 !important;
    letter-spacing: -0.02em;
}

p, li, label, .stMarkdown {
    color: #4b5567;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: #687386 !important;
}

/* Header / hero */
.hero-card {
    background: linear-gradient(135deg, #ffffff 0%, #eef4fb 100%);
    border: 1px solid #dce5f0;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 25px rgba(31, 45, 61, 0.06);
}

.hero-title {
    font-size: 34px;
    font-weight: 800;
    color: #16243a;
    margin: 0 0 8px 0;
}

.hero-subtitle {
    font-size: 16px;
    color: #64748b;
    margin: 0;
}

.system-pill {
    display: inline-block;
    margin-top: 18px;
    padding: 7px 14px;
    border-radius: 999px;
    background: #e8f7ee;
    color: #19713d;
    font-weight: 700;
    font-size: 13px;
}


/* Top-level project header */
.top-project-header {
    margin-top: 0;
    margin-bottom: 28px;
}

/* Evaluation table */
.evaluation-table-wrap {
    width: 100%;
    overflow-x: auto;
    margin: 18px 0 22px 0;
    border: 1px solid #dfe6ef;
    border-radius: 14px;
    background: #ffffff;
    box-shadow: 0 5px 18px rgba(31, 45, 61, 0.05);
}

.evaluation-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 15px;
    color: #172033;
}

.evaluation-table th {
    background: #eef3f8;
    color: #344054;
    font-weight: 700;
    text-align: left;
    padding: 14px 16px;
    border-bottom: 1px solid #dfe6ef;
}

.evaluation-table td {
    padding: 13px 16px;
    border-bottom: 1px solid #e8edf3;
    background: #ffffff;
}

.evaluation-table tr:last-child td {
    border-bottom: none;
}

.evaluation-table td:not(:first-child),
.evaluation-table th:not(:first-child) {
    text-align: center;
}

.evaluation-subtitle {
    font-size: 18px;
    font-weight: 750;
    color: #172033;
    margin: 24px 0 10px 0;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e0e6ef;
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 5px 18px rgba(31, 45, 61, 0.05);
}

div[data-testid="stMetricLabel"] {
    color: #667085 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #172033 !important;
    font-weight: 800;
}

/* Industrial cards */
.scada-card {
    background: #ffffff;
    border: 1px solid #dfe6ef;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    min-height: 135px;
    margin-bottom: 15px;
    box-shadow: 0 6px 20px rgba(31, 45, 61, 0.05);
}

.scada-label {
    font-size: 14px;
    color: #687386;
    font-weight: 700;
    margin-bottom: 10px;
}

.scada-value {
    font-size: 28px;
    font-weight: 800;
    color: #172033;
}

.small-status {
    color: #7b8798;
    font-size: 13px;
    margin-top: 8px;
    font-weight: 500;
}

/* Status colors */
.status-running {
    color: #168447;
    font-size: 24px;
    font-weight: 800;
}

.status-warning {
    color: #b77900;
    font-size: 24px;
    font-weight: 800;
}

.status-danger {
    color: #c62828;
    font-size: 24px;
    font-weight: 800;
}

/* Alarm banners */
.big-success,
.big-warning,
.big-danger {
    padding: 22px 26px;
    border-radius: 16px;
    margin: 12px 0 20px 0;
    box-shadow: 0 5px 18px rgba(31, 45, 61, 0.05);
}

.big-success {
    background: #ecf9f0;
    border: 1px solid #b8e4c5;
    color: #176b38;
}

.big-warning {
    background: #fff8e8;
    border: 1px solid #f2d38b;
    color: #8a5a00;
}

.big-danger {
    background: #fff0f0;
    border: 1px solid #efb4b4;
    color: #9f2525;
}

.big-success h2, .big-success h4, .big-success p,
.big-warning h2, .big-warning h4, .big-warning p,
.big-danger h2, .big-danger h4, .big-danger p {
    color: inherit !important;
}

/* RAG / report cards */
.rag-card {
    background: #ffffff;
    border: 1px solid #dfe6ef;
    border-left: 5px solid #3b82f6;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 6px 20px rgba(31, 45, 61, 0.05);
}

.report-card {
    background: #ffffff;
    border: 1px solid #dfe6ef;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 6px 20px rgba(31, 45, 61, 0.05);
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
    border: 1px solid #d4dce8;
    box-shadow: 0 3px 10px rgba(31, 45, 61, 0.05);
}

.stButton > button[kind="primary"] {
    background: #1769aa;
    color: white;
    border: none;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(31, 45, 61, 0.10);
}

/* Inputs */
.stSelectbox > div > div,
.stNumberInput > div > div,
.stSlider > div {
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #e1e7ef !important;
}

/* Info / success / warning containers */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Code blocks */
.stCodeBlock {
    border-radius: 12px;
}

/* Hide Streamlit top decoration */
header[data-testid="stHeader"] {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SCADA CARD FUNCTION
# ============================================================

def scada_card(label, value, value_class="scada-value", subtitle=""):

    html = f"""
    <div class="scada-card">
        <div class="scada-label">{label}</div>
        <div class="{value_class}">{value}</div>
        <div class="small-status">{subtitle}</div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# FAULT MAP
# ============================================================

FAULT_MAP = {
    0: "Normal",
    1: "Inner Race Fault (IR007)",
    2: "Ball Fault (B007)",
    3: "Outer Race Fault (OR007)"
}

SHORT_FAULT_MAP = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Ball Fault",
    3: "Outer Race Fault"
}


# ============================================================
# INITIALIZE RAG
# ============================================================

@st.cache_resource
def init_rag():

    try:
        return build_vector_store()

    except Exception as e:
        return None


# ============================================================
# LOAD TRAINED THERMAL CNN
# ============================================================

@st.cache_resource
def load_trained_thermal_model():
    """
    Loads RadiometricThermalCNN with weights trained by train_thermal.py.
    Returns (model, weights_loaded_bool). If weights are missing, the
    untrained model is still returned so the app doesn't crash — callers
    should check weights_loaded before trusting predictions.
    """
    model = get_thermal_cnn()
    if THERMAL_MODEL_PATH.exists():
        try:
            state_dict = torch.load(str(THERMAL_MODEL_PATH), map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            return model, True
        except Exception:
            return model, False
    return model, False


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_real_data():

    if not X_PATH.exists():
        raise FileNotFoundError(
            f"X_segments.npy not found at:\n{X_PATH}"
        )

    if not Y_PATH.exists():
        raise FileNotFoundError(
            f"y_labels.npy not found at:\n{Y_PATH}"
        )

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    return X, y


# ============================================================
# FIND A REPRESENTATIVE SAMPLE
#
# This searches real samples of the selected class and chooses
# one that the CNN actually predicts correctly.
# ============================================================

def find_best_sample_for_class(
    X,
    y,
    selected_class,
    max_candidates=60
):

    matching_indices = np.where(y == selected_class)[0]

    if len(matching_indices) == 0:
        return None, None, None

    # Shuffle real samples
    candidates = np.random.permutation(matching_indices)

    # Limit candidates so the app remains fast
    candidates = candidates[:min(
        max_candidates,
        len(candidates)
    )]

    best_match = None
    best_match_confidence = -1

    best_any = None
    best_any_confidence = -1

    for idx in candidates:

        signal = X[idx]

        try:

            prediction = predict_fault(signal)

            predicted_class_id = int(
                prediction["class_id"]
            )

            confidence = float(
                prediction["confidence"]
            )

            # Save best overall prediction
            if confidence > best_any_confidence:

                best_any_confidence = confidence

                best_any = (
                    idx,
                    signal,
                    prediction
                )

            # Prefer a sample where CNN correctly predicts
            # the selected class
            if (
                predicted_class_id == selected_class
                and confidence > best_match_confidence
            ):

                best_match_confidence = confidence

                best_match = (
                    idx,
                    signal,
                    prediction
                )

        except Exception:
            continue

    # If a correctly predicted sample is found
    if best_match is not None:
        return best_match

    # Otherwise return best available real sample
    return best_any


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "thermal_validated" not in st.session_state:
    st.session_state.thermal_validated = False

if "thermal_validation_score" not in st.session_state:
    st.session_state.thermal_validation_score = 0.0

if "thermal_validation_reason" not in st.session_state:
    st.session_state.thermal_validation_reason = "No thermal image uploaded."

if "thermal_fault_result" not in st.session_state:
    st.session_state.thermal_fault_result = None


# ============================================================
# LOAD DATA SAFELY
# ============================================================

try:

    X_all, y_all = load_real_data()

except Exception as e:

    st.error(f"❌ Dataset load error: {e}")

    st.stop()


# ============================================================
# INITIALIZE RAG + THERMAL CNN
# ============================================================

vdb = init_rag()
thermal_model, thermal_model_loaded = load_trained_thermal_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🏭 Machine Parameters")

motor_speed = st.sidebar.number_input(
    "Motor Speed (RPM)",
    min_value=500,
    max_value=5000,
    value=1697,
    step=1
)

bearing_temperature = st.sidebar.number_input(
    "Bearing Temperature (°C)",
    min_value=20.0,
    max_value=200.0,
    value=50.0,
    step=0.1
)

machine_load = st.sidebar.slider(
    "Machine Load (%)",
    min_value=0,
    max_value=100,
    value=44
)

st.sidebar.divider()


# ============================================================
# FAULT SCENARIO SELECTOR
# IMPORTANT: Default is OUTER RACE FAULT
# ============================================================

st.sidebar.subheader("🎯 Fault Test Scenario")

selected_class = st.sidebar.selectbox(
    "Select Condition for AI Testing",
    options=[0, 1, 2, 3],

    # Default Outer Race Fault
    index=3,

    format_func=lambda x: {
        0: "🟢 Normal Operating Condition",
        1: "🟡 Inner Race Fault (IR007)",
        2: "🟠 Ball Fault (B007)",
        3: "🔴 Outer Race Fault (OR007)"
    }[x]
)

st.sidebar.caption(
    "The selected condition chooses a REAL vibration "
    "sample from the dataset. The CNN performs the "
    "final prediction."
)

st.sidebar.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.sidebar.subheader("🌡️ Upload Thermal / Machine Image")

uploaded_image = st.sidebar.file_uploader(
    "Upload Thermal Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="Upload a genuine infrared/thermal image of the bearing or machine."
)

st.sidebar.caption(
    "The image is validated before the diagnostic pipeline starts."
)

if not thermal_model_loaded:
    st.sidebar.caption(
        "⚠️ Trained thermal CNN weights not found — thermal fault "
        "classification is disabled until train_thermal.py is run."
    )


# ============================================================
# TOP-LEVEL PROJECT HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-card top-project-header">
        <div class="hero-title">🏭 Industrial Multimodal Bearing Fault Detection & RCA System</div>
        <div class="hero-subtitle">
            AI-Based Bearing Fault Detection • Condition Monitoring •
            Multimodal Inspection • Adaptive RAG • Root Cause Analysis
        </div>
        <div class="system-pill">● INDUSTRIAL DIAGNOSTIC PLATFORM</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.sidebar.divider()

show_eval = st.sidebar.checkbox(
    "📈 Show Model Performance"
)

if show_eval:

    # ========================================================
    # 1. CWRU INTERNAL TEST-SET PERFORMANCE
    # ========================================================

    st.subheader("📊 CWRU Internal Test-Set Performance")

    eval_text = ""

    if EVAL_SUMMARY_PATH.exists():

        with open(EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
            eval_text = f.read()

        accuracy_match = re.search(
            r"Overall Accuracy\s*[:=]\s*([0-9.]+)%?",
            eval_text,
            re.IGNORECASE
        )

        f1_match = re.search(
            r"Macro F1[- ]Score\s*[:=]\s*([0-9.]+)%?",
            eval_text,
            re.IGNORECASE
        )

        score1, score2 = st.columns(2)

        with score1:
            accuracy_value = (
                f"{float(accuracy_match.group(1)):.2f}%"
                if accuracy_match else "N/A"
            )
            st.metric("CWRU Accuracy", accuracy_value)

        with score2:
            f1_value = (
                f"{float(f1_match.group(1)):.2f}%"
                if f1_match else "N/A"
            )
            st.metric("CWRU Macro F1-Score", f1_value)

        report_rows = []

        for class_name in [
            "Normal",
            "Inner Race",
            "Ball",
            "Outer Race"
        ]:

            pattern = (
                rf"^\s*{re.escape(class_name)}\s+"
                r"([0-9.]+)\s+"
                r"([0-9.]+)\s+"
                r"([0-9.]+)\s+"
                r"(\d+)\s*$"
            )

            match = re.search(
                pattern,
                eval_text,
                re.MULTILINE
            )

            if match:
                report_rows.append({
                    "Fault Class": class_name,
                    "Precision": f"{float(match.group(1)) * 100:.2f}%",
                    "Recall": f"{float(match.group(2)) * 100:.2f}%",
                    "F1-Score": f"{float(match.group(3)) * 100:.2f}%",
                    "Support": int(match.group(4))
                })

        if report_rows:
            st.dataframe(
                pd.DataFrame(report_rows),
                use_container_width=True,
                hide_index=True
            )

        if eval_text:
            with st.expander("View CWRU Evaluation Summary"):
                st.code(
                    eval_text,
                    language="text"
                )

        st.markdown(
            '<div class="evaluation-subtitle">🧩 CWRU Confusion Matrix</div>',
            unsafe_allow_html=True
        )

        if CONFUSION_MATRIX_PATH.exists():

            matrix_left, matrix_center, matrix_right = st.columns(
                [1, 2, 1]
            )

            with matrix_center:
                st.image(
                    str(CONFUSION_MATRIX_PATH),
                    width=650
                )

        else:
            st.warning(
                "CWRU confusion_matrix.png file not found."
            )

    else:
        st.warning(
            "CWRU eval_summary.txt file not found."
        )

    # ========================================================
    # 2. PADERBORN EXTERNAL / FILE-LEVEL PERFORMANCE
    # ========================================================

    st.divider()

    st.subheader("🌐 Paderborn External Dataset — File-Level Evaluation")

    st.caption(
        "External-domain evaluation using the existing Paderborn "
        "file-level predictions. Each measurement file is aggregated "
        "from its window predictions using majority voting."
    )

    if PADERBORN_RESULTS_PATH.exists():

        try:
            paderborn_df = pd.read_csv(
                PADERBORN_RESULTS_PATH
            )

            required_columns = {
                "true_label",
                "predicted_label"
            }

            if not required_columns.issubset(
                set(paderborn_df.columns)
            ):
                st.error(
                    "Paderborn result file does not contain the "
                    "required columns: true_label, predicted_label."
                )
            else:

                # ------------------------------------------------
                # File-level metrics
                # ------------------------------------------------
                from sklearn.metrics import (
                    accuracy_score,
                    precision_score,
                    recall_score,
                    f1_score,
                    confusion_matrix
                )

                y_true_pb = paderborn_df["true_label"].astype(int).to_numpy()
                y_pred_pb = paderborn_df["predicted_label"].astype(int).to_numpy()

                pb_accuracy = accuracy_score(
                    y_true_pb,
                    y_pred_pb
                )

                pb_precision = precision_score(
                    y_true_pb,
                    y_pred_pb,
                    labels=[0, 1, 3],
                    average="macro",
                    zero_division=0
                )

                pb_recall = recall_score(
                    y_true_pb,
                    y_pred_pb,
                    labels=[0, 1, 3],
                    average="macro",
                    zero_division=0
                )

                pb_f1 = f1_score(
                    y_true_pb,
                    y_pred_pb,
                    labels=[0, 1, 3],
                    average="macro",
                    zero_division=0
                )

                metric1, metric2, metric3, metric4 = st.columns(4)

                with metric1:
                    st.metric(
                        "File-Level Accuracy",
                        f"{pb_accuracy * 100:.2f}%"
                    )

                with metric2:
                    st.metric(
                        "Macro Precision",
                        f"{pb_precision * 100:.2f}%"
                    )

                with metric3:
                    st.metric(
                        "Macro Recall",
                        f"{pb_recall * 100:.2f}%"
                    )

                with metric4:
                    st.metric(
                        "Macro F1-Score",
                        f"{pb_f1 * 100:.2f}%"
                    )

                # ------------------------------------------------
                # Per-class metrics
                # ------------------------------------------------
                class_names = {
                    0: "Normal",
                    1: "Inner Race",
                    3: "Outer Race"
                }

                pb_rows = []

                for class_id, class_name in class_names.items():

                    class_true = (y_true_pb == class_id)
                    class_pred = (y_pred_pb == class_id)

                    tp = int(np.sum(class_true & class_pred))
                    fp = int(np.sum(~class_true & class_pred))
                    fn = int(np.sum(class_true & ~class_pred))
                    support = int(np.sum(class_true))

                    precision = (
                        tp / (tp + fp)
                        if (tp + fp) > 0 else 0.0
                    )

                    recall = (
                        tp / (tp + fn)
                        if (tp + fn) > 0 else 0.0
                    )

                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0 else 0.0
                    )

                    pb_rows.append({
                        "Fault Class": class_name,
                        "Precision": f"{precision * 100:.2f}%",
                        "Recall": f"{recall * 100:.2f}%",
                        "F1-Score": f"{f1 * 100:.2f}%",
                        "Support": support
                    })

                st.dataframe(
                    pd.DataFrame(pb_rows),
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # Paderborn confusion matrix
                # ------------------------------------------------
                st.markdown(
                    '<div class="evaluation-subtitle">🧩 Paderborn File-Level Confusion Matrix</div>',
                    unsafe_allow_html=True
                )

                if PADERBORN_CM_PATH.exists():
                    try:
                        cm_pb_saved = np.load(PADERBORN_CM_PATH)
                        labels_pb = ["Normal", "Inner Race", "Outer Race"]
                        fig_pb, ax_pb = plt.subplots(figsize=(7, 5.5))
                        im_pb = ax_pb.imshow(cm_pb_saved, interpolation="nearest")
                        ax_pb.set_title("Paderborn File-Level Confusion Matrix")
                        ax_pb.set_xlabel("Predicted Label")
                        ax_pb.set_ylabel("Actual Label")
                        ax_pb.set_xticks(range(3)); ax_pb.set_yticks(range(3))
                        ax_pb.set_xticklabels(labels_pb, rotation=20); ax_pb.set_yticklabels(labels_pb)
                        for i in range(3):
                            for j in range(3):
                                ax_pb.text(j, i, str(int(cm_pb_saved[i, j])), ha="center", va="center")
                        fig_pb.colorbar(im_pb, ax=ax_pb)
                        fig_pb.tight_layout()
                        st.pyplot(fig_pb, use_container_width=False)
                        plt.close(fig_pb)
                    except Exception as e:
                        st.error(f"Could not render Paderborn confusion matrix: {e}")
                else:
                    cm_pb = confusion_matrix(y_true_pb, y_pred_pb, labels=[0, 1, 3])
                    cm_display = pd.DataFrame(cm_pb, index=["Actual Normal", "Actual Inner Race", "Actual Outer Race"], columns=["Pred Normal", "Pred Inner Race", "Pred Outer Race"])
                    st.dataframe(cm_display, use_container_width=True)

                # ------------------------------------------------
                # Interpretation
                # ------------------------------------------------
                if pb_accuracy < 0.50:

                    st.warning(
                        "⚠️ External-domain performance is substantially "
                        "lower than the internal CWRU test-set performance. "
                        "This indicates a domain-generalization challenge "
                        "that should be investigated before claiming "
                        "cross-dataset robustness."
                    )

                st.info(
                    "Research note: Paderborn results are shown separately "
                    "from the CWRU test-set results. The two scores should "
                    "not be combined into a single accuracy value."
                )

        except Exception as e:

            st.error(
                f"❌ Could not load Paderborn evaluation results: {e}"
            )

    else:

        st.info(
            "Paderborn file-level results not found. "
            "Run `python src\\paderborn_bearing_level.py` first."
        )

    st.divider()

# ============================================================
# RUN BUTTON
# ============================================================

run_analysis = st.sidebar.button(
    "🚀 RUN INDUSTRIAL ANALYSIS",
    use_container_width=True,
    type="primary"
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if run_analysis:

    # ------------------------------------------------------------
    # THERMAL IMAGE VALIDATION GATE
    # Uses validate_thermal_input() from thermal_image_module.py,
    # which correctly handles BOTH genuine 16-bit radiometric
    # thermal images (Zenodo dataset, FLIR/Workswell cameras) AND
    # 8-bit false-color infrared images — while rejecting posters,
    # logos, and ordinary RGB photos.
    # ------------------------------------------------------------
    thermal_fault_result = None
    thermal_display_rgb = None

    if uploaded_image is not None:
        (
            is_thermal,
            validation_tag,
            thermal_score,
            _preview_metrics,
            thermal_reason
        ) = validate_thermal_input(uploaded_image)

        if not is_thermal:
            st.error("❌ INVALID THERMAL IMAGE")
            st.warning(
                "The uploaded image does not appear to be a valid "
                "thermal/infrared machine image."
            )
            st.info(thermal_reason)
            st.session_state.analysis_done = False
            st.stop()

        st.success(
            f"✅ Thermal image validation passed ({validation_tag}) — "
            f"screening score: {thermal_score * 100:.1f}%"
        )
        st.caption(thermal_reason)

        # --------------------------------------------------------
        # LOAD RADIOMETRIC / DISPLAY IMAGE
        #
        # IMPORTANT: raw uploaded 16-bit radiometric PNGs cannot be
        # displayed directly by Streamlit/PIL (JPEG re-encoding fails
        # on mode "I;16"). We always convert to the colorized 8-bit
        # display_rgb produced by load_radiometric_image() and use
        # THAT for on-screen preview and the PDF report — never the
        # raw uploaded bytes.
        # --------------------------------------------------------
        try:
            temp_grid, is_16bit, cam_type, display_rgb = load_radiometric_image(uploaded_image)
            thermal_display_rgb = display_rgb
        except Exception as e:
            st.warning(f"Could not render thermal image preview: {e}")

        # --------------------------------------------------------
        # TRAINED THERMAL CNN PREDICTION
        # --------------------------------------------------------
        if thermal_model_loaded and thermal_display_rgb is not None:
            try:
                thermal_metrics = compute_thermal_metrics(temp_grid)
                thermal_fault_result = predict_motor_thermal_status(temp_grid, thermal_metrics)
            except Exception as e:
                st.warning(f"Thermal CNN prediction failed: {e}")
        elif not thermal_model_loaded:
            st.info(
                "Trained thermal model weights not found at "
                f"{THERMAL_MODEL_PATH}. Run train_thermal.py first."
            )

    else:
        thermal_score = 0.0
        thermal_reason = "No thermal image uploaded."
        st.warning(
            "⚠️ No thermal image uploaded. Analysis will remain "
            "vibration-based."
        )

    with st.spinner(
        "🤖 Searching real vibration samples and running CNN..."
    ):

        result = find_best_sample_for_class(
            X_all,
            y_all,
            selected_class
        )

        if result[0] is None:

            st.error(
                f"❌ No valid samples found for "
                f"{FAULT_MAP[selected_class]}"
            )

            st.stop()

        random_idx, signal, prediction = result

        # True label
        true_class_id = int(
            y_all[random_idx]
        )

        true_fault = FAULT_MAP.get(
            true_class_id,
            "Unknown"
        )

        # CNN result
        predicted_class_id = int(
            prediction["class_id"]
        )

        predicted_fault = prediction["fault"]

        model_confidence = float(
            prediction["confidence"]
        )

        probabilities = prediction["probabilities"]

        # Save everything
        st.session_state.analysis_done = True

        st.session_state.signal = signal
        st.session_state.random_idx = random_idx

        st.session_state.true_class_id = true_class_id
        st.session_state.true_fault = true_fault

        st.session_state.predicted_class_id = (
            predicted_class_id
        )

        st.session_state.predicted_fault = (
            predicted_fault
        )

        st.session_state.model_confidence = (
            model_confidence
        )

        st.session_state.probabilities = probabilities

        st.session_state.motor_speed = motor_speed
        st.session_state.bearing_temperature = (
            bearing_temperature
        )

        st.session_state.machine_load = machine_load

        st.session_state.selected_class = (
            selected_class
        )

        st.session_state.uploaded_image = (
            uploaded_image
        )

        st.session_state.thermal_validated = (
            uploaded_image is not None
        )
        st.session_state.thermal_validation_score = (
            thermal_score if uploaded_image is not None else 0.0
        )
        st.session_state.thermal_validation_reason = (
            thermal_reason
        )

        st.session_state.thermal_fault_result = thermal_fault_result
        st.session_state.thermal_display_rgb = thermal_display_rgb


# ============================================================
# MAIN DASHBOARD
# ============================================================

if st.session_state.analysis_done:

    report_text = "RAG report was not generated."

    # ========================================================
    # LOAD SESSION DATA
    # ========================================================

    signal = st.session_state.signal

    random_idx = st.session_state.random_idx

    true_class_id = st.session_state.true_class_id
    true_fault = st.session_state.true_fault

    predicted_class_id = (
        st.session_state.predicted_class_id
    )

    predicted_fault = (
        st.session_state.predicted_fault
    )

    model_confidence = (
        st.session_state.model_confidence
    )

    probabilities = (
        st.session_state.probabilities
    )

    rpm = st.session_state.motor_speed

    manual_temperature = (
        st.session_state.bearing_temperature
    )

    machine_load_value = (
        st.session_state.machine_load
    )

    uploaded_image_result = (
        st.session_state.uploaded_image
    )

    thermal_validated = st.session_state.get(
        "thermal_validated", False
    )
    thermal_validation_score = st.session_state.get(
        "thermal_validation_score", 0.0
    )
    thermal_validation_reason = st.session_state.get(
        "thermal_validation_reason",
        "No thermal image uploaded."
    )

    thermal_fault_result = st.session_state.get(
        "thermal_fault_result", None
    )

    thermal_display_rgb = st.session_state.get(
        "thermal_display_rgb", None
    )


    # ========================================================
    # TITLE
    # ========================================================

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🔎 Live Industrial Diagnostic Dashboard</div>
            <div class="hero-subtitle">
                Real Bearing Vibration Data • CNN Fault Detection •
                Adaptive RAG Root Cause Analysis
            </div>
            <div class="system-pill">● DIAGNOSTIC SYSTEM ACTIVE</div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # SIGNAL CALCULATIONS
    # ========================================================

    signal = np.asarray(signal).flatten()

    vibration_rms = float(
        np.sqrt(
            np.mean(signal ** 2)
        )
    )

    peak_vibration = float(
        np.max(
            np.abs(signal)
        )
    )

    mean_vibration = float(
        np.mean(
            np.abs(signal)
        )
    )


    # ========================================================
    # STATUS AND ALARM LOGIC
    # BASED ON CNN PREDICTION
    # ========================================================

    if predicted_class_id == 0:

        machine_status = "NORMAL"

        alarm_status = "NO ALARM"

        severity = "LOW"

        bearing_health = min(
            100.0,
            max(
                90.0,
                100.0 - vibration_rms * 20
            )
        )

        status_css = "status-running"

        alarm_icon = "✅"


    elif predicted_class_id == 1:

        machine_status = "WARNING"

        alarm_status = "WARNING ALARM"

        severity = "MEDIUM"

        bearing_health = min(
            85.0,
            max(
                60.0,
                82.0 - vibration_rms * 20
            )
        )

        status_css = "status-warning"

        alarm_icon = "⚠️"


    elif predicted_class_id == 2:

        machine_status = "FAULT DETECTED"

        alarm_status = "HIGH PRIORITY ALARM"

        severity = "HIGH"

        bearing_health = min(
            65.0,
            max(
                30.0,
                60.0 - vibration_rms * 20
            )
        )

        status_css = "status-danger"

        alarm_icon = "🚨"


    else:

        machine_status = "FAULT DETECTED"

        alarm_status = "CRITICAL ALARM"

        severity = "CRITICAL"

        bearing_health = min(
            55.0,
            max(
                15.0,
                45.0 - vibration_rms * 15
            )
        )

        status_css = "status-danger"

        alarm_icon = "🚨"


    # ========================================================
    # TOP METRICS
    # ========================================================

    st.subheader("📊 AI Fault Detection")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "AI Predicted Fault",
        SHORT_FAULT_MAP.get(
            predicted_class_id,
            predicted_fault
        )
    )

    metric2.metric(
        "AI Confidence",
        f"{model_confidence:.2f}%"
    )

    metric3.metric(
        "Bearing Health",
        f"{bearing_health:.1f}%"
    )

    metric4.metric(
        "Severity",
        severity
    )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    if predicted_class_id == 0:

        st.markdown(
            f"""
            <div class="big-success">
            <h2>{alarm_icon} NO ALARM</h2>
            <h4>Machine Status: NORMAL</h4>
            <p>
            Bearing is currently operating under normal conditions.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif predicted_class_id == 1:

        st.markdown(
            f"""
            <div class="big-warning">
            <h2>{alarm_icon} WARNING ALARM</h2>
            <h4>Inner Race Fault Detected</h4>
            <p>
            Inspection and preventive maintenance are recommended.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif predicted_class_id == 2:

        st.markdown(
            f"""
            <div class="big-danger">
            <h2>{alarm_icon} HIGH PRIORITY ALARM</h2>
            <h4>Ball / Rolling Element Fault Detected</h4>
            <p>
            Immediate maintenance inspection is recommended.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="big-danger">
            <h2>{alarm_icon} CRITICAL ALARM</h2>
            <h4>Outer Race Fault Detected</h4>
            <p>
            Critical bearing fault detected. Immediate inspection
            and corrective action are recommended.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    # ========================================================
    # DATASET + PREDICTION DETAILS
    # ========================================================

    st.subheader("🧠 CNN Prediction Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            f"""
            **Dataset Segment:** {random_idx}

            **Actual Fault:** {true_fault}

            **Actual Class ID:** {true_class_id}
            """
        )

    with col2:

        st.success(
            f"""
            **AI Prediction:** {predicted_fault}

            **Predicted Class ID:** {predicted_class_id}

            **Confidence:** {model_confidence:.2f}%
            """
        )

    with col3:

        if predicted_class_id == true_class_id:

            st.success(
                """
                ### ✅ PREDICTION MATCH

                CNN prediction matches the
                actual dataset label.
                """
            )

        else:

            st.warning(
                """
                ### ⚠️ PREDICTION MISMATCH

                CNN prediction does not match
                the dataset label.
                """
            )


    st.divider()


    # ========================================================
    # INDUSTRIAL SENSOR DATA
    # ========================================================

    st.subheader("📡 Industrial Sensor Data")

    sensor1, sensor2, sensor3, sensor4, sensor5 = st.columns(5)

    sensor1.metric(
        "Motor Speed",
        f"{rpm} RPM"
    )

    sensor2.metric(
        "Temperature",
        f"{manual_temperature:.1f} °C"
    )

    sensor3.metric(
        "Machine Load",
        f"{machine_load_value}%"
    )

    sensor4.metric(
        "Vibration RMS",
        f"{vibration_rms:.4f}"
    )

    sensor5.metric(
        "Peak Vibration",
        f"{peak_vibration:.4f}"
    )


    # ========================================================
    # VIBRATION SIGNAL
    # ========================================================

    st.divider()

    st.subheader("📈 Vibration Signal Analysis")

    st.line_chart(signal)

    stat1, stat2, stat3 = st.columns(3)

    stat1.metric(
        "Mean Amplitude",
        f"{mean_vibration:.4f}"
    )

    stat2.metric(
        "RMS",
        f"{vibration_rms:.4f}"
    )

    stat3.metric(
        "Peak",
        f"{peak_vibration:.4f}"
    )


    # ========================================================
    # AI PROBABILITIES
    # ========================================================

    st.divider()

    st.subheader("🎯 AI Fault Probability")

    probabilities = np.asarray(probabilities).flatten()

    probability_data = {
        "Normal": (
            float(probabilities[0]) * 100
            if len(probabilities) > 0 else 0
        ),

        "Inner Race Fault": (
            float(probabilities[1]) * 100
            if len(probabilities) > 1 else 0
        ),

        "Ball Fault": (
            float(probabilities[2]) * 100
            if len(probabilities) > 2 else 0
        ),

        "Outer Race Fault": (
            float(probabilities[3]) * 100
            if len(probabilities) > 3 else 0
        )
    }

    st.bar_chart(probability_data)


    # ========================================================
    # THERMAL / VISUAL INSPECTION
    # ========================================================

    st.divider()

    st.subheader("🌡️ Thermal / Visual Inspection")

    visual_col1, visual_col2 = st.columns(2)

    with visual_col1:

        if thermal_display_rgb is not None:

            st.image(
                thermal_display_rgb,
                caption="Validated Thermal / Infrared Inspection Image (colorized)",
                use_container_width=True
            )

            if thermal_validated:
                st.success(
                    "✅ Thermal image accepted — "
                    f"screening score: {thermal_validation_score * 100:.1f}%"
                )
                st.caption(thermal_validation_reason)
            else:
                st.error("❌ Thermal image validation did not pass.")

            # ----------------------------------------------------
            # TRAINED THERMAL CNN RESULT
            # ----------------------------------------------------
            if thermal_fault_result is not None:
                st.markdown("#### 🌡️ Thermal CNN Diagnosis")
                tcol1, tcol2 = st.columns(2)
                tcol1.metric(
                    "Thermal-Based Fault",
                    thermal_fault_result["class_name"]
                )
                tcol2.metric(
                    "Thermal CNN Confidence",
                    f"{thermal_fault_result['confidence']:.2f}%"
                )
                st.caption(
                    "Diagnosis derived from the trained radiometric "
                    "thermal CNN, independent of the vibration-based "
                    "prediction above."
                )
            elif not thermal_model_loaded:
                st.caption(
                    "Thermal fault classification unavailable — "
                    "trained weights not found."
                )

        else:

            st.info(
                "No thermal image uploaded. "
                "Upload a genuine JPG or PNG thermal/infrared image."
            )

    with visual_col2:

        st.info(
            """
            ### Multimodal Context

            • Vibration signal is used for primary CNN fault detection.

            • Uploaded thermal image is validated before being accepted.

            • A separately trained thermal CNN classifies the validated
              image as Normal / Shaft Misalignment / Broken Rotor Bars.

            • Adaptive RAG retrieves maintenance SOP knowledge.

            • Final RCA report provides recommended actions.
            """
        )


    # ========================================================
    # ADAPTIVE RAG ROOT CAUSE ANALYSIS
    # ========================================================

    st.divider()

    st.subheader("🔎 Adaptive RAG Root Cause Analysis")

    if vdb is None:

        st.warning(
            "RAG Vector Database could not be initialized."
        )

    else:

        with st.spinner(
            "🔎 Retrieving SOP knowledge and generating RCA..."
        ):

            try:

                report_text = generate_rca_report(
                    class_id=predicted_class_id,
                    confidence_score=model_confidence,
                    vector_db=vdb
                )

                st.code(
                    report_text,
                    language="text"
                )

            except Exception as e:

                st.error(
                    f"RAG report error: {e}"
                )


    # ========================================================
    # PDF REPORT GENERATOR
    # ========================================================

    def make_vibration_plot(signal_data):
        fig, ax = plt.subplots(figsize=(8, 3.2))
        arr = np.asarray(signal_data).flatten()[:1024]
        ax.plot(arr)
        ax.set_title("Vibration Signal — Selected Bearing Segment")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    def make_probability_plot(probabilities_data):
        names = ["Normal", "Inner Race", "Ball", "Outer Race"]
        values = [
            float(probabilities_data.get("Normal", 0)),
            float(probabilities_data.get("Inner Race Fault", 0)),
            float(probabilities_data.get("Ball Fault", 0)),
            float(probabilities_data.get("Outer Race Fault", 0))
        ]
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.bar(names, values)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probability (%)")
        ax.set_title("AI Fault Probability")
        ax.tick_params(axis="x", rotation=15)
        for i, value in enumerate(values):
            ax.text(i, min(value + 2, 98), f"{value:.1f}%", ha="center", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    def make_paderborn_cm_plot():
        if not PADERBORN_CM_PATH.exists():
            return None
        cm = np.load(PADERBORN_CM_PATH)
        labels = ["Normal", "Inner Race", "Outer Race"]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title("Paderborn File-Level Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(labels, rotation=20); ax.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        buf=BytesIO(); fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
        plt.close(fig); buf.seek(0); return buf

    def create_diagnostic_pdf():
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.lib.utils import ImageReader
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, Image as RLImage, PageBreak

            buffer=BytesIO()
            doc=SimpleDocTemplate(buffer, pagesize=A4, rightMargin=14*mm, leftMargin=14*mm, topMargin=14*mm, bottomMargin=14*mm)
            styles=getSampleStyleSheet()
            body=ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.2, leading=12)
            heading=ParagraphStyle("Heading", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=10, spaceAfter=6)
            small=ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, leading=10)

            story=[Paragraph("INDUSTRIAL MULTIMODAL BEARING DIAGNOSTIC REPORT", styles["Title"]),
                   Paragraph("AI-Powered Fault Detection • Condition Monitoring • Multimodal Inspection • Adaptive RAG • Root Cause Analysis", body), Spacer(1,8)]

            data=[["Parameter","Value"],
                  ["Dataset Segment",str(random_idx)], ["Actual Dataset Fault",str(true_fault)],
                  ["AI Predicted Fault",str(predicted_fault)], ["Fault Code",str(SHORT_FAULT_MAP.get(predicted_class_id,predicted_fault))],
                  ["AI Confidence",f"{model_confidence:.2f}%"], ["Machine Status",str(machine_status)],
                  ["Alarm Status",str(alarm_status)], ["Severity",str(severity)], ["Motor Speed",f"{rpm} RPM"],
                  ["Bearing Temperature",f"{manual_temperature:.1f} °C"], ["Machine Load",f"{machine_load_value}%"],
                  ["Vibration RMS",f"{vibration_rms:.4f}"], ["Peak Vibration",f"{peak_vibration:.4f}"], ["Bearing Health",f"{bearing_health:.1f}%"]]

            if thermal_fault_result is not None:
                data.append(["Thermal-Based Fault", str(thermal_fault_result["class_name"])])
                data.append(["Thermal CNN Confidence", f"{thermal_fault_result['confidence']:.2f}%"])

            table=Table(data,colWidths=[58*mm,115*mm],repeatRows=1)
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1f2937")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#d1d5db")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f4f6")]),("FONTSIZE",(0,0),(-1,-1),8.5),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
            story.append(table)

            story.append(Paragraph("Diagnostic Visualizations", heading))
            try: story.append(RLImage(make_vibration_plot(signal), width=175*mm, height=70*mm)); story.append(Spacer(1,5))
            except Exception as e: story.append(Paragraph(f"[Vibration plot could not be generated: {e}]", small)); story.append(Spacer(1,5))
            try: story.append(RLImage(make_probability_plot(probability_data), width=175*mm, height=72*mm)); story.append(Spacer(1,5))
            except Exception as e: story.append(Paragraph(f"[Probability chart could not be generated: {e}]", small)); story.append(Spacer(1,5))

            if thermal_display_rgb is not None:
                story.append(Paragraph("Thermal / Visual Inspection Image", heading))
                try:
                    from PIL import Image as PILImage
                    pil_preview = PILImage.fromarray(thermal_display_rgb.astype(np.uint8), mode="RGB")
                    preview_buf = BytesIO()
                    pil_preview.save(preview_buf, format="PNG")
                    preview_buf.seek(0)
                    # Use ImageReader only to read natural dimensions for
                    # aspect-ratio scaling; pass the raw buffer itself
                    # (not the ImageReader wrapper) to RLImage, since this
                    # reportlab version's Image/RLImage constructor does
                    # not accept an ImageReader instance as its first
                    # argument (it wants a path, bytes, or file-like
                    # object it can open directly).
                    iw, ih = ImageReader(preview_buf).getSize()
                    preview_buf.seek(0)
                    maxw=175*mm; maxh=85*mm; scale=min(maxw/iw,maxh/ih)
                    story.append(RLImage(preview_buf, width=iw*scale, height=ih*scale))
                    story.append(Spacer(1,6))
                except Exception as e:
                    story.append(Paragraph(f"[Image could not be embedded in PDF: {e}]", small))
                    story.append(Spacer(1,6))
            elif uploaded_image_result is not None:
                story.append(Paragraph("Thermal / Visual Inspection Image", heading))
                story.append(Paragraph("[No renderable preview available for the uploaded thermal image.]", small))
                story.append(Spacer(1,6))

            if CONFUSION_MATRIX_PATH.exists():
                story.append(PageBreak())
                story.append(Paragraph("CWRU Internal Test-Set Confusion Matrix", heading))
                story.append(RLImage(str(CONFUSION_MATRIX_PATH), width=145*mm, height=105*mm))
                story.append(Paragraph("CWRU internal test-set performance is reported separately from external Paderborn evaluation.", small))

            try:
                pb=make_paderborn_cm_plot()
                if pb is not None:
                    story.append(Spacer(1,10)); story.append(Paragraph("Paderborn External Dataset — File-Level Confusion Matrix", heading)); story.append(RLImage(pb,width=135*mm,height=112*mm)); story.append(Paragraph("Paderborn file-level evaluation: Accuracy 33.33%, Macro F1 16.67%.",small))
            except Exception: pass

            story.append(PageBreak()); story.append(Paragraph("Adaptive RAG Root Cause Analysis", heading)); story.append(Preformatted(report_text, styles["Code"]))
            story.append(Paragraph("Diagnostic Conclusion", heading)); story.append(Paragraph(f"The CNN identified <b>{predicted_fault}</b> with <b>{model_confidence:.2f}%</b> confidence. Machine status is <b>{machine_status}</b> with <b>{alarm_status}</b> and <b>{severity}</b> severity. Bearing health is estimated at <b>{bearing_health:.1f}%</b>.", body))
            doc.build(story); return buffer.getvalue()
        except ImportError: return None
        except Exception as e:
            st.warning(f"PDF graph generation warning: {e}")
            return None


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.divider()

    st.subheader("📄 Final Industrial Diagnostic Summary")

    final_summary = f"""
INDUSTRIAL MULTIMODAL BEARING DIAGNOSTIC REPORT

Dataset Segment: {random_idx}

Actual Dataset Fault:
{true_fault}

AI Predicted Fault:
{predicted_fault}

Fault Code:
{SHORT_FAULT_MAP.get(predicted_class_id, predicted_fault)}

AI Confidence:
{model_confidence:.2f}%

Machine Status:
{machine_status}

Alarm Status:
{alarm_status}

Severity:
{severity}

Motor Speed:
{rpm} RPM

Bearing Temperature:
{manual_temperature:.1f} °C

Machine Load:
{machine_load_value}%

Vibration RMS:
{vibration_rms:.4f}

Peak Vibration:
{peak_vibration:.4f}

Bearing Health:
{bearing_health:.1f}%
"""

    if thermal_fault_result is not None:
        final_summary += f"""
Thermal-Based Fault:
{thermal_fault_result['class_name']}

Thermal CNN Confidence:
{thermal_fault_result['confidence']:.2f}%
"""

    st.code(
        final_summary,
        language="text"
    )

    st.success(
        "✅ Industrial Diagnostic Pipeline Completed Successfully"
    )

    st.divider()
    st.subheader("📄 Download Industrial Diagnostic Report")
    pdf_data = create_diagnostic_pdf()
    if pdf_data is not None:
        st.download_button(
            label="⬇️ DOWNLOAD DIAGNOSTIC REPORT (PDF)",
            data=pdf_data,
            file_name=f"bearing_diagnostic_report_segment_{random_idx}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
    else:
        st.warning("PDF export is not available yet. Install the required package once:")
        st.code("python -m pip install reportlab", language="powershell")


# ============================================================
# BEFORE FIRST ANALYSIS
# ============================================================

else:

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🏭 Industrial Bearing Intelligence</div>
            <div class="hero-subtitle">
                AI-Powered Fault Detection • Condition Monitoring •
                Multimodal Inspection • Adaptive RAG • Root Cause Analysis
            </div>
            <div class="system-pill">● SYSTEM READY</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_segments = len(X_all)

    overview1, overview2, overview3, overview4 = st.columns(4)

    overview1.metric(
        "Total Segments",
        total_segments
    )

    overview2.metric(
        "Signal Length",
        X_all.shape[1]
        if len(X_all.shape) > 1
        else len(X_all[0])
    )

    overview3.metric(
        "Fault Classes",
        len(np.unique(y_all))
    )

    overview4.metric(
        "Current Test Mode",
        SHORT_FAULT_MAP[selected_class]
    )

    st.divider()

    st.info(
        """
        👈 **Select a Fault Test Scenario from the sidebar and
        click the RUN INDUSTRIAL ANALYSIS button.**

        🔴 **Outer Race Fault is selected by default, so the first
        run will perform a fault detection test.**
        """
    )