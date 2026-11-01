import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from model import Bearing1DCNN
from rag_engine import generate_rca_report, CLASS_MAPPING

st.set_page_config(page_title="Industrial RCA Dashboard", layout="wide", page_icon="⚙️")

# Dashboard Title
st.title("⚙️ Autonomous Industrial Root-Cause Analysis (RCA)")
st.markdown("**Domain:** Industrial AI / Industry 4.0 | **Model:** 1D-CNN + Adaptive RAG")
st.markdown("---")

# Load Model
@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Bearing1DCNN(num_classes=4).to(device)
    model_path = os.path.join("saved_models", "bearing_cnn.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    return model, device

model, device = load_trained_model()

# Sidebar Control Panel
st.sidebar.header("🕹️ Signal Input Panel")
data_path = os.path.join("dataset", "processed", "X_segments.npy")
labels_path = os.path.join("dataset", "processed", "y_labels.npy")

if os.path.exists(data_path) and os.path.exists(labels_path):
    X_data = np.load(data_path)
    y_data = np.load(labels_path)
    
    sample_idx = st.sidebar.slider("Select Sample Index for Diagnostics:", 0, len(X_data) - 1, 10)
    selected_signal = X_data[sample_idx]
    actual_label = CLASS_MAPPING.get(y_data[sample_idx], "Unknown")
    st.sidebar.info(f"**Ground Truth Label:** {actual_label}")
else:
    st.sidebar.error("Processed dataset not found! Generating dummy signal...")
    selected_signal = np.random.randn(1024)

# Run Diagnostics Button
if st.sidebar.button("🔍 Run Diagnostics", type="primary"):
    # Preprocess & Predict
    sig = selected_signal.astype(np.float32)
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    tensor_input = torch.tensor(sig_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_input)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_class = torch.max(probs, 1)

    predicted_id = int(pred_class.item())
    confidence_val = float(conf.item()) * 100
    predicted_fault = CLASS_MAPPING[predicted_id]

    # Layout Columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Vibration Waveform Analysis")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(selected_signal, color='#1f77b4', linewidth=0.8)
        ax.set_title(f"Time-Series Signal (Window = 1024 points)")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Amplitude (g)")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

    with col2:
        st.subheader("🏷️ Model Diagnostic Output")
        if predicted_id == 0:
            st.success(f"**Status:** {predicted_fault}")
        else:
            st.error(f"**Fault Detected:** {predicted_fault}")
            
        st.metric(label="Diagnostic Confidence Score", value=f"{confidence_val:.2f}%")

    st.markdown("---")
    st.subheader("📑 Automated Root-Cause Analysis & SOP Report")
    
    report_text = generate_rca_report(class_id=predicted_id, confidence_score=confidence_val)
    st.code(report_text, language="text")