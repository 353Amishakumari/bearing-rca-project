import os
import sys
import numpy as np

# ============================================================
# PROJECT PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from predict import predict_fault, CLASS_NAMES

# RAG imports
try:
    from rag_engine import generate_rca_report, build_vector_store
    RAG_AVAILABLE = True
except Exception as e:
    print(f"⚠️ RAG module import warning: {e}")
    RAG_AVAILABLE = False


# ============================================================
# PROJECT PATHS
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "X_segments.npy"
)

LABEL_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "y_labels.npy"
)

THERMAL_IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "thermal_images",
    "test_image.jpg"
)

SOP_PATH = os.path.join(
    PROJECT_ROOT,
    "sop_docs",
    "Bearing_Maintenance_SOP.txt"
)


# ============================================================
# INDUSTRIAL FAULT DETAILS
# ============================================================

FAULT_DETAILS = {
    0: {
        "name": "Normal",
        "machine_status": "NORMAL",
        "bearing_health": 98.0,
        "alarm": "NO ALARM",
        "severity": "LOW"
    },

    1: {
        "name": "Inner Race Fault (IR007)",
        "machine_status": "FAULT DETECTED",
        "bearing_health": 68.0,
        "alarm": "WARNING",
        "severity": "MEDIUM"
    },

    2: {
        "name": "Ball Fault",
        "machine_status": "FAULT DETECTED",
        "bearing_health": 55.0,
        "alarm": "HIGH ALARM",
        "severity": "HIGH"
    },

    3: {
        "name": "Outer Race Fault",
        "machine_status": "FAULT DETECTED",
        "bearing_health": 44.5,
        "alarm": "CRITICAL ALARM",
        "severity": "CRITICAL"
    }
}


# ============================================================
# LOAD VIBRATION SAMPLE
# ============================================================

def load_vibration_sample(index=0):

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"❌ Vibration dataset not found:\n{DATA_PATH}"
        )

    X = np.load(DATA_PATH)

    if index < 0 or index >= len(X):
        raise IndexError(
            f"❌ Invalid index {index}. "
            f"Valid range: 0 to {len(X) - 1}"
        )

    return X[index]


# ============================================================
# GET ACTUAL DATASET LABEL
# ============================================================

def get_actual_label(index=0):

    if not os.path.exists(LABEL_PATH):
        return None

    y = np.load(LABEL_PATH)

    if index < 0 or index >= len(y):
        return None

    return int(y[index])


# ============================================================
# VIBRATION STATISTICS
# ============================================================

def calculate_signal_statistics(signal):

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    rms = float(
        np.sqrt(np.mean(signal ** 2))
    )

    peak = float(
        np.max(np.abs(signal))
    )

    mean_amplitude = float(
        np.mean(np.abs(signal))
    )

    std = float(
        np.std(signal)
    )

    return {
        "mean_amplitude": round(mean_amplitude, 4),
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "std": round(std, 4)
    }


# ============================================================
# INDUSTRIAL SENSOR CONTEXT
# ============================================================

def generate_sensor_data(
    predicted_class,
    signal_stats
):

    if predicted_class == 0:

        motor_speed = 1750
        temperature = 55.0
        health = 98.0

    elif predicted_class == 1:

        motor_speed = 1725
        temperature = 68.5
        health = 68.0

    elif predicted_class == 2:

        motor_speed = 1705
        temperature = 75.2
        health = 55.0

    else:

        motor_speed = 1681
        temperature = 87.6
        health = 44.5

    return {
        "motor_speed": motor_speed,
        "temperature": temperature,
        "vibration_rms": signal_stats["rms"],
        "peak_vibration": signal_stats["peak"],
        "bearing_health": health
    }


# ============================================================
# LOAD SOP TEXT
# ============================================================

def load_sop_text():

    if not os.path.exists(SOP_PATH):

        return """
Industrial Bearing Maintenance SOP

1. Inspect bearing housing for physical damage.
2. Check bearing alignment.
3. Inspect lubrication condition.
4. Check mounting bolts and mechanical looseness.
5. Measure vibration levels.
6. Inspect bearing race and rolling elements.
7. Replace damaged bearing components if required.
8. Perform post-maintenance vibration validation.
"""

    try:

        with open(
            SOP_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    except Exception:

        return """
Industrial bearing maintenance procedure unavailable.
Perform physical inspection of the bearing assembly.
"""


# ============================================================
# MULTIMODAL ANALYSIS
# ============================================================

def run_multimodal_analysis(
    vibration_signal,
    thermal_image_path,
    text
):

    result = {
        "status": "WARNING",
        "context_vector": None,
        "message": ""
    }

    try:

        # Import exact function from your multimodal_fusion.py
        from multimodal_fusion import (
            create_multimodal_context
        )

        context = create_multimodal_context(
            vibration_signal=vibration_signal,
            image_path=thermal_image_path,
            rag_text=text
        )

        result["status"] = "SUCCESS"
        result["context_vector"] = context
        result["message"] = (
            "Multimodal fusion completed successfully."
        )

    except Exception as e:

        result["status"] = "WARNING"
        result["message"] = str(e)

        print(
            f"⚠️ Multimodal fusion warning: {e}"
        )

    return result


# ============================================================
# RAG ROOT CAUSE ANALYSIS
# ============================================================

def run_rag_analysis(
    predicted_class,
    predicted_fault,
    confidence,
    sensor_data
):

    # --------------------------------------------------------
    # Fallback report
    # --------------------------------------------------------

    fallback_report = f"""
INDUSTRIAL ROOT CAUSE ANALYSIS

Predicted Fault:
{predicted_fault}

AI Confidence:
{confidence:.2f}%

Motor Speed:
{sensor_data['motor_speed']} RPM

Bearing Temperature:
{sensor_data['temperature']} °C

Vibration RMS:
{sensor_data['vibration_rms']}

Peak Vibration:
{sensor_data['peak_vibration']}

Recommended Actions:
1. Inspect the bearing assembly.
2. Check bearing housing and mounting condition.
3. Inspect lubrication condition.
4. Check mechanical alignment.
5. Verify bearing race condition.
6. Replace damaged components if required.
7. Perform vibration validation after maintenance.
"""

    # --------------------------------------------------------
    # If RAG unavailable
    # --------------------------------------------------------

    if not RAG_AVAILABLE:

        return fallback_report

    try:

        # Build/load ChromaDB vector store
        vector_db = build_vector_store()

        # Correct function arguments
        report = generate_rca_report(
            class_id=predicted_class,
            confidence_score=confidence,
            vector_db=vector_db
        )

        return report

    except Exception as e:

        print(
            f"⚠️ RAG analysis warning: {e}"
        )

        return fallback_report


# ============================================================
# MAIN INDUSTRIAL AI AGENT
# ============================================================

def run_industrial_agent(
    dataset_index=0,
    thermal_image_path=None,
    run_multimodal=True
):

    print("\n" + "=" * 70)
    print("🏭 INDUSTRIAL MULTIMODAL BEARING RCA AGENT")
    print("=" * 70)

    # ========================================================
    # STEP 1: LOAD VIBRATION DATA
    # ========================================================

    print(
        "\n📡 STEP 1: Loading vibration sensor data..."
    )

    vibration_signal = load_vibration_sample(
        dataset_index
    )

    actual_class = get_actual_label(
        dataset_index
    )

    print(
        f"✅ Vibration segment loaded: "
        f"shape {vibration_signal.shape}"
    )

    # ========================================================
    # STEP 2: SIGNAL STATISTICS
    # ========================================================

    print(
        "\n📊 STEP 2: Calculating vibration statistics..."
    )

    signal_stats = calculate_signal_statistics(
        vibration_signal
    )

    print(
        f"   RMS  : {signal_stats['rms']}"
    )

    print(
        f"   Peak : {signal_stats['peak']}"
    )

    # ========================================================
    # STEP 3: CNN PREDICTION
    # ========================================================

    print(
        "\n🧠 STEP 3: Running CNN fault prediction..."
    )

    prediction = predict_fault(
        vibration_signal
    )

    predicted_class = prediction["class_id"]

    predicted_fault = prediction["fault"]

    confidence = prediction["confidence"]

    print(
        f"✅ Predicted Fault: {predicted_fault}"
    )

    print(
        f"🎯 CNN Confidence: {confidence}%"
    )

    # ========================================================
    # STEP 4: INDUSTRIAL SENSOR CONTEXT
    # ========================================================

    print(
        "\n⚙️ STEP 4: Creating industrial sensor context..."
    )

    sensor_data = generate_sensor_data(
        predicted_class,
        signal_stats
    )

    print(
        f"   Motor Speed       : "
        f"{sensor_data['motor_speed']} RPM"
    )

    print(
        f"   Temperature       : "
        f"{sensor_data['temperature']} °C"
    )

    print(
        f"   Bearing Health    : "
        f"{sensor_data['bearing_health']}%"
    )

    # ========================================================
    # STEP 5: LOAD SOP TEXT
    # ========================================================

    print(
        "\n📄 STEP 5: Loading maintenance SOP..."
    )

    sop_text = load_sop_text()

    print(
        "✅ SOP context loaded."
    )

    # ========================================================
    # STEP 6: MULTIMODAL FUSION
    # ========================================================

    if thermal_image_path is None:

        thermal_image_path = THERMAL_IMAGE_PATH

    multimodal_result = {
        "status": "SKIPPED",
        "context_vector": None,
        "message": "Multimodal analysis disabled."
    }

    if run_multimodal:

        print(
            "\n🌐 STEP 6: Running multimodal fusion..."
        )

        multimodal_result = run_multimodal_analysis(
            vibration_signal=vibration_signal,
            thermal_image_path=thermal_image_path,
            text=sop_text
        )

        print(
            f"   Multimodal Status: "
            f"{multimodal_result['status']}"
        )

        print(
            f"   {multimodal_result['message']}"
        )

    # ========================================================
    # STEP 7: ADAPTIVE RAG RCA
    # ========================================================

    print(
        "\n🔎 STEP 7: Running Adaptive RAG "
        "Root Cause Analysis..."
    )

    rca_report = run_rag_analysis(
        predicted_class=predicted_class,
        predicted_fault=predicted_fault,
        confidence=confidence,
        sensor_data=sensor_data
    )

    print(
        "✅ RCA report generated."
    )

    # ========================================================
    # FINAL INDUSTRIAL STATUS
    # ========================================================

    fault_info = FAULT_DETAILS.get(
        predicted_class,
        FAULT_DETAILS[0]
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_result = {

        # Dataset information
        "dataset_index": dataset_index,

        "actual_class": actual_class,

        "actual_fault": (
            CLASS_NAMES.get(actual_class)
            if actual_class is not None
            else None
        ),

        # CNN prediction
        "predicted_class": predicted_class,

        "predicted_fault": predicted_fault,

        "confidence": confidence,

        "probabilities": prediction["probabilities"],

        # Industrial status
        "machine_status": fault_info[
            "machine_status"
        ],

        "alarm_status": fault_info[
            "alarm"
        ],

        "severity": fault_info[
            "severity"
        ],

        # Sensor data
        "sensor_data": sensor_data,

        # Signal statistics
        "signal_statistics": signal_stats,

        # Multimodal
        "multimodal_status": multimodal_result[
            "status"
        ],

        "multimodal_context": multimodal_result[
            "context_vector"
        ],

        "multimodal_message": multimodal_result[
            "message"
        ],

        # RCA
        "rca_report": rca_report
    }

    print("\n" + "=" * 70)
    print("🎉 INDUSTRIAL AI AGENT PIPELINE COMPLETED")
    print("=" * 70)

    return final_result


# ============================================================
# DISPLAY FINAL REPORT
# ============================================================

def print_final_report(result):

    print("\n")
    print("=" * 70)
    print("🏭 FINAL INDUSTRIAL DIAGNOSTIC REPORT")
    print("=" * 70)

    print(
        f"\n📌 Dataset Segment: "
        f"{result['dataset_index']}"
    )

    if result["actual_class"] is not None:

        print(
            f"📌 Actual Dataset Fault: "
            f"{result['actual_fault']}"
        )

    print(
        f"\n🧠 AI Predicted Fault: "
        f"{result['predicted_fault']}"
    )

    print(
        f"🎯 AI Confidence: "
        f"{result['confidence']}%"
    )

    print(
        f"\n🏭 Machine Status: "
        f"{result['machine_status']}"
    )

    print(
        f"🚨 Alarm Status: "
        f"{result['alarm_status']}"
    )

    print(
        f"⚠️ Severity: "
        f"{result['severity']}"
    )

    print("\n📊 SENSOR DATA")

    for key, value in result[
        "sensor_data"
    ].items():

        print(
            f"   {key}: {value}"
        )

    print(
        f"\n🌐 Multimodal Status: "
        f"{result['multimodal_status']}"
    )

    print(
        f"📝 Multimodal Message: "
        f"{result['multimodal_message']}"
    )

    if result["multimodal_context"] is not None:

        context = result["multimodal_context"]

        print(
            f"🧠 Fused Context Shape: "
            f"{np.asarray(context).shape}"
        )

    print(
        "\n🔎 ROOT CAUSE ANALYSIS"
    )

    print("-" * 70)

    print(
        result["rca_report"]
    )

    print("=" * 70)


# ============================================================
# RUN TEST
# ============================================================

if __name__ == "__main__":

    result = run_industrial_agent(
        dataset_index=2620,
        thermal_image_path=THERMAL_IMAGE_PATH,
        run_multimodal=True
    )

    print_final_report(result)