# Multimodal Fusion:
# Extract features from vibration, thermal image, and RAG text.
# Project all three modalities into a common 256D space.
# Concatenate them (256+256+256 = 768D) and fuse them
# through a neural network to generate a unified 512D context vector.
import os
import sys
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.append(CURRENT_DIR)

from model import Bearing1DCNN
from thermal_image_module import (
    load_thermal_model,
    extract_thermal_features
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "bearing_cnn.pth"
)

FUSION_DIM = 512
TEXT_FEATURE_DIM = 256


# ============================================================
# VIBRATION FEATURE EXTRACTOR
# ============================================================

class VibrationFeatureExtractor(nn.Module):
    """
    Extracts features from the trained 1D-CNN.

    Output:
        512-dimensional vibration feature vector.
    """

    def __init__(self):

        super().__init__()

        # Load original CNN model
        self.cnn = Bearing1DCNN(
            num_classes=4
        )

        self.cnn.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=DEVICE
            )
        )

        # Freeze CNN
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):

        # ----------------------------------------------------
        # IMPORTANT:
        # This assumes your Bearing1DCNN has a feature layer
        # named "features".
        # ----------------------------------------------------

        if hasattr(self.cnn, "features"):

            features = self.cnn.features(x)

            # Flatten CNN features
            features = features.view(
                features.size(0),
                -1
            )

        else:
            raise AttributeError(
                "❌ Bearing1DCNN does not have a 'features' layer. "
                "Send me model.py code so I can match the fusion "
                "module exactly."
            )

        return features


# ============================================================
# TEXT / RAG FEATURE ENCODER
# ============================================================

class SimpleTextEncoder:
    """
    Temporary lightweight text encoder.

    Converts RAG text into a fixed-size numeric vector.

    NOTE:
    Later this can be replaced by DeBERTa / SentenceTransformer.
    """

    def __init__(self, output_dim=TEXT_FEATURE_DIM):

        self.output_dim = output_dim

    def encode(self, text):

        # Convert text to UTF-8 bytes
        text_bytes = text.encode(
            "utf-8"
        )

        # Create fixed-size vector
        vector = np.zeros(
            self.output_dim,
            dtype=np.float32
        )

        # Simple hashing-based representation
        for i, byte_value in enumerate(text_bytes):

            index = i % self.output_dim

            vector[index] += (
                byte_value / 255.0
            )

        # Normalize
        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        return torch.tensor(
            vector,
            dtype=torch.float32
        )


# ============================================================
# MULTIMODAL FUSION NETWORK
# ============================================================

class MultimodalFusion(nn.Module):
    """
    Combines:

    1. Vibration features
    2. Thermal image features
    3. RAG/Text features

    Output:
        512D Unified Industrial Context Vector
    """

    def __init__(
        self,
        vibration_dim,
        image_dim=512,
        text_dim=TEXT_FEATURE_DIM,
        fusion_dim=FUSION_DIM
    ):

        super().__init__()

        self.vibration_projection = nn.Sequential(
            nn.Linear(vibration_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Combine all three modalities
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, fusion_dim),
            nn.ReLU()
        )

    def forward(
        self,
        vibration_features,
        image_features,
        text_features
    ):

        # Project each modality
        vibration = self.vibration_projection(
            vibration_features
        )

        image = self.image_projection(
            image_features
        )

        text = self.text_projection(
            text_features
        )

        # Concatenate
        combined = torch.cat(
            [vibration, image, text],
            dim=1
        )

        # Final unified context vector
        fused_features = self.fusion_layer(
            combined
        )

        return fused_features


# ============================================================
# PREPROCESS VIBRATION SIGNAL
# ============================================================

def preprocess_vibration_signal(signal):

    signal = np.asarray(
        signal,
        dtype=np.float32
    ).squeeze()

    # Same normalization as preprocess.py
    mean = np.mean(signal)
    std = np.std(signal) + 1e-8

    signal = (
        signal - mean
    ) / std

    # Final shape: (1, 1, 1024)
    signal = np.expand_dims(
        signal,
        axis=0
    )

    signal = np.expand_dims(
        signal,
        axis=0
    )

    return torch.tensor(
        signal,
        dtype=torch.float32
    )


# ============================================================
# COMPLETE MULTIMODAL FUSION PIPELINE
# ============================================================

def create_multimodal_context(
    vibration_signal,
    image_path,
    rag_text
):

    print("\n" + "=" * 65)
    print("🏭 MULTIMODAL INDUSTRIAL CONTEXT FUSION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. VIBRATION FEATURES
    # --------------------------------------------------------

    vibration_model = VibrationFeatureExtractor().to(
        DEVICE
    )

    vibration_model.eval()

    vibration_tensor = preprocess_vibration_signal(
        vibration_signal
    ).to(DEVICE)

    with torch.no_grad():

        vibration_features = vibration_model(
            vibration_tensor
        )

    vibration_dim = vibration_features.shape[1]

    print(
        f"📈 Vibration features: "
        f"{vibration_features.shape}"
    )

    # --------------------------------------------------------
    # 2. THERMAL IMAGE FEATURES
    # --------------------------------------------------------

    thermal_features = extract_thermal_features(
        image_path
    )

    image_features = torch.tensor(
        thermal_features,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    print(
        f"🌡️ Image features: "
        f"{image_features.shape}"
    )

    # --------------------------------------------------------
    # 3. TEXT / RAG FEATURES
    # --------------------------------------------------------

    text_encoder = SimpleTextEncoder()

    text_features = text_encoder.encode(
        rag_text
    ).unsqueeze(0).to(DEVICE)

    print(
        f"📄 Text features: "
        f"{text_features.shape}"
    )

    # --------------------------------------------------------
    # 4. FUSION
    # --------------------------------------------------------

    fusion_model = MultimodalFusion(
        vibration_dim=vibration_dim
    ).to(DEVICE)

    fusion_model.eval()

    with torch.no_grad():

        fused_context = fusion_model(
            vibration_features,
            image_features,
            text_features
        )

    print(
        f"🧠 Unified Industrial Context: "
        f"{fused_context.shape}"
    )

    return fused_context.cpu().numpy()[0]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("🚀 Testing Multimodal Fusion Module")

    # --------------------------------------------------------
    # 1. Load one vibration sample
    # --------------------------------------------------------

    data_path = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "X_segments.npy"
    )

    X_segments = np.load(
        data_path
    )

    vibration_signal = X_segments[0]

    # --------------------------------------------------------
    # 2. Thermal image path
    # --------------------------------------------------------

    TEST_IMAGE_PATH = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "thermal_images",
        "test_image.jpg"
    )

    # --------------------------------------------------------
    # 3. Sample RAG text
    # --------------------------------------------------------

    rag_text = """
    Bearing vibration analysis indicates a possible
    outer race defect. Inspect the bearing housing,
    lubrication condition, mounting alignment, and
    replace the damaged bearing if the fault severity
    exceeds the maintenance threshold.
    """

    # --------------------------------------------------------
    # Run fusion
    # --------------------------------------------------------

    try:

        fused_context = create_multimodal_context(
            vibration_signal,
            TEST_IMAGE_PATH,
            rag_text
        )

        print("\n" + "=" * 65)
        print("✅ MULTIMODAL FUSION SUCCESSFUL")
        print("=" * 65)

        print(
            f"Final context vector shape: "
            f"{fused_context.shape}"
        )

        print(
            f"Number of fused features: "
            f"{len(fused_context)}"
        )

        print(
            "\nFirst 10 values:"
        )

        print(
            fused_context[:10]
        )

    except Exception as error:

        print("\n❌ FUSION ERROR")
        print(error)

        print(
            "\n👉 If the error says "
            "'features layer not found', send me your "
            "src/model.py code."
        )