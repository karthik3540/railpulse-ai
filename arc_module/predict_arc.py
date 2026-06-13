"""
RailPulse AI — Arc Module Inference
Loads trained ArcCNN and predicts defect class from arc waveform.
"""

import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from arc_module.arc_cnn import ArcCNN

# ── Configuration ──────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
CLASS_NAMES = ["normal_contact", "stagger_defect", "tension_anomaly", "wire_wear"]


def load_arc_model(model_path: str = None, le_path: str = None):
    """
    Load the trained ArcCNN model and label encoder.

    Returns
    -------
    tuple : (model, label_encoder)
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "arc_cnn.pt")
    if le_path is None:
        le_path = os.path.join(MODELS_DIR, "arc_label_encoder.pkl")

    model = ArcCNN(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    le = None
    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            le = pickle.load(f)

    return model, le


def predict_arc(waveform: np.ndarray, model: ArcCNN, le=None) -> dict:
    """
    Predict defect class from a raw arc waveform.

    Parameters
    ----------
    waveform : np.ndarray
        1-D array of 512 samples.
    model : ArcCNN
        Trained model in eval mode.
    le : LabelEncoder, optional

    Returns
    -------
    dict with keys:
        defect_class, label_id, confidence, risk_index, severity, all_probs
    """
    # Normalize by max absolute value
    max_abs = np.max(np.abs(waveform)) + 1e-10
    normalized = (waveform / max_abs).astype(np.float32)

    # Reshape to (1, 1, 512) for Conv1d
    tensor = torch.from_numpy(normalized).reshape(1, 1, -1)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().numpy()

    label_id = int(np.argmax(probs))
    confidence = float(probs[label_id]) * 100

    # Defect class name
    if le is not None:
        defect_class = le.inverse_transform([label_id])[0]
    else:
        defect_class = CLASS_NAMES[label_id] if label_id < len(CLASS_NAMES) else f"class_{label_id}"

    # Risk index
    if defect_class == "normal_contact":
        risk_index = float(np.clip((1 - probs[label_id]) * 50, 0, 100))
    else:
        risk_index = float(np.clip(confidence * 0.7 + (1 - probs[0]) * 30, 0, 100))

    # Severity
    if risk_index > 75:
        severity = "CRITICAL"
    elif risk_index > 40:
        severity = "WARNING"
    else:
        severity = "OK"

    # All probabilities as dict
    if le is not None:
        all_probs = {le.inverse_transform([i])[0]: round(float(p) * 100, 2) for i, p in enumerate(probs)}
    else:
        all_probs = {CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)}

    return {
        "defect_class": defect_class,
        "label_id": label_id,
        "confidence": round(confidence, 2),
        "risk_index": round(risk_index, 2),
        "severity": severity,
        "all_probs": all_probs,
    }
