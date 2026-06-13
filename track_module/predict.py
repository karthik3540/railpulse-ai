"""
RailPulse AI — Track Module Inference
Loads trained models and predicts defect class + risk from a vibration window.
"""

import numpy as np
import pickle
import os

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from track_module.features import extract_features

# ── Configuration ──────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

CLASS_NAMES = ["ball_fault", "corrugation", "crack_defect", "misalignment", "normal"]


def load_models(models_dir: str = MODELS_DIR):
    """
    Load the 3 trained model files (scaler, isolation forest, classifier)
    plus the label encoder.

    Returns
    -------
    tuple : (scaler, iso_forest, clf, label_encoder)
    """
    with open(os.path.join(models_dir, "track_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, "track_iso_forest.pkl"), "rb") as f:
        iso_forest = pickle.load(f)
    with open(os.path.join(models_dir, "track_xgb_classifier.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(models_dir, "track_label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return scaler, iso_forest, clf, le


def predict_track(window: np.ndarray, scaler, iso_forest, clf, le=None) -> dict:
    """
    Predict defect class and risk index from a raw vibration window.

    Parameters
    ----------
    window : np.ndarray
        1-D vibration signal (e.g. 256 samples).
    scaler : StandardScaler
    iso_forest : IsolationForest
    clf : XGBClassifier
    le : LabelEncoder, optional

    Returns
    -------
    dict with keys:
        defect_class, label_id, confidence, anomaly_score, risk_index, severity
    """
    # Extract features
    features = extract_features(window)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features_2d = features.reshape(1, -1)

    # Scale
    features_scaled = scaler.transform(features_2d)

    # Classification
    label_id = int(clf.predict(features_scaled)[0])
    proba = clf.predict_proba(features_scaled)[0]
    confidence = float(np.max(proba)) * 100  # percentage

    # Anomaly score: IsolationForest decision_function (lower = more anomalous)
    anomaly_raw = iso_forest.decision_function(features_scaled)[0]
    # Normalize to 0-100 scale (more negative → higher anomaly score)
    anomaly_score = float(np.clip((1 - anomaly_raw) * 50, 0, 100))

    # Defect class name
    if le is not None:
        defect_class = le.inverse_transform([label_id])[0]
    else:
        defect_class = CLASS_NAMES[label_id] if label_id < len(CLASS_NAMES) else f"class_{label_id}"

    # Risk index: weighted combination of confidence and anomaly
    if defect_class == "normal":
        risk_index = float(np.clip(anomaly_score * 0.3, 0, 100))
    else:
        risk_index = float(np.clip(confidence * 0.6 + anomaly_score * 0.4, 0, 100))

    # Severity
    if risk_index > 75:
        severity = "CRITICAL"
    elif risk_index > 40:
        severity = "WARNING"
    else:
        severity = "OK"

    return {
        "defect_class": defect_class,
        "label_id": label_id,
        "confidence": round(confidence, 2),
        "anomaly_score": round(anomaly_score, 2),
        "risk_index": round(risk_index, 2),
        "severity": severity,
    }
