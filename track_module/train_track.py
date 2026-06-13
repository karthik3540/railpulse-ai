"""
RailPulse AI — Track Module Training
Trains XGBClassifier + IsolationForest + StandardScaler on vibration features.
Saves 3 .pkl files to models/ directory.
"""

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from track_module.features import extract_features

np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vibration", "all_classes.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
NUM_POINTS = 256


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading vibration data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} samples")

    signal_cols = [f"s_{i}" for i in range(NUM_POINTS)]
    X_raw = df[signal_cols].values  # (25000, 256)
    y_labels = df["label"].values

    # ── Extract features ───────────────────────────────────────────────
    print("Extracting 18 features per sample...")
    X_features = np.array([extract_features(row) for row in X_raw])
    print(f"  Feature matrix shape: {X_features.shape}")

    # Handle any NaN/Inf from feature extraction
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Encode labels ──────────────────────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    print(f"  Classes: {list(le.classes_)}")

    # ── Train/test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ── Scale features ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Train IsolationForest ──────────────────────────────────────────
    print("\nTraining IsolationForest for anomaly detection...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
    )
    iso_forest.fit(X_train_scaled)

    # ── Train XGBClassifier ────────────────────────────────────────────
    print("Training XGBClassifier (n_estimators=300, max_depth=6)...")
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    clf.fit(X_train_scaled, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  TEST ACCURACY: {acc:.4f}  ({'PASS' if acc > 0.85 else 'FAIL'} -- target >85%)")
    print(f"{'='*60}\n")

    print("Classification Report:")
    print(classification_report(
        y_test, y_pred, target_names=le.classes_
    ))

    # ── Save models ────────────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, "track_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "track_iso_forest.pkl"), "wb") as f:
        pickle.dump(iso_forest, f)
    with open(os.path.join(MODELS_DIR, "track_xgb_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODELS_DIR, "track_label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"[OK] Models saved to {MODELS_DIR}/")
    print(f"   - track_scaler.pkl")
    print(f"   - track_iso_forest.pkl")
    print(f"   - track_xgb_classifier.pkl")
    print(f"   - track_label_encoder.pkl")


if __name__ == "__main__":
    main()
