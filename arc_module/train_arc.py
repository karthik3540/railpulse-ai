"""
RailPulse AI — Arc Module Training
Trains ArcCNN for 30 epochs with Adam + CosineAnnealingLR.
Saves model state dict to models/arc_cnn.pt.
"""

import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from arc_module.arc_cnn import ArcCNN, ArcDataset

np.random.seed(42)
torch.manual_seed(42)

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "arc", "all_classes.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
NUM_POINTS = 512
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading arc waveform data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} samples")

    signal_cols = [f"s_{i}" for i in range(NUM_POINTS)]
    X_raw = df[signal_cols].values.astype(np.float32)  # (12000, 512)
    y_labels = df["label"].values

    # ── Encode labels ──────────────────────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    print(f"  Classes: {list(le.classes_)}")

    # ── Train/test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ── DataLoaders ────────────────────────────────────────────────────
    train_dataset = ArcDataset(X_train, y_train)
    test_dataset = ArcDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model, Loss, Optimizer ─────────────────────────────────────────
    model = ArcCNN(num_classes=len(le.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nTraining ArcCNN on {DEVICE} for {EPOCHS} epochs...")
    print(f"  Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"  Batch size: {BATCH_SIZE}\n")

    # ── Training loop ──────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for signals, labels in train_loader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:2d}/{EPOCHS}]  Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.4f}")

    # ── Evaluate ───────────────────────────────────────────────────────
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(DEVICE)
            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"  TEST ACCURACY: {acc:.4f}  {'PASS' if acc > 0.85 else 'FAIL'} -- target >85%)")
    print(f"{'='*60}\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    # ── Save model ─────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "arc_cnn.pt")
    torch.save(model.state_dict(), model_path)

    # Save label encoder too
    import pickle
    with open(os.path.join(MODELS_DIR, "arc_label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"[OK] Model saved to {model_path}")
    print(f"   - arc_cnn.pt")
    print(f"   - arc_label_encoder.pkl")


if __name__ == "__main__":
    main()
