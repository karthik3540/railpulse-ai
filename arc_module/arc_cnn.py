"""
RailPulse AI — 1D-CNN Model for Arc Waveform Classification
ArcCNN: 5-layer 1D-CNN  |  ArcDataset: PyTorch Dataset for DataLoader
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class ArcCNN(nn.Module):
    """
    5-layer 1D-CNN for arc waveform classification.
    Input:  (batch, 1, 512)
    Output: (batch, 4)
    """

    def __init__(self, num_classes: int = 4):
        super(ArcCNN, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Layer 1: Conv1d(1, 32, 16) + BN + ReLU + MaxPool
            nn.Conv1d(1, 32, kernel_size=16, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # Layer 2: Conv1d(32, 64, 8) + BN + ReLU + MaxPool
            nn.Conv1d(32, 64, kernel_size=8, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # Layer 3: Conv1d(64, 128, 4) + BN + ReLU + AdaptiveAvgPool(8)
            nn.Conv1d(64, 128, kernel_size=4, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
        )

        # Classifier: 128 * 8 = 1024
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 1024)
        x = self.classifier(x)
        return x


class ArcDataset(Dataset):
    """
    PyTorch Dataset for arc waveform data.

    Parameters
    ----------
    signals : np.ndarray
        Shape (N, 512) — raw waveform samples.
    labels : np.ndarray
        Shape (N,) — integer class labels.
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        signal = self.signals[idx]
        # Normalize by max absolute value
        max_abs = np.max(np.abs(signal)) + 1e-10
        signal = signal / max_abs
        # Reshape to (1, 512) for Conv1d
        signal = signal.reshape(1, -1)
        return torch.from_numpy(signal), torch.tensor(self.labels[idx])
