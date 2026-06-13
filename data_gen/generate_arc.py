"""
RailPulse AI — Synthetic Arc Waveform Data Generator
Generates 3000 samples/class × 4 classes = 12,000 total
Each sample: 512 data points at 10,000 Hz
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────
SAMPLE_RATE = 10000         # Hz
NUM_POINTS = 512            # data points per sample
SAMPLES_PER_CLASS = 3000
CLASSES = ["normal_contact", "wire_wear", "tension_anomaly", "stagger_defect"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "arc")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_classes.csv")

t = np.linspace(0, NUM_POINTS / SAMPLE_RATE, NUM_POINTS, endpoint=False)


def generate_normal_contact(n: int) -> np.ndarray:
    """Normal pantograph-catenary contact: smooth sinusoidal current draw."""
    samples = []
    for _ in range(n):
        freq = np.random.uniform(45, 55)        # ~50Hz power frequency
        amplitude = np.random.uniform(0.8, 1.2)
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        # small harmonics
        signal += 0.05 * np.sin(2 * np.pi * 3 * freq * t)
        signal += np.random.normal(0, 0.03, NUM_POINTS)
        samples.append(signal)
    return np.array(samples)


def generate_wire_wear(n: int) -> np.ndarray:
    """Wire wear: irregular contact causes amplitude modulation + micro-arcing bursts."""
    samples = []
    for _ in range(n):
        freq = np.random.uniform(45, 55)
        amplitude = np.random.uniform(0.9, 1.3)
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        # amplitude modulation (irregular contact)
        mod_freq = np.random.uniform(200, 500)
        mod_depth = np.random.uniform(0.3, 0.6)
        signal *= (1 + mod_depth * np.sin(2 * np.pi * mod_freq * t))
        # micro-arcing bursts
        n_bursts = np.random.randint(5, 15)
        for _ in range(n_bursts):
            start = np.random.randint(0, NUM_POINTS - 20)
            length = np.random.randint(5, 20)
            burst_freq = np.random.uniform(1000, 3000)
            burst_amp = np.random.uniform(0.3, 0.8)
            burst_t = np.arange(length) / SAMPLE_RATE
            signal[start:start + length] += burst_amp * np.sin(2 * np.pi * burst_freq * burst_t)
        signal += np.random.normal(0, 0.06, NUM_POINTS)
        samples.append(signal)
    return np.array(samples)


def generate_tension_anomaly(n: int) -> np.ndarray:
    """Tension anomaly: low-frequency envelope variation + contact bounce."""
    samples = []
    for _ in range(n):
        freq = np.random.uniform(45, 55)
        amplitude = np.random.uniform(0.8, 1.2)
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        # tension variation: slow envelope
        tension_freq = np.random.uniform(5, 20)
        tension_depth = np.random.uniform(0.4, 0.8)
        envelope = 1 + tension_depth * np.sin(2 * np.pi * tension_freq * t)
        signal *= envelope
        # contact bounce: intermittent zero-crossings
        n_bounces = np.random.randint(3, 8)
        for _ in range(n_bounces):
            start = np.random.randint(0, NUM_POINTS - 30)
            length = np.random.randint(10, 30)
            decay = np.exp(-np.linspace(0, 3, length))
            signal[start:start + length] *= decay
        signal += np.random.normal(0, 0.05, NUM_POINTS)
        samples.append(signal)
    return np.array(samples)


def generate_stagger_defect(n: int) -> np.ndarray:
    """Stagger defect: lateral oscillation causes periodic contact-loss arcs."""
    samples = []
    for _ in range(n):
        freq = np.random.uniform(45, 55)
        amplitude = np.random.uniform(0.9, 1.3)
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        # periodic contact loss at stagger frequency
        stagger_freq = np.random.uniform(1, 5)  # Hz — relates to train speed
        stagger_signal = np.sin(2 * np.pi * stagger_freq * t)
        # when stagger_signal > threshold → arc event
        threshold = np.random.uniform(0.3, 0.7)
        arc_mask = stagger_signal > threshold
        # add high-frequency arc noise where contact is lost
        arc_freq = np.random.uniform(2000, 5000)
        arc_amp = np.random.uniform(0.5, 1.5)
        arc_noise = arc_amp * np.sin(2 * np.pi * arc_freq * t) * arc_mask
        signal += arc_noise
        # add broadband noise during arcing
        signal += 0.3 * np.random.normal(0, 1, NUM_POINTS) * arc_mask
        signal += np.random.normal(0, 0.04, NUM_POINTS)
        samples.append(signal)
    return np.array(samples)


GENERATORS = {
    "normal_contact": generate_normal_contact,
    "wire_wear": generate_wire_wear,
    "tension_anomaly": generate_tension_anomaly,
    "stagger_defect": generate_stagger_defect,
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []
    all_labels = []

    for cls_name in CLASSES:
        print(f"Generating {SAMPLES_PER_CLASS} samples for class '{cls_name}'...")
        data = GENERATORS[cls_name](SAMPLES_PER_CLASS)
        all_data.append(data)
        all_labels.extend([cls_name] * SAMPLES_PER_CLASS)

    # Stack all samples: shape (12000, 512)
    all_data = np.vstack(all_data)

    # Create DataFrame with columns: s_0, s_1, ..., s_511, label
    columns = [f"s_{i}" for i in range(NUM_POINTS)]
    df = pd.DataFrame(all_data, columns=columns)
    df["label"] = all_labels

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved {len(df)} samples ({NUM_POINTS} points each) to {OUTPUT_FILE}")
    print(f"   Classes: {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
