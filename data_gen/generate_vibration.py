"""
RailPulse AI — Synthetic Vibration Data Generator
Generates 5000 samples/class × 5 classes = 25,000 total
Each sample: 256 data points at 1000 Hz (physics-informed sine + noise)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────
SAMPLE_RATE = 1000          # Hz
NUM_POINTS = 256            # data points per sample
SAMPLES_PER_CLASS = 5000
CLASSES = ["normal", "ball_fault", "crack_defect", "corrugation", "misalignment"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vibration")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_classes.csv")

t = np.linspace(0, NUM_POINTS / SAMPLE_RATE, NUM_POINTS, endpoint=False)


def generate_normal(n: int) -> np.ndarray:
    """Normal rail vibration: low-amplitude broadband noise + gentle hum."""
    samples = []
    for _ in range(n):
        base_freq = np.random.uniform(18, 25)
        amplitude = np.random.uniform(0.3, 0.6)
        signal = amplitude * np.sin(2 * np.pi * base_freq * t)
        signal += 0.05 * np.sin(2 * np.pi * 50 * t)           # mains hum
        signal += np.random.normal(0, 0.08, NUM_POINTS)        # sensor noise
        samples.append(signal)
    return np.array(samples)


def generate_ball_fault(n: int) -> np.ndarray:
    """Bearing ball fault: periodic impulse at ball-pass frequency."""
    samples = []
    for _ in range(n):
        bpfo = np.random.uniform(80, 120)       # ball-pass freq outer race
        amplitude = np.random.uniform(1.5, 3.0)
        mod_depth = np.random.uniform(0.6, 0.9)
        carrier = np.sin(2 * np.pi * 20 * t)
        modulation = 1 + mod_depth * np.sin(2 * np.pi * bpfo * t)
        signal = amplitude * carrier * modulation
        # add periodic impulses
        impulse_period = int(SAMPLE_RATE / bpfo * np.random.uniform(0.8, 1.2))
        impulse_period = max(impulse_period, 2)
        impulses = np.zeros(NUM_POINTS)
        impulses[::impulse_period] = np.random.uniform(2.0, 4.0)
        signal += impulses
        signal += np.random.normal(0, 0.15, NUM_POINTS)
        samples.append(signal)
    return np.array(samples)


def generate_crack_defect(n: int) -> np.ndarray:
    """Rail crack: sharp transient spikes + resonance ringing."""
    samples = []
    for _ in range(n):
        base_freq = np.random.uniform(40, 70)
        ring_freq = np.random.uniform(150, 300)
        signal = 0.4 * np.sin(2 * np.pi * base_freq * t)
        # ringing from crack resonance
        decay = np.exp(-np.random.uniform(8, 15) * t)
        signal += np.random.uniform(1.5, 2.5) * decay * np.sin(2 * np.pi * ring_freq * t)
        # random transient spikes
        n_spikes = np.random.randint(3, 8)
        spike_locs = np.random.choice(NUM_POINTS, n_spikes, replace=False)
        signal[spike_locs] += np.random.uniform(2.0, 5.0, n_spikes)
        signal += np.random.normal(0, 0.12, NUM_POINTS)
        samples.append(signal)
    return np.array(samples)


def generate_corrugation(n: int) -> np.ndarray:
    """Rail corrugation: strong periodic undulation at specific wavelength."""
    samples = []
    for _ in range(n):
        corr_freq = np.random.uniform(30, 60)
        amplitude = np.random.uniform(1.0, 2.0)
        harmonic_amp = np.random.uniform(0.3, 0.7)
        signal = amplitude * np.sin(2 * np.pi * corr_freq * t)
        signal += harmonic_amp * np.sin(2 * np.pi * 2 * corr_freq * t)  # 2nd harmonic
        signal += 0.2 * np.sin(2 * np.pi * 3 * corr_freq * t)          # 3rd harmonic
        # speed-dependent modulation
        mod_freq = np.random.uniform(3, 8)
        signal *= (1 + 0.3 * np.sin(2 * np.pi * mod_freq * t))
        signal += np.random.normal(0, 0.1, NUM_POINTS)
        samples.append(signal)
    return np.array(samples)


def generate_misalignment(n: int) -> np.ndarray:
    """Track misalignment: asymmetric vibration with strong 1× and 2× components."""
    samples = []
    for _ in range(n):
        fund_freq = np.random.uniform(15, 30)
        amp_1x = np.random.uniform(1.2, 2.0)
        amp_2x = np.random.uniform(0.8, 1.5)
        phase_shift = np.random.uniform(0, np.pi)
        signal = amp_1x * np.sin(2 * np.pi * fund_freq * t)
        signal += amp_2x * np.sin(2 * np.pi * 2 * fund_freq * t + phase_shift)
        signal += 0.3 * np.sin(2 * np.pi * 3 * fund_freq * t + 2 * phase_shift)
        # asymmetry — clipping on one side
        clip_level = np.random.uniform(1.5, 2.5)
        signal = np.clip(signal, -clip_level * 1.5, clip_level)
        signal += np.random.normal(0, 0.12, NUM_POINTS)
        samples.append(signal)
    return np.array(samples)


GENERATORS = {
    "normal": generate_normal,
    "ball_fault": generate_ball_fault,
    "crack_defect": generate_crack_defect,
    "corrugation": generate_corrugation,
    "misalignment": generate_misalignment,
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

    # Stack all samples: shape (25000, 256)
    all_data = np.vstack(all_data)

    # Create DataFrame with columns: s_0, s_1, ..., s_255, label
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
