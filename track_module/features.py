"""
RailPulse AI — FFT Feature Extractor
Extracts 18 time-domain and frequency-domain features from a vibration window.
Applies Butterworth bandpass filter (5–400 Hz) before extraction.
"""

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.stats import kurtosis, skew

# ── Filter Configuration ───────────────────────────────────────────────────
SAMPLE_RATE = 1000  # Hz
LOW_CUT = 5.0       # Hz
HIGH_CUT = 400.0    # Hz
FILTER_ORDER = 4


def _bandpass_filter(signal: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Apply Butterworth bandpass filter 5–400 Hz."""
    nyquist = fs / 2
    low = LOW_CUT / nyquist
    high = HIGH_CUT / nyquist
    sos = butter(FILTER_ORDER, [low, high], btype="band", output="sos")
    return sosfilt(sos, signal)


def extract_features(window: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract 18 features from a single vibration window.

    Parameters
    ----------
    window : np.ndarray
        1-D array of vibration amplitudes (e.g. 256 samples).
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        1-D array of 18 features.
    """
    # Apply bandpass filter
    x = _bandpass_filter(window, fs)

    # ── Time-domain features ───────────────────────────────────────────
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    crest_factor = peak / (rms + 1e-10)
    kurt = kurtosis(x)
    skewness = skew(x)
    std = np.std(x)
    impulse_factor = peak / (np.mean(np.abs(x)) + 1e-10)

    # Zero Crossing Rate (ZCR)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(x))) > 0)
    zcr = zero_crossings / len(x)

    # ── Frequency-domain features ──────────────────────────────────────
    N = len(x)
    fft_vals = np.fft.rfft(x)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Total power
    total_power = np.sum(fft_mag ** 2)

    # FFT statistics
    fft_max = np.max(fft_mag)
    fft_mean = np.mean(fft_mag)
    fft_std = np.std(fft_mag)

    # Spectral centroid
    spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)

    # Spectral spread
    spectral_spread = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * fft_mag) / (np.sum(fft_mag) + 1e-10)
    )

    # Peak frequency
    peak_freq = freqs[np.argmax(fft_mag)]

    # Band energies
    def band_energy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return np.sum(fft_mag[mask] ** 2)

    band_5_50 = band_energy(5, 50)
    band_50_150 = band_energy(50, 150)
    band_150_400 = band_energy(150, 400)

    # ── Stack all 18 features ──────────────────────────────────────────
    features = np.array([
        rms,                # 1
        peak,               # 2
        crest_factor,       # 3
        kurt,               # 4
        skewness,           # 5
        std,                # 6
        spectral_centroid,  # 7
        spectral_spread,    # 8
        peak_freq,          # 9
        band_5_50,          # 10
        band_50_150,        # 11
        band_150_400,       # 12
        impulse_factor,     # 13
        zcr,                # 14
        fft_max,            # 15
        fft_mean,           # 16
        fft_std,            # 17
        total_power,        # 18
    ])

    return features
