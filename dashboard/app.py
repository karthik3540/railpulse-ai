"""
RailPulse AI — Streamlit Dashboard (Premium UI)
Main application with 5 tabs: Track Health | Arc Intelligence | Route Map | Report | Train Network
"""

import sys
import os

# ── Ensure repo root is on sys.path (for Streamlit Cloud deployment) ──────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
import io

from track_module.predict import load_models, predict_track
from arc_module.predict_arc import load_arc_model, predict_arc
from fusion_engine.fusion import generate_demo_events, compute_lhs, DefectEvent
from fusion_engine.network import generate_fleet_events, compute_consensus
from reports.generate_report import generate_report

np.random.seed(42)

# ── Model paths (relative to repo root) ────────────────────────────────────
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DEMO_DIR = os.path.join(REPO_ROOT, "demo_data")

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RailPulse AI — Railway Predictive Maintenance",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Dark Theme CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Base ─────────────────────────────────────── */
    .stApp {
        background: #060d1f;
        font-family: 'Inter', sans-serif;
    }

    /* ── Scrollbar ────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #060d1f; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #64FFDA, #536DFE);
        border-radius: 3px;
    }

    /* ── Sidebar ──────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 50%, #0a1628 100%);
        border-right: 1px solid rgba(100,255,218,0.1);
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #8892B0 !important;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        padding: 4px 0;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #64FFDA !important;
    }

    /* ── Headers ──────────────────────────────────── */
    h1, h2, h3 {
        color: #CCD6F6 !important;
        font-family: 'Inter', sans-serif !important;
    }
    h1 {
        background: linear-gradient(135deg, #64FFDA 0%, #536DFE 50%, #FF6B9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.5px;
    }

    /* ── Text ─────────────────────────────────────── */
    p, span, label, .stMarkdown { color: #8892B0 !important; }

    /* ── Glassmorphism Metric Cards ───────────────── */
    div[data-testid="stMetric"] {
        background: rgba(17, 34, 64, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(100,255,218,0.15);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(100,255,218,0.4);
        box-shadow: 0 12px 40px rgba(100,255,218,0.15), inset 0 1px 0 rgba(255,255,255,0.1);
    }
    div[data-testid="stMetric"] label {
        color: #64FFDA !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #E6F1FF !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* ── File Uploader ────────────────────────────── */
    div[data-testid="stFileUploader"] {
        background: rgba(17, 34, 64, 0.4);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(100,255,218,0.25);
        border-radius: 16px;
        padding: 10px;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #64FFDA;
        background: rgba(17, 34, 64, 0.6);
    }

    /* ── Buttons ──────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #64FFDA 0%, #00E5A0 100%) !important;
        color: #0A1628 !important;
        border: 2px solid #64FFDA !important;
        border-radius: 12px;
        font-weight: 800 !important;
        padding: 14px 32px !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(100,255,218,0.35);
        min-height: 50px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(100,255,218,0.55);
        background: linear-gradient(135deg, #00E5A0 0%, #64FFDA 100%) !important;
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #536DFE 0%, #7C4DFF 100%) !important;
        color: #FFFFFF !important;
        border: 2px solid #536DFE !important;
        border-radius: 12px;
        font-weight: 800 !important;
        padding: 14px 32px !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 20px rgba(83,109,254,0.4);
        min-height: 50px;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(83,109,254,0.6);
        background: linear-gradient(135deg, #7C4DFF 0%, #536DFE 100%) !important;
    }

    /* ── Select boxes ─────────────────────────────── */
    div[data-baseweb="select"] {
        background: rgba(17, 34, 64, 0.6) !important;
        border: 1px solid rgba(100,255,218,0.3) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] > div {
        color: #CCD6F6 !important;
    }

    /* ── File uploader button ─────────────────────── */
    div[data-testid="stFileUploader"] button {
        background: rgba(83,109,254,0.3) !important;
        color: #CCD6F6 !important;
        border: 1px solid rgba(83,109,254,0.5) !important;
        font-weight: 600 !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background: rgba(83,109,254,0.5) !important;
        border-color: #536DFE !important;
    }

    /* ── Slider ───────────────────────────────────── */
    div[data-testid="stSlider"] > div > div {
        color: #CCD6F6 !important;
    }

    /* ── Divider ──────────────────────────────────── */
    hr {
        border-color: rgba(100,255,218,0.1) !important;
        margin: 1.5rem 0;
    }

    /* ── Severity Badges ─────────────────────────── */
    .severity-critical {
        background: linear-gradient(135deg, #FF5370, #FF8A80);
        color: #fff;
        padding: 6px 20px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255,83,112,0.4);
        animation: pulse-critical 2s infinite;
    }
    .severity-warning {
        background: linear-gradient(135deg, #FFCB6B, #FFE082);
        color: #1a1a2e;
        padding: 6px 20px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255,203,107,0.3);
    }
    .severity-ok {
        background: linear-gradient(135deg, #C3E88D, #AED581);
        color: #1a1a2e;
        padding: 6px 20px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(195,232,141,0.3);
    }

    @keyframes pulse-critical {
        0%, 100% { box-shadow: 0 0 5px rgba(255,83,112,0.5); }
        50% { box-shadow: 0 0 25px rgba(255,83,112,0.8); }
    }

    /* ── Result Card ─────────────────────────────── */
    .result-card {
        background: rgba(17, 34, 64, 0.5);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(100,255,218,0.15);
        border-radius: 20px;
        padding: 28px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .result-card-critical {
        border-left: 4px solid #FF5370;
    }
    .result-card-warning {
        border-left: 4px solid #FFCB6B;
    }
    .result-card-ok {
        border-left: 4px solid #C3E88D;
    }

    /* ── Info Box ─────────────────────────────────── */
    .info-box {
        background: rgba(83,109,254,0.1);
        border: 1px solid rgba(83,109,254,0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .info-box p { color: #CCD6F6 !important; margin: 0; font-size: 0.9rem; }

    /* ── Sample Button Area ───────────────────────── */
    .sample-area {
        background: rgba(100,255,218,0.05);
        border: 1px dashed rgba(100,255,218,0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 8px 0;
    }

    /* ── Status badges ────────────────────────────── */
    .status-confirmed {
        background: linear-gradient(135deg, #64FFDA, #00E5A0);
        color: #0A1628;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .status-unconfirmed {
        background: rgba(255,255,255,0.1);
        color: #8892B0;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid rgba(255,255,255,0.15);
    }

    /* ── Plotly charts dark ────────────────────────── */
    .js-plotly-plot .plotly .modebar { display: none !important; }

    /* ── Data table ───────────────────────────────── */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Tab-hero gradient line ────────────────────── */
    .gradient-line {
        height: 3px;
        background: linear-gradient(90deg, #64FFDA, #536DFE, #FF6B9D, #536DFE, #64FFDA);
        background-size: 200% 100%;
        animation: shimmer 3s ease infinite;
        border-radius: 2px;
        margin: 8px 0 24px 0;
    }
    @keyframes shimmer {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)


# ── Plotly Theme ───────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,22,40,0.8)",
    font=dict(family="Inter", color="#8892B0"),
    title_font=dict(color="#CCD6F6", size=16),
    xaxis=dict(gridcolor="rgba(100,255,218,0.07)", zerolinecolor="rgba(100,255,218,0.1)"),
    yaxis=dict(gridcolor="rgba(100,255,218,0.07)", zerolinecolor="rgba(100,255,218,0.1)"),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ── Helper Functions ───────────────────────────────────────────────────────
def severity_badge(severity: str) -> str:
    cls = f"severity-{severity.lower()}"
    return f'<span class="{cls}">{severity}</span>'


def make_gauge(value, title, max_val=100, color_ranges=None):
    """Create a Plotly gauge chart."""
    if color_ranges is None:
        color_ranges = [
            [0, 30, "#C3E88D"], [30, 60, "#FFCB6B"],
            [60, 80, "#FF8A80"], [80, 100, "#FF5370"],
        ]
    steps = [dict(range=[r[0], r[1]], color=r[2]) for r in color_ranges]
    bar_color = "#64FFDA"
    for r in color_ranges:
        if r[0] <= value <= r[1]:
            bar_color = r[2]
            break

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title, font=dict(size=14, color="#CCD6F6")),
        number=dict(font=dict(size=32, color="#E6F1FF")),
        gauge=dict(
            axis=dict(range=[0, max_val], tickcolor="#8892B0", tickfont=dict(color="#8892B0")),
            bar=dict(color=bar_color, thickness=0.3),
            bgcolor="rgba(10,22,40,0.8)",
            bordercolor="rgba(100,255,218,0.1)",
            steps=steps,
            threshold=dict(line=dict(color="#64FFDA", width=2), value=value, thickness=0.8),
        ),
    ))
    fig.update_layout(
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        margin=dict(l=30, r=30, t=50, b=10),
    )
    return fig


def load_sample_csv(filename):
    """Load a sample CSV from demo_data/ folder."""
    path = os.path.join(DEMO_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── Model Loading with caching ─────────────────────────────────────────────
@st.cache_resource
def cached_load_track_models():
    required = ["track_scaler.pkl", "track_iso_forest.pkl", "track_xgb_classifier.pkl", "track_label_encoder.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        return None
    try:
        return load_models(MODELS_DIR)
    except Exception:
        return None

@st.cache_resource
def cached_load_arc_model():
    model_path = os.path.join(MODELS_DIR, "arc_cnn.pt")
    if not os.path.exists(model_path):
        return None
    try:
        return load_arc_model(model_path)
    except Exception:
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 24px 0 16px 0;">
        <div style="font-size: 2.5rem; margin-bottom: 4px;">🚂</div>
        <h1 style="font-size: 1.5rem; margin: 0; letter-spacing: -0.5px;
                   background: linear-gradient(135deg, #64FFDA, #536DFE);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            RailPulse AI
        </h1>
        <p style="color: #536DFE !important; font-size: 0.75rem; margin-top: 4px;
                  text-transform: uppercase; letter-spacing: 2px;">
            Predictive Maintenance
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔧 Track Health", "⚡ Arc Intelligence", "🗺️ Route Map", "📄 Report", "🚆 Train Network"],
        index=0,
    )

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding: 10px 0;">
        <p style="color: #536DFE !important; font-size: 0.7rem; letter-spacing: 1px;">
            FAR AWAY HACKATHON 2026<br>
            <span style="color: #64FFDA !important;">Delhi — Agra Corridor</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 1: TRACK HEALTH
# ══════════════════════════════════════════════════════════════════════════
if page == "🔧 Track Health":
    st.markdown("# 🔧 Track Health Analysis")
    st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)
    st.markdown("Upload a vibration CSV file to analyze track defects using **XGBoost + Isolation Forest**.")

    models = cached_load_track_models()
    if models is None:
        st.warning("Models not found. Run `python data_gen/generate_vibration.py` then `python track_module/train_track.py` first.")
        st.stop()

    scaler, iso_forest, clf, le = models

    # ── Sample data + Upload ──────────────────────────────────────────
    st.markdown("---")
    col_up, col_sample = st.columns([3, 1])

    with col_up:
        uploaded = st.file_uploader("Upload Vibration CSV", type=["csv"], key="track_upload")

    with col_sample:
        st.markdown("##### 📁 Sample Files")
        sample_files = {
            "Mixed (10 samples)": "vibration_mixed.csv",
            "Crack Defect": "vibration_crack_defect.csv",
            "Ball Fault": "vibration_ball_fault.csv",
            "Normal": "vibration_normal.csv",
        }
        selected_sample = st.selectbox("Load sample data:", list(sample_files.keys()), key="track_sample")
        if st.button("Load Sample", key="load_track_sample", use_container_width=True):
            sample_df = load_sample_csv(sample_files[selected_sample])
            if sample_df is not None:
                st.session_state["track_df"] = sample_df
                st.session_state["track_source"] = selected_sample

    # Determine active DataFrame
    df = None
    source_name = ""
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source_name = uploaded.name
    elif "track_df" in st.session_state:
        df = st.session_state["track_df"]
        source_name = st.session_state.get("track_source", "sample")

    if df is not None:
        st.success(f"Loaded **{len(df)} samples** from `{source_name}`")

        signal_cols = [c for c in df.columns if c.startswith("s_")]
        if not signal_cols:
            signal_cols = [c for c in df.columns if c != "label"]

        sample_idx = st.slider("Select sample index", 0, len(df) - 1, 0)
        window = df[signal_cols].iloc[sample_idx].values.astype(np.float64)

        # ── Plotly: Waveform + FFT ─────────────────────────────────
        col_wave, col_fft = st.columns(2)

        with col_wave:
            t_axis = np.arange(len(window)) / 1000
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_axis, y=window, mode="lines",
                line=dict(color="#64FFDA", width=1),
                fill="tozeroy", fillcolor="rgba(100,255,218,0.08)",
                name="Amplitude",
            ))
            fig.update_layout(
                title="Raw Vibration Waveform",
                xaxis_title="Time (s)", yaxis_title="Amplitude",
                **PLOTLY_LAYOUT, height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_fft:
            fft_vals = np.fft.rfft(window)
            fft_mag = np.abs(fft_vals)
            freqs = np.fft.rfftfreq(len(window), d=1.0 / 1000)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=freqs, y=fft_mag, mode="lines",
                line=dict(color="#536DFE", width=1),
                fill="tozeroy", fillcolor="rgba(83,109,254,0.08)",
                name="Magnitude",
            ))
            fig.update_layout(
                title="FFT Spectrum",
                xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
                xaxis=dict(range=[0, 500], **PLOTLY_LAYOUT["xaxis"]),
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis"},
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Predict ────────────────────────────────────────────────
        result = predict_track(window, scaler, iso_forest, clf, le)

        st.markdown("---")
        sev = result["severity"].lower()
        st.markdown(f"""
        <div class="result-card result-card-{sev}">
            <h3 style="margin-top:0;">Detection Results {severity_badge(result['severity'])}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Gauge charts
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(make_gauge(result["confidence"], "Confidence %"), use_container_width=True)
        with col2:
            st.plotly_chart(make_gauge(result["risk_index"], "Risk Index"), use_container_width=True)
        with col3:
            st.plotly_chart(make_gauge(abs(result["anomaly_score"]) * 10, "Anomaly Score", max_val=100), use_container_width=True)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Defect Class", result["defect_class"].replace("_", " ").title())
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1f}%")
        with col3:
            st.metric("Risk Index", f"{result['risk_index']:.1f}")
        with col4:
            st.metric("Severity", result["severity"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: ARC INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚡ Arc Intelligence":
    st.markdown("# ⚡ Arc Intelligence")
    st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)
    st.markdown("Upload an arc waveform CSV to analyze OHE defects using **1D-CNN deep learning**.")

    arc_models = cached_load_arc_model()
    if arc_models is None:
        st.warning("Models not found. Run `python data_gen/generate_arc.py` then `python arc_module/train_arc.py` first.")
        st.stop()

    model, le = arc_models

    # ── Sample data + Upload ──────────────────────────────────────────
    st.markdown("---")
    col_up, col_sample = st.columns([3, 1])

    with col_up:
        uploaded = st.file_uploader("Upload Arc Waveform CSV", type=["csv"], key="arc_upload")

    with col_sample:
        st.markdown("##### 📁 Sample Files")
        sample_files = {
            "Mixed (12 samples)": "arc_mixed.csv",
            "Wire Wear": "arc_wire_wear.csv",
            "Tension Anomaly": "arc_tension_anomaly.csv",
            "Normal Contact": "arc_normal_contact.csv",
        }
        selected_sample = st.selectbox("Load sample data:", list(sample_files.keys()), key="arc_sample")
        if st.button("Load Sample", key="load_arc_sample", use_container_width=True):
            sample_df = load_sample_csv(sample_files[selected_sample])
            if sample_df is not None:
                st.session_state["arc_df"] = sample_df
                st.session_state["arc_source"] = selected_sample

    # Determine active DataFrame
    df = None
    source_name = ""
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source_name = uploaded.name
    elif "arc_df" in st.session_state:
        df = st.session_state["arc_df"]
        source_name = st.session_state.get("arc_source", "sample")

    if df is not None:
        st.success(f"Loaded **{len(df)} samples** from `{source_name}`")

        signal_cols = [c for c in df.columns if c.startswith("s_")]
        if not signal_cols:
            signal_cols = [c for c in df.columns if c != "label"]

        sample_idx = st.slider("Select sample index", 0, len(df) - 1, 0)
        waveform = df[signal_cols].iloc[sample_idx].values.astype(np.float64)

        # ── Plotly: Waveform + Spectrogram ─────────────────────────
        col_wave, col_spec = st.columns(2)

        with col_wave:
            t_axis = np.arange(len(waveform)) / 10000
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_axis, y=waveform, mode="lines",
                line=dict(color="#FFCB6B", width=1),
                fill="tozeroy", fillcolor="rgba(255,203,107,0.08)",
            ))
            fig.update_layout(
                title="Raw Arc Waveform",
                xaxis_title="Time (s)", yaxis_title="Amplitude",
                **PLOTLY_LAYOUT, height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_spec:
            nperseg = min(64, len(waveform))
            f_stft, t_stft, Zxx = stft(waveform, fs=10000, nperseg=nperseg, noverlap=nperseg // 2)
            fig = go.Figure(go.Heatmap(
                z=np.abs(Zxx), x=t_stft, y=f_stft,
                colorscale="Inferno", showscale=False,
            ))
            fig.update_layout(
                title="STFT Spectrogram",
                xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
                **PLOTLY_LAYOUT, height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Predict ────────────────────────────────────────────────
        result = predict_arc(waveform, model, le)

        st.markdown("---")
        sev = result["severity"].lower()
        st.markdown(f"""
        <div class="result-card result-card-{sev}">
            <h3 style="margin-top:0;">CNN Detection Results {severity_badge(result['severity'])}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Gauge charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(make_gauge(result["confidence"], "Confidence %"), use_container_width=True)
        with col2:
            st.plotly_chart(make_gauge(result["risk_index"], "Risk Index"), use_container_width=True)

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Defect Class", result["defect_class"].replace("_", " ").title())
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1f}%")
        with col3:
            st.metric("Severity", result["severity"])

        # ── Probability chart ──────────────────────────────────────
        st.markdown("### Class Probabilities")
        probs = result["all_probs"]
        classes = list(probs.keys())
        values = list(probs.values())
        colors = ["#64FFDA" if v == max(values) else "#536DFE" for v in values]

        fig = go.Figure(go.Bar(
            x=values, y=classes, orientation="h",
            marker=dict(color=colors, line=dict(color="#64FFDA", width=1)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="#CCD6F6"),
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 105], title="Probability (%)", **PLOTLY_LAYOUT["xaxis"]),
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis"},
            height=250, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: ROUTE MAP
# ══════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Route Map":
    st.markdown("# 🗺️ Delhi-Agra Corridor Route Map")
    st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)
    st.markdown("Interactive map showing defect events with **colour-coded severity markers**.")

    track_events, ohe_events = generate_demo_events()
    lhs = compute_lhs(track_events, ohe_events)

    # ── LHS Metric Cards ───────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Track LHS", f"{lhs['track_lhs']:.1f}")
    with col2:
        st.metric("OHE LHS", f"{lhs['ohe_lhs']:.1f}")
    with col3:
        st.metric("Composite LHS", f"{lhs['composite_lhs']:.1f}")
    with col4:
        st.metric("Priority", lhs["priority"])

    st.markdown("---")

    # ── Folium Map ─────────────────────────────────────────────────────
    center_lat = (28.6 + 27.2) / 2
    center_lon = (77.2 + 78.0) / 2
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="CartoDB dark_matter",
    )

    severity_colors = {"CRITICAL": "#FF5370", "WARNING": "#FFCB6B", "OK": "#C3E88D"}

    corridor_coords = [
        [28.6, 77.2], [28.3, 77.3], [27.9, 77.5],
        [27.5, 77.7], [27.2, 78.0],
    ]
    folium.PolyLine(
        corridor_coords, color="#64FFDA",
        weight=3, opacity=0.6, dash_array="10",
    ).add_to(m)

    for event in track_events:
        color = severity_colors.get(event.severity, "#C3E88D")
        popup_html = f"""
        <div style="font-family: Inter, sans-serif; min-width: 200px;">
            <h4 style="color: #1a1a2e; margin-bottom: 5px;">🔧 Track Defect</h4>
            <b>Class:</b> {event.defect_class}<br>
            <b>Risk:</b> {event.risk_index:.1f}<br>
            <b>Confidence:</b> {event.confidence:.1f}%<br>
            <b>Severity:</b> <span style="color:{color}; font-weight:bold;">{event.severity}</span><br>
            <b>Time:</b> {event.timestamp}
        </div>
        """
        folium.CircleMarker(
            location=[event.lat, event.lon],
            radius=max(6, event.risk_index / 8),
            color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Track: {event.defect_class} (Risk: {event.risk_index:.0f})",
        ).add_to(m)

    for event in ohe_events:
        color = severity_colors.get(event.severity, "#C3E88D")
        popup_html = f"""
        <div style="font-family: Inter, sans-serif; min-width: 200px;">
            <h4 style="color: #1a1a2e; margin-bottom: 5px;">⚡ OHE Defect</h4>
            <b>Class:</b> {event.defect_class}<br>
            <b>Risk:</b> {event.risk_index:.1f}<br>
            <b>Confidence:</b> {event.confidence:.1f}%<br>
            <b>Severity:</b> <span style="color:{color}; font-weight:bold;">{event.severity}</span><br>
            <b>Time:</b> {event.timestamp}
        </div>
        """
        folium.RegularPolygonMarker(
            location=[event.lat, event.lon],
            number_of_sides=4, radius=max(8, event.risk_index / 6),
            color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"OHE: {event.defect_class} (Risk: {event.risk_index:.0f})",
        ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: rgba(10,22,40,0.92); padding: 16px 22px; border-radius: 12px;
                border: 1px solid rgba(100,255,218,0.15); font-family: Inter, sans-serif;
                backdrop-filter: blur(10px);">
        <h4 style="color: #CCD6F6; margin: 0 0 10px 0; font-size: 13px;">Legend</h4>
        <p style="margin: 4px 0; color: #CCD6F6; font-size: 11px;">
            <span style="color:#FF5370;">&#9679;</span> CRITICAL &nbsp;
            <span style="color:#FFCB6B;">&#9679;</span> WARNING &nbsp;
            <span style="color:#C3E88D;">&#9679;</span> OK
        </p>
        <p style="margin: 4px 0; color: #8892B0; font-size: 10px;">
            ● Circle = Track &nbsp; ◆ Diamond = OHE
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=None, height=600)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4: REPORT
# ══════════════════════════════════════════════════════════════════════════
elif page == "📄 Report":
    st.markdown("# 📄 Maintenance Report")
    st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)
    st.markdown("Generate and download a comprehensive **PDF maintenance report** with ranked defect events.")

    track_events, ohe_events = generate_demo_events()
    lhs = compute_lhs(track_events, ohe_events)

    # LHS Gauge
    st.plotly_chart(make_gauge(
        lhs["composite_lhs"], "Composite LHS Score",
        color_ranges=[[0, 30, "#C3E88D"], [30, 50, "#FFCB6B"], [50, 70, "#FF8A80"], [70, 100, "#FF5370"]],
    ), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Track Events", len(track_events))
    with col2:
        st.metric("OHE Events", len(ohe_events))
    with col3:
        st.metric("Composite LHS", f"{lhs['composite_lhs']:.1f}")
    with col4:
        st.metric("Priority", lhs["priority"])

    st.markdown("---")

    all_events = list(track_events) + list(ohe_events)
    all_events.sort(key=lambda e: e.risk_index, reverse=True)

    events_data = []
    for e in all_events:
        events_data.append({
            "Type": e.asset_type.upper(),
            "Defect Class": e.defect_class,
            "Risk Index": e.risk_index,
            "Confidence (%)": e.confidence,
            "Severity": e.severity,
            "Location": f"({e.lat:.3f}, {e.lon:.3f})",
        })

    st.dataframe(pd.DataFrame(events_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    if st.button("Generate PDF Report", use_container_width=True):
        with st.spinner("Generating PDF report..."):
            pdf_bytes = generate_report(track_events, ohe_events, lhs)

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="railpulse_maintenance_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.success("Report generated successfully!")


# ══════════════════════════════════════════════════════════════════════════
# TAB 5: TRAIN NETWORK
# ══════════════════════════════════════════════════════════════════════════
elif page == "🚆 Train Network":
    st.markdown("# 🚆 Distributed Train Network")
    st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)
    st.markdown(
        "Every train is an independent sensor node. The **consensus engine** "
        "cross-validates detections across the fleet, confirming real defects "
        "and filtering false positives."
    )

    fleet = generate_fleet_events(n_trains=4)
    consensus = compute_consensus(fleet)

    confirmed = [e for e in consensus if e.status == "CONFIRMED"]
    unconfirmed = [e for e in consensus if e.status == "UNCONFIRMED"]

    # ── Fleet overview ─────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fleet Size", len(fleet))
    with col2:
        st.metric("Total Detections", len(consensus))
    with col3:
        st.metric("Confirmed", len(confirmed))
    with col4:
        st.metric("Unconfirmed", len(unconfirmed))

    st.markdown("---")

    # ── Per-train breakdown ────────────────────────────────────────────
    st.markdown("### 🚂 Fleet Breakdown")
    train_cols = st.columns(len(fleet))
    for i, train_data in enumerate(fleet):
        with train_cols[i]:
            n_events = len(train_data["track_events"]) + len(train_data["ohe_events"])
            st.metric(f"Train {train_data['train_id']}", f"{n_events} events")

    st.markdown("---")

    # ── Consensus donut ────────────────────────────────────────────────
    col_chart, col_map = st.columns([1, 2])

    with col_chart:
        st.markdown("### Status Distribution")
        fig = go.Figure(go.Pie(
            labels=["CONFIRMED", "UNCONFIRMED"],
            values=[len(confirmed), len(unconfirmed)],
            hole=0.65,
            marker=dict(colors=["#64FFDA", "rgba(100,255,218,0.2)"],
                        line=dict(color="#0a1628", width=3)),
            textfont=dict(color="#CCD6F6", size=13),
            textinfo="label+value",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#8892B0"),
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            annotations=[dict(
                text=f"<b>{len(consensus)}</b><br>Total",
                x=0.5, y=0.5, font=dict(size=20, color="#CCD6F6"), showarrow=False,
            )],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_map:
        st.markdown("### Network Map")
        center_lat = (28.6 + 27.2) / 2
        center_lon = (77.2 + 78.0) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="CartoDB dark_matter")

        severity_colors = {"CRITICAL": "#FF5370", "WARNING": "#FFCB6B", "OK": "#C3E88D"}

        folium.PolyLine(
            [[28.6, 77.2], [28.3, 77.3], [27.9, 77.5], [27.5, 77.7], [27.2, 78.0]],
            color="#64FFDA", weight=3, opacity=0.6, dash_array="10",
        ).add_to(m)

        for event in consensus:
            color = severity_colors.get(event.severity, "#C3E88D")
            is_confirmed = event.status == "CONFIRMED"
            fill_opacity = 0.9 if is_confirmed else 0.25

            popup_html = f"""
            <div style="font-family: Inter, sans-serif; min-width: 200px;">
                <h4 style="color: #1a1a2e;">{event.status}</h4>
                <b>Defect:</b> {event.defect_class}<br>
                <b>Risk:</b> {event.risk_index:.1f}<br>
                <b>Trains:</b> {event.confirming_trains}<br>
                <b>Severity:</b> <span style="color:{color}; font-weight:bold;">{event.severity}</span>
            </div>
            """
            folium.CircleMarker(
                location=[event.lat, event.lon],
                radius=max(7, event.risk_index / 7),
                color=color, fill=True, fill_color=color,
                fill_opacity=fill_opacity, weight=2 if is_confirmed else 1,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{event.status}: {event.defect_class} (Trains: {event.confirming_trains})",
            ).add_to(m)

        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                    background: rgba(10,22,40,0.92); padding: 14px 18px; border-radius: 10px;
                    border: 1px solid rgba(100,255,218,0.15); font-family: Inter, sans-serif;">
            <p style="margin: 3px 0; color: #CCD6F6; font-size: 10px;">
                <span style="color:#FF5370;">&#9679;</span> CRITICAL &nbsp;
                <span style="color:#FFCB6B;">&#9679;</span> WARNING &nbsp;
                <span style="color:#C3E88D;">&#9679;</span> OK
            </p>
            <p style="margin: 3px 0; color: #8892B0; font-size: 9px;">
                Solid = CONFIRMED &nbsp; Faded = UNCONFIRMED
            </p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        st_folium(m, width=None, height=400)

    # ── Consensus Table ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Consensus Events")

    table_data = []
    for e in consensus:
        table_data.append({
            "Location": f"({e.lat:.3f}, {e.lon:.3f})",
            "Defect": e.defect_class,
            "Type": e.asset_type.upper(),
            "Status": e.status,
            "Trains": e.confirming_trains,
            "Risk": e.risk_index,
            "Confidence (%)": e.confidence,
            "Severity": e.severity,
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
