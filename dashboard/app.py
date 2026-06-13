"""
RailPulse AI — Streamlit Dashboard
Main application with 4 tabs: Track Health | Arc Intelligence | Route Map | Report
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
import folium
from streamlit_folium import st_folium

from track_module.predict import load_models, predict_track
from arc_module.predict_arc import load_arc_model, predict_arc
from fusion_engine.fusion import generate_demo_events, compute_lhs, DefectEvent
from reports.generate_report import generate_report

np.random.seed(42)

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RailPulse AI — Railway Predictive Maintenance",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Theme CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main app background */
    .stApp {
        background-color: #0A1628;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #112240;
        border-right: 1px solid #1d3461;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #CCD6F6 !important;
        font-weight: 500;
    }

    /* Headers */
    h1, h2, h3 {
        color: #CCD6F6 !important;
        font-family: 'Inter', sans-serif !important;
    }
    h1 {
        background: linear-gradient(135deg, #64FFDA, #82B1FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }

    /* Text */
    p, span, label, .stMarkdown {
        color: #8892B0 !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #112240, #1d3461);
        border: 1px solid #1d3461;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(100,255,218,0.1);
    }
    div[data-testid="stMetric"] label {
        color: #8892B0 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #CCD6F6 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        background: #112240;
        border: 2px dashed #1d3461;
        border-radius: 12px;
        padding: 10px;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #64FFDA;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #64FFDA, #82B1FF);
        color: #0A1628 !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(100,255,218,0.3);
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #64FFDA, #82B1FF);
        color: #0A1628 !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Divider */
    hr {
        border-color: #1d3461 !important;
    }

    /* Radio buttons */
    .stRadio > div {
        gap: 8px;
    }

    /* Severity badges */
    .severity-critical {
        background: linear-gradient(135deg, #FF5370, #FF8A80);
        color: #fff;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        animation: pulse-critical 2s infinite;
    }
    .severity-warning {
        background: linear-gradient(135deg, #FFCB6B, #FFE082);
        color: #1a1a2e;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .severity-ok {
        background: linear-gradient(135deg, #C3E88D, #AED581);
        color: #1a1a2e;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }

    @keyframes pulse-critical {
        0%, 100% { box-shadow: 0 0 5px rgba(255,83,112,0.5); }
        50% { box-shadow: 0 0 20px rgba(255,83,112,0.8); }
    }

    /* Hero section */
    .hero-container {
        text-align: center;
        padding: 20px 0;
    }

    /* Stat card */
    .stat-card {
        background: linear-gradient(145deg, #112240, #1d3461);
        border: 1px solid #1d3461;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: Dark matplotlib figures ────────────────────────────────────────
def dark_fig(figsize=(10, 4)):
    """Create a matplotlib figure with dark theme."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#112240")
    ax.set_facecolor("#0A1628")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("#CCD6F6")
    for spine in ax.spines.values():
        spine.set_color("#1d3461")
    return fig, ax


def dark_fig_dual(figsize=(14, 4)):
    """Create a dual-subplot matplotlib figure with dark theme."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#112240")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0A1628")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("#CCD6F6")
        for spine in ax.spines.values():
            spine.set_color("#1d3461")
    return fig, ax1, ax2


def severity_badge(severity: str) -> str:
    """Return HTML for a colored severity badge."""
    cls = f"severity-{severity.lower()}"
    return f'<span class="{cls}">{severity}</span>'


# ── Model paths (relative to repo root) ────────────────────────────────────
MODELS_DIR = os.path.join(REPO_ROOT, "models")


# ── Model Loading with caching ─────────────────────────────────────────────
@st.cache_resource
def cached_load_track_models():
    required = ["track_scaler.pkl", "track_iso_forest.pkl", "track_xgb_classifier.pkl", "track_label_encoder.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        return None
    try:
        return load_models(MODELS_DIR)
    except Exception as e:
        return None

@st.cache_resource
def cached_load_arc_model():
    model_path = os.path.join(MODELS_DIR, "arc_cnn.pt")
    if not os.path.exists(model_path):
        return None
    try:
        return load_arc_model(model_path)
    except Exception as e:
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <h1 style="font-size: 1.8rem; margin-bottom: 4px;">🚂 RailPulse AI</h1>
        <p style="color: #64FFDA !important; font-size: 0.85rem; margin-top: 0;">
            Predictive Maintenance System
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔧 Track Health", "⚡ Arc Intelligence", "🗺️ Route Map", "📄 Report"],
        index=0,
    )

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding: 10px 0;">
        <p style="color: #555 !important; font-size: 0.75rem;">
            FAR AWAY Hackathon 2026<br>
            Delhi-Agra Railway Corridor
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 1: TRACK HEALTH
# ══════════════════════════════════════════════════════════════════════════
if page == "🔧 Track Health":
    st.markdown("# 🔧 Track Health Analysis")
    st.markdown("Upload a vibration CSV file to analyze track defects using **XGBoost + Isolation Forest**.")
    st.markdown("---")

    models = cached_load_track_models()
    if models is None:
        st.warning("Models not found. Run `python data_gen/generate_vibration.py` then `python track_module/train_track.py` first.")
        st.stop()

    scaler, iso_forest, clf, le = models

    uploaded = st.file_uploader("Upload Vibration CSV", type=["csv"], key="track_upload")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df)} samples from `{uploaded.name}`")

        # Use first sample for analysis
        signal_cols = [c for c in df.columns if c.startswith("s_")]
        if not signal_cols:
            # Try using all numeric columns except 'label'
            signal_cols = [c for c in df.columns if c != "label"]

        sample_idx = st.slider("Select sample index", 0, len(df) - 1, 0)
        window = df[signal_cols].iloc[sample_idx].values.astype(np.float64)

        # ── Plot: Raw Waveform + FFT ───────────────────────────────────
        fig, ax1, ax2 = dark_fig_dual(figsize=(14, 4))

        # Raw waveform
        t_axis = np.arange(len(window)) / 1000  # seconds
        ax1.plot(t_axis, window, color="#64FFDA", linewidth=0.8, alpha=0.9)
        ax1.fill_between(t_axis, window, alpha=0.15, color="#64FFDA")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Raw Vibration Waveform")

        # FFT
        fft_vals = np.fft.rfft(window)
        fft_mag = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(len(window), d=1.0 / 1000)
        ax2.plot(freqs, fft_mag, color="#82B1FF", linewidth=0.8)
        ax2.fill_between(freqs, fft_mag, alpha=0.15, color="#82B1FF")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.set_title("FFT Spectrum")
        ax2.set_xlim(0, 500)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Predict ────────────────────────────────────────────────────
        result = predict_track(window, scaler, iso_forest, clf, le)

        st.markdown("---")
        st.markdown("### 📊 Detection Results")

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Defect Class", result["defect_class"].replace("_", " ").title())
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1f}%")
        with col3:
            st.metric("Anomaly Score", f"{result['anomaly_score']:.1f}")
        with col4:
            st.metric("Risk Index", f"{result['risk_index']:.1f}")

        # Severity badge
        st.markdown(f"#### Severity: {severity_badge(result['severity'])}", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: ARC INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚡ Arc Intelligence":
    st.markdown("# ⚡ Arc Intelligence")
    st.markdown("Upload an arc waveform CSV to analyze OHE defects using **1D-CNN deep learning**.")
    st.markdown("---")

    arc_models = cached_load_arc_model()
    if arc_models is None:
        st.warning("Models not found. Run `python data_gen/generate_arc.py` then `python arc_module/train_arc.py` first.")
        st.stop()

    model, le = arc_models

    uploaded = st.file_uploader("Upload Arc Waveform CSV", type=["csv"], key="arc_upload")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df)} samples from `{uploaded.name}`")

        signal_cols = [c for c in df.columns if c.startswith("s_")]
        if not signal_cols:
            signal_cols = [c for c in df.columns if c != "label"]

        sample_idx = st.slider("Select sample index", 0, len(df) - 1, 0)
        waveform = df[signal_cols].iloc[sample_idx].values.astype(np.float64)

        # ── Plot: Spectrogram (STFT) ───────────────────────────────────
        fig, ax1, ax2 = dark_fig_dual(figsize=(14, 4))

        # Raw waveform
        t_axis = np.arange(len(waveform)) / 10000
        ax1.plot(t_axis, waveform, color="#FFCB6B", linewidth=0.8, alpha=0.9)
        ax1.fill_between(t_axis, waveform, alpha=0.15, color="#FFCB6B")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Raw Arc Waveform")

        # Spectrogram via STFT
        nperseg = min(64, len(waveform))
        f_stft, t_stft, Zxx = stft(waveform, fs=10000, nperseg=nperseg, noverlap=nperseg // 2)
        ax2.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading="gouraud", cmap="inferno")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_title("STFT Spectrogram")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Predict ────────────────────────────────────────────────────
        result = predict_arc(waveform, model, le)

        st.markdown("---")
        st.markdown("### 📊 CNN Detection Results")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Defect Class", result["defect_class"].replace("_", " ").title())
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1f}%")
        with col3:
            st.metric("Risk Index", f"{result['risk_index']:.1f}")
        with col4:
            st.metric("Severity", result["severity"])

        # Severity badge
        st.markdown(f"#### Severity: {severity_badge(result['severity'])}", unsafe_allow_html=True)

        # Class probabilities
        st.markdown("### 📈 Class Probabilities")
        probs_df = pd.DataFrame(
            list(result["all_probs"].items()),
            columns=["Class", "Probability (%)"],
        ).sort_values("Probability (%)", ascending=False)

        fig, ax = dark_fig(figsize=(8, 3))
        bars = ax.barh(probs_df["Class"], probs_df["Probability (%)"], color="#82B1FF", edgecolor="#64FFDA", linewidth=0.5)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Class Probabilities")
        ax.set_xlim(0, 100)
        for bar, val in zip(bars, probs_df["Probability (%)"]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", color="white", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: ROUTE MAP
# ══════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Route Map":
    st.markdown("# 🗺️ Delhi-Agra Corridor Route Map")
    st.markdown("Interactive map showing defect events with **colour-coded severity markers**.")
    st.markdown("---")

    # Generate demo events
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

    # Severity color mapping
    severity_colors = {
        "CRITICAL": "#FF5370",
        "WARNING": "#FFCB6B",
        "OK": "#C3E88D",
    }

    # Draw railway corridor line
    corridor_coords = [
        [28.6, 77.2],   # Delhi
        [28.3, 77.3],
        [27.9, 77.5],
        [27.5, 77.7],
        [27.2, 78.0],   # Agra
    ]
    folium.PolyLine(
        corridor_coords,
        color="#64FFDA",
        weight=3,
        opacity=0.6,
        dash_array="10",
    ).add_to(m)

    # Track events: CircleMarker
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
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Track: {event.defect_class} (Risk: {event.risk_index:.0f})",
        ).add_to(m)

    # OHE events: RegularPolygonMarker
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
            number_of_sides=4,
            radius=max(8, event.risk_index / 6),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"OHE: {event.defect_class} (Risk: {event.risk_index:.0f})",
        ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: rgba(10,22,40,0.9); padding: 15px 20px; border-radius: 10px;
                border: 1px solid #1d3461; font-family: Inter, sans-serif;">
        <h4 style="color: #CCD6F6; margin: 0 0 10px 0; font-size: 13px;">Legend</h4>
        <p style="margin: 4px 0; color: #CCD6F6; font-size: 11px;">
            <span style="color:#FF5370;">●</span> CRITICAL &nbsp;
            <span style="color:#FFCB6B;">●</span> WARNING &nbsp;
            <span style="color:#C3E88D;">●</span> OK
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
    st.markdown("Generate and download a comprehensive **PDF maintenance report** with ranked defect events.")
    st.markdown("---")

    track_events, ohe_events = generate_demo_events()
    lhs = compute_lhs(track_events, ohe_events)

    # Preview
    st.markdown("### 📊 Report Preview")

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

    # Event summary table
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

    st.dataframe(
        pd.DataFrame(events_data),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Generate and download PDF
    if st.button("🔄 Generate PDF Report", use_container_width=True):
        with st.spinner("Generating PDF report..."):
            pdf_bytes = generate_report(track_events, ohe_events, lhs)

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name="railpulse_maintenance_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.success("✅ Report generated successfully!")
