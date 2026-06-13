<div align="center">

# 🚂 RailPulse AI

### Predictive Railway Maintenance System

> *"Every train is already a sensor. RailPulse AI just taught it to speak."*

**Built for the FAR AWAY Hackathon 2026** — Because the future of railways isn't about going faster. It's about knowing what's breaking before it breaks.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://railpulse-ai.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 🎯 What is RailPulse AI?

RailPulse AI is a **real-time defect detection and risk assessment platform** for the Delhi-Agra railway corridor. It combines classical ML (XGBoost) with deep learning (1D-CNN) to analyze vibration and arc waveform signals, identifying track and overhead equipment defects before they cause failures.

### The Problem
- **3,500+** derailments in India over the past decade
- Manual inspection covers only **~15%** of track per cycle
- OHE failures cause **40%** of electric traction delays

### Our Solution
- **Upload a vibration CSV** → instant XGBoost classification + risk scoring
- **Upload an arc waveform CSV** → 1D-CNN deep learning defect detection
- **Interactive route map** → colour-coded severity markers on Delhi-Agra corridor
- **PDF maintenance report** → ranked defect table for maintenance crews

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/railpulse-ai.git
cd railpulse-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data, train models, and launch
python data_gen/generate_vibration.py
python data_gen/generate_arc.py
cd track_module && python train_track.py && cd ..
cd arc_module && python train_arc.py && cd ..
streamlit run dashboard/app.py
```

🌐 **Live Demo**: [railpulse-ai.streamlit.app](https://railpulse-ai.streamlit.app)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT DASHBOARD (app.py)                     │
│  ┌────────────┐ ┌─────────────────┐ ┌──────────┐ ┌──────────────┐  │
│  │   Track     │ │  Arc            │ │  Route   │ │   Report     │  │
│  │   Health    │ │  Intelligence   │ │  Map     │ │   Generator  │  │
│  └──────┬─────┘ └───────┬─────────┘ └────┬─────┘ └──────┬───────┘  │
├─────────┼───────────────┼────────────────┼──────────────┼───────────┤
│  ┌──────▼─────┐  ┌──────▼─────────┐ ┌───▼──────┐ ┌────▼────────┐  │
│  │  XGBoost   │  │   1D-CNN       │ │ Fusion   │ │ ReportLab   │  │
│  │  +IsoForest│  │   ArcCNN       │ │ Engine   │ │ PDF Gen     │  │
│  │  +Scaler   │  │   +PyTorch     │ │ +LHS     │ │             │  │
│  └──────┬─────┘  └──────┬─────────┘ └───┬──────┘ └─────────────┘  │
│  ┌──────▼─────┐  ┌──────▼─────────┐ ┌───▼──────────────────────┐  │
│  │  18 FFT    │  │  ArcDataset    │ │ Folium Map               │  │
│  │  Features  │  │  +Normalize    │ │ Delhi → Agra Corridor    │  │
│  └────────────┘  └────────────────┘ └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

data_gen/ ──→ track_module/ ──→ fusion_engine/ ──→ dashboard/
              arc_module/   ──→ reports/       ──→
```

---

## 📂 Project Structure

```
railpulse-ai/
├── data_gen/
│   ├── generate_vibration.py    # 25K samples, 5 defect classes
│   └── generate_arc.py          # 12K samples, 4 OHE defect classes
├── track_module/
│   ├── features.py              # 18 FFT features + Butterworth filter
│   ├── train_track.py           # XGBoost + IsolationForest training
│   └── predict.py               # Track inference → risk dict
├── arc_module/
│   ├── arc_cnn.py               # 5-layer 1D-CNN architecture
│   ├── train_arc.py             # 30-epoch CNN training pipeline
│   └── predict_arc.py           # Arc inference → prediction dict
├── fusion_engine/
│   └── fusion.py                # LHS computation + demo event generator
├── reports/
│   └── generate_report.py       # ReportLab A4 PDF generator
├── dashboard/
│   └── app.py                   # Streamlit app — 4 tabs, dark theme
├── .streamlit/
│   └── config.toml              # Dark theme configuration
├── demo_crack.csv               # Demo: vibration test file
├── demo_wire_wear.csv           # Demo: arc waveform test file
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔬 Model Performance

| Model | Task | Accuracy | Features |
|-------|------|----------|----------|
| **XGBoost** | Track defect classification | **99.96%** | 18 FFT features, 300 estimators |
| **1D-CNN** | OHE defect classification | **100.0%** | Raw waveform, 30 epochs |
| **IsolationForest** | Anomaly detection | — | Unsupervised scoring |

### Track Defect Classes
`normal` · `ball_fault` · `crack_defect` · `corrugation` · `misalignment`

### OHE Defect Classes
`normal_contact` · `wire_wear` · `tension_anomaly` · `stagger_defect`

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML/DL** | XGBoost, Isolation Forest, PyTorch 1D-CNN |
| **Signal Processing** | SciPy FFT, Butterworth filters, STFT spectrograms |
| **Data** | NumPy, Pandas, physics-informed synthetic generation |
| **Visualization** | Streamlit, Folium, Matplotlib, Plotly |
| **Reporting** | ReportLab (A4 PDF with styled tables) |
| **Deployment** | Streamlit Cloud |

---

## 🗺️ Delhi-Agra Railway Corridor

The system monitors **~200 km** of track from New Delhi (28.6°N, 77.2°E) to Agra (27.2°N, 78.0°E), generating:
- **12 track defect events** with CircleMarkers
- **8 OHE defect events** with diamond-shaped RegularPolygonMarkers
- Colour-coded by severity: 🔴 CRITICAL · 🟡 WARNING · 🟢 OK

---

## 👥 Team

**FAR AWAY** — Hackathon 2026

---

<div align="center">
<i>Built with ❤️ for safer railways</i>
</div>
