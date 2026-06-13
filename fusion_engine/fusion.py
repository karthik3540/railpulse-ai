"""
RailPulse AI — Fusion Engine
LHS (Likelihood of Health Score) computation + demo event generator
for Delhi-Agra railway corridor visualization.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict

np.random.seed(42)

# ── Defect Event Dataclass ─────────────────────────────────────────────────

@dataclass
class DefectEvent:
    """Represents a single defect detection event along the railway corridor."""
    lat: float
    lon: float
    asset_type: str          # "track" or "ohe"
    defect_class: str
    risk_index: float
    confidence: float
    severity: str            # "CRITICAL", "WARNING", "OK"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ── LHS Computation ────────────────────────────────────────────────────────

def compute_lhs(track_events: List[DefectEvent], ohe_events: List[DefectEvent]) -> Dict:
    """
    Compute Likelihood of Health Score (LHS) for track and OHE subsystems.

    Parameters
    ----------
    track_events : list of DefectEvent
    ohe_events : list of DefectEvent

    Returns
    -------
    dict with keys: track_lhs, ohe_lhs, composite_lhs, priority
    """
    # Track LHS: weighted average of risk indices
    if track_events:
        track_risks = [e.risk_index for e in track_events]
        track_severities = [e.severity for e in track_events]
        severity_weights = {"CRITICAL": 1.5, "WARNING": 1.0, "OK": 0.3}
        weights = [severity_weights.get(s, 1.0) for s in track_severities]
        track_lhs = float(np.average(track_risks, weights=weights))
    else:
        track_lhs = 0.0

    # OHE LHS
    if ohe_events:
        ohe_risks = [e.risk_index for e in ohe_events]
        ohe_severities = [e.severity for e in ohe_events]
        severity_weights = {"CRITICAL": 1.5, "WARNING": 1.0, "OK": 0.3}
        weights = [severity_weights.get(s, 1.0) for s in ohe_severities]
        ohe_lhs = float(np.average(ohe_risks, weights=weights))
    else:
        ohe_lhs = 0.0

    # Composite LHS: weighted combination (track is more critical)
    composite_lhs = 0.6 * track_lhs + 0.4 * ohe_lhs

    # Priority classification
    if composite_lhs > 70:
        priority = "IMMEDIATE ACTION"
    elif composite_lhs > 50:
        priority = "HIGH PRIORITY"
    elif composite_lhs > 30:
        priority = "MODERATE"
    else:
        priority = "ROUTINE"

    return {
        "track_lhs": round(track_lhs, 2),
        "ohe_lhs": round(ohe_lhs, 2),
        "composite_lhs": round(composite_lhs, 2),
        "priority": priority,
    }


# ── Demo Event Generator ──────────────────────────────────────────────────

TRACK_DEFECTS = ["normal", "ball_fault", "crack_defect", "corrugation", "misalignment"]
OHE_DEFECTS = ["normal_contact", "wire_wear", "tension_anomaly", "stagger_defect"]


def _severity(risk: float) -> str:
    if risk > 75:
        return "CRITICAL"
    elif risk > 40:
        return "WARNING"
    return "OK"


def generate_demo_events() -> tuple:
    """
    Generate 12 track + 8 OHE demo events along the Delhi-Agra railway corridor.

    Corridor: lat 28.6 → 27.2, lon 77.2 → 78.0

    Returns
    -------
    tuple : (track_events, ohe_events)
    """
    np.random.seed(42)

    # Delhi-Agra corridor coordinates
    lats = np.linspace(28.6, 27.2, 20)
    lons = np.linspace(77.2, 78.0, 20)

    base_time = datetime(2026, 6, 13, 8, 0, 0)

    # ── 12 Track events ───────────────────────────────────────────────
    track_events = []
    track_indices = np.random.choice(len(lats), 12, replace=False)
    for i, idx in enumerate(sorted(track_indices)):
        defect = np.random.choice(TRACK_DEFECTS)
        if defect == "normal":
            risk = np.random.uniform(5, 25)
            conf = np.random.uniform(85, 99)
        else:
            risk = np.random.uniform(30, 95)
            conf = np.random.uniform(70, 98)

        event = DefectEvent(
            lat=float(lats[idx] + np.random.uniform(-0.02, 0.02)),
            lon=float(lons[idx] + np.random.uniform(-0.02, 0.02)),
            asset_type="track",
            defect_class=defect,
            risk_index=round(risk, 2),
            confidence=round(conf, 2),
            severity=_severity(risk),
            timestamp=(base_time + timedelta(minutes=i * 15)).isoformat(),
        )
        track_events.append(event)

    # ── 8 OHE events ──────────────────────────────────────────────────
    ohe_events = []
    ohe_indices = np.random.choice(len(lats), 8, replace=False)
    for i, idx in enumerate(sorted(ohe_indices)):
        defect = np.random.choice(OHE_DEFECTS)
        if defect == "normal_contact":
            risk = np.random.uniform(5, 20)
            conf = np.random.uniform(88, 99)
        else:
            risk = np.random.uniform(35, 90)
            conf = np.random.uniform(72, 97)

        event = DefectEvent(
            lat=float(lats[idx] + np.random.uniform(-0.02, 0.02)),
            lon=float(lons[idx] + np.random.uniform(-0.02, 0.02)),
            asset_type="ohe",
            defect_class=defect,
            risk_index=round(risk, 2),
            confidence=round(conf, 2),
            severity=_severity(risk),
            timestamp=(base_time + timedelta(minutes=i * 20)).isoformat(),
        )
        ohe_events.append(event)

    return track_events, ohe_events
