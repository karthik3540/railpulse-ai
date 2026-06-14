"""
Train-as-a-Sensor (TaaS) Distributed Network

Every train on the route becomes an independent sensing node. The consensus
engine cross-validates defect detections across multiple trains, reducing
false positives and increasing confidence for genuinely confirmed track/OHE
issues. As fleet size grows, coverage and accuracy scale automatically —
with zero additional infrastructure.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

from fusion_engine.fusion import (
    DefectEvent,
    generate_demo_events,
    _severity,
    TRACK_DEFECTS,
    OHE_DEFECTS,
)
from datetime import datetime, timedelta


# ── Consensus Event Dataclass ──────────────────────────────────────────────

@dataclass
class ConsensusEvent:
    """A defect event that has been cross-validated across multiple trains."""
    lat: float
    lon: float
    defect_class: str
    asset_type: str
    risk_index: float
    confidence: float
    severity: str
    status: str              # "CONFIRMED" or "UNCONFIRMED"
    confirming_trains: int   # number of trains that reported this defect


# ── Fleet Event Generator ──────────────────────────────────────────────────

def generate_fleet_events(n_trains: int = 4) -> List[Dict]:
    """
    Simulate n_trains independent passes over the Delhi-Agra corridor.

    Each train uses a different random seed so readings vary slightly.
    Each train produces its own set of track + OHE DefectEvent objects.

    Parameters
    ----------
    n_trains : int
        Number of trains in the fleet.

    Returns
    -------
    list of dict, each with keys:
        train_id, track_events, ohe_events
    """
    fleet = []

    for train_id in range(n_trains):
        # Each train gets a unique seed for variation
        seed = 100 + train_id
        np.random.seed(seed)

        # Delhi-Agra corridor coordinates
        lats = np.linspace(28.6, 27.2, 20)
        lons = np.linspace(77.2, 78.0, 20)
        base_time = datetime(2026, 6, 13, 6 + train_id * 2, 0, 0)

        # ── Track events ───────────────────────────────────────────
        track_events = []
        n_track = np.random.randint(8, 14)
        track_indices = np.random.choice(len(lats), min(n_track, len(lats)), replace=False)

        for i, idx in enumerate(sorted(track_indices)):
            defect = np.random.choice(TRACK_DEFECTS)
            if defect == "normal":
                risk = np.random.uniform(5, 25)
                conf = np.random.uniform(85, 99)
            else:
                risk = np.random.uniform(30, 95)
                conf = np.random.uniform(70, 98)

            event = DefectEvent(
                lat=float(lats[idx] + np.random.uniform(-0.01, 0.01)),
                lon=float(lons[idx] + np.random.uniform(-0.01, 0.01)),
                asset_type="track",
                defect_class=defect,
                risk_index=round(risk, 2),
                confidence=round(conf, 2),
                severity=_severity(risk),
                timestamp=(base_time + timedelta(minutes=i * 12)).isoformat(),
            )
            track_events.append(event)

        # ── OHE events ────────────────────────────────────────────
        ohe_events = []
        n_ohe = np.random.randint(5, 10)
        ohe_indices = np.random.choice(len(lats), min(n_ohe, len(lats)), replace=False)

        for i, idx in enumerate(sorted(ohe_indices)):
            defect = np.random.choice(OHE_DEFECTS)
            if defect == "normal_contact":
                risk = np.random.uniform(5, 20)
                conf = np.random.uniform(88, 99)
            else:
                risk = np.random.uniform(35, 90)
                conf = np.random.uniform(72, 97)

            event = DefectEvent(
                lat=float(lats[idx] + np.random.uniform(-0.01, 0.01)),
                lon=float(lons[idx] + np.random.uniform(-0.01, 0.01)),
                asset_type="ohe",
                defect_class=defect,
                risk_index=round(risk, 2),
                confidence=round(conf, 2),
                severity=_severity(risk),
                timestamp=(base_time + timedelta(minutes=i * 15)).isoformat(),
            )
            ohe_events.append(event)

        fleet.append({
            "train_id": train_id,
            "track_events": track_events,
            "ohe_events": ohe_events,
        })

    return fleet


# ── Consensus Engine ───────────────────────────────────────────────────────

def _location_key(lat: float, lon: float, precision: float = 0.02) -> tuple:
    """Quantize lat/lon to grid cells for proximity grouping."""
    return (round(lat / precision) * precision, round(lon / precision) * precision)


def compute_consensus(fleet_events: List[Dict]) -> List[ConsensusEvent]:
    """
    Cross-validate defect detections across multiple trains.

    Groups events by location proximity (~0.02 degrees) AND same defect_class.
    - If reported by 2+ different trains: status = 'CONFIRMED',
      confidence = avg + 10 (capped at 99)
    - If reported by only 1 train: status = 'UNCONFIRMED',
      confidence = original

    Parameters
    ----------
    fleet_events : list of dict
        Output from generate_fleet_events().

    Returns
    -------
    list of ConsensusEvent
    """
    # Group all events by (location_key, defect_class)
    groups = defaultdict(list)

    for train_data in fleet_events:
        train_id = train_data["train_id"]
        all_events = train_data["track_events"] + train_data["ohe_events"]

        for event in all_events:
            key = (_location_key(event.lat, event.lon), event.defect_class)
            groups[key].append((train_id, event))

    # Build consensus events
    consensus = []

    for (loc_key, defect_class), reports in groups.items():
        unique_trains = set(tid for tid, _ in reports)
        n_trains = len(unique_trains)

        # Average metrics across reports
        avg_lat = np.mean([e.lat for _, e in reports])
        avg_lon = np.mean([e.lon for _, e in reports])
        avg_risk = np.mean([e.risk_index for _, e in reports])
        avg_conf = np.mean([e.confidence for _, e in reports])
        asset_type = reports[0][1].asset_type

        if n_trains >= 2:
            status = "CONFIRMED"
            final_conf = min(avg_conf + 10, 99.0)
        else:
            status = "UNCONFIRMED"
            final_conf = avg_conf

        consensus.append(ConsensusEvent(
            lat=round(float(avg_lat), 4),
            lon=round(float(avg_lon), 4),
            defect_class=defect_class,
            asset_type=asset_type,
            risk_index=round(float(avg_risk), 2),
            confidence=round(float(final_conf), 2),
            severity=_severity(avg_risk),
            status=status,
            confirming_trains=n_trains,
        ))

    # Sort by risk_index descending
    consensus.sort(key=lambda e: e.risk_index, reverse=True)

    return consensus
