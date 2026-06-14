"""Quick test for network module"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion_engine.network import generate_fleet_events, compute_consensus

fleet = generate_fleet_events(n_trains=4)
print(f"Fleet size: {len(fleet)} trains")
for train in fleet:
    tid = train["train_id"]
    nt = len(train["track_events"])
    no = len(train["ohe_events"])
    print(f"  Train {tid}: {nt} track + {no} OHE events")

consensus = compute_consensus(fleet)
confirmed = [e for e in consensus if e.status == "CONFIRMED"]
unconfirmed = [e for e in consensus if e.status == "UNCONFIRMED"]

print(f"\nConsensus: {len(consensus)} total events")
print(f"  CONFIRMED:   {len(confirmed)}")
print(f"  UNCONFIRMED: {len(unconfirmed)}")
print(f"\nTop 5 by risk:")
for e in consensus[:5]:
    print(f"  [{e.status:12s}] {e.defect_class:20s} risk={e.risk_index:5.1f}  conf={e.confidence:5.1f}%  trains={e.confirming_trains}  sev={e.severity}")

print("\n[PASS] Network module works")
