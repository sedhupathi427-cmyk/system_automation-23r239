"""
Robotic Arm Joint Health Monitoring System
Bonus Challenge - System Automation

Install dependency:
    pip install scikit-learn

Run:
    python joint_health_monitor.py

The program creates:
  - historical_joint_data.csv (120 labeled historical readings)
  - maintenance_log.txt (10 sample maintenance notes)
Then trains a model and demonstrates all four decisions.
"""

import csv
import random
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

random.seed(42)
HERE = Path(__file__).resolve().parent

# Demonstration thresholds. Confirm real limits with the robot manufacturer
# before using any similar logic on physical equipment.
DANGEROUS = {
    "torque_nm": 100.0,
    "temperature_c": 90.0,
    "vibration_mms": 12.0,
}
MONITOR_PROBABILITY = 0.35
MAINTENANCE_PROBABILITY = 0.65

JOINTS = ("shoulder", "elbow", "wrist")
FEATURES = [
    "shoulder_torque", "shoulder_temp", "shoulder_vibration",
    "elbow_torque", "elbow_temp", "elbow_vibration",
    "wrist_torque", "wrist_temp", "wrist_vibration",
]

def make_historical_dataset(path, rows=120):
    """Create synthetic labeled historical records; 1 means wear/failure risk."""
    fields = ["record_id"] + FEATURES + ["wear_label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(1, rows + 1):
            row = {"record_id": i}
            risk_score = 0
            for joint in JOINTS:
                # Normal operating ranges for this synthetic example.
                torque = random.uniform(20, 65)
                temp = random.uniform(30, 65)
                vibration = random.uniform(0.5, 5.0)

                # Some records simulate wear developing over time.
                if random.random() < 0.38:
                    torque += random.uniform(10, 28)
                    temp += random.uniform(8, 20)
                    vibration += random.uniform(1.5, 5.0)

                row[f"{joint}_torque"] = round(torque, 2)
                row[f"{joint}_temp"] = round(temp, 2)
                row[f"{joint}_vibration"] = round(vibration, 2)

                if torque >= 72 or temp >= 72 or vibration >= 7.0:
                    risk_score += 1

            row["wear_label"] = int(risk_score >= 2)
            writer.writerow(row)
    print(f"Created {rows} labeled historical readings: {path}")

def make_maintenance_log(path):
    notes = [
        "Arm A shoulder joint has a grinding noise during movement.",
        "Arm B elbow temperature has increased compared with last week.",
        "Arm C wrist vibration is higher than normal; inspect bearing.",
        "Arm D completed routine cleaning; no issue observed.",
        "Arm A technician noticed intermittent torque sensor readings.",
        "Arm E packaging labels were reordered for the next shift.",
        "Arm B lubricant appears low near the elbow joint.",
        "Arm F safety guard was cleaned and inspected.",
        "Arm C unusual rattling noise observed near the wrist.",
        "Arm D production meeting moved to 2 PM.",
    ]
    path.write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Created sample maintenance log with {len(notes)} entries: {path}")

def load_training_data(path):
    X, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append([float(row[name]) for name in FEATURES])
            y.append(int(row["wear_label"]))
    return X, y

def parse_maintenance_log(path):
    """Map arm letters to relevant notes. General notes are retained but not assigned."""
    arm_notes = {}
    relevant_words = (
        "grinding", "noise", "temperature", "vibration", "bearing",
        "torque", "lubricant", "rattling", "sensor", "overheat", "wear"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        lower = line.lower()
        relevant = any(word in lower for word in relevant_words)
        # Sample log convention: "Arm A", "Arm B", etc.
        arm = None
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if f"arm {letter.lower()}" in lower:
                arm = letter
                break
        if arm and relevant:
            arm_notes.setdefault(arm, []).append(line)
    return arm_notes

def flatten_readings(arm):
    values = []
    for joint in JOINTS:
        values.extend([
            float(arm[joint]["torque_nm"]),
            float(arm[joint]["temperature_c"]),
            float(arm[joint]["vibration_mms"]),
        ])
    return values

def immediate_safety_check(arm):
    """Deterministic rule check; it does not call the ML model."""
    reasons = []
    for joint in JOINTS:
        reading = arm[joint]
        if reading["torque_nm"] >= DANGEROUS["torque_nm"]:
            reasons.append(f"{joint}: torque {reading['torque_nm']} Nm >= {DANGEROUS['torque_nm']} Nm")
        if reading["temperature_c"] >= DANGEROUS["temperature_c"]:
            reasons.append(f"{joint}: temperature {reading['temperature_c']} C >= {DANGEROUS['temperature_c']} C")
        if reading["vibration_mms"] >= DANGEROUS["vibration_mms"]:
            reasons.append(f"{joint}: vibration {reading['vibration_mms']} mm/s >= {DANGEROUS['vibration_mms']} mm/s")
    return reasons

def detect_spoofing(arm):
    """
    Heuristic consistency check:
    flag a very low torque reading when temperature and vibration are both high.
    This catches this particular inconsistent-reading pattern, not every attack.
    """
    alerts = []
    for joint in JOINTS:
        r = arm[joint]
        if r["torque_nm"] < 5 and r["temperature_c"] > 70 and r["vibration_mms"] > 8:
            alerts.append(
                f"{joint}: suspiciously low torque ({r['torque_nm']} Nm) "
                "while temperature and vibration are high"
            )
    return alerts

def analyze_arm(arm_id, arm, model, maintenance_notes):
    safety_reasons = immediate_safety_check(arm)
    spoof_alerts = detect_spoofing(arm)

    # Priority 1: hard safety rules always run before and independently of ML.
    if safety_reasons:
        return "EMERGENCY_STOP", 1.0, safety_reasons, spoof_alerts, maintenance_notes

    # A detected spoofing pattern is treated conservatively.
    if spoof_alerts:
        return "EMERGENCY_STOP", 1.0, ["Possible spoofed sensor reading"] + spoof_alerts, spoof_alerts, maintenance_notes

    probability = float(model.predict_proba([flatten_readings(arm)])[0][1])
    relevant_notes = maintenance_notes or []
    serious_note_words = ("grinding", "rattling", "bearing", "lubricant", "sensor", "torque")
    note_is_serious = any(
        word in note.lower()
        for note in relevant_notes
        for word in serious_note_words
    )

    if probability >= MAINTENANCE_PROBABILITY or note_is_serious:
        decision = "SCHEDULE_MAINTENANCE"
    elif probability >= MONITOR_PROBABILITY or relevant_notes:
        decision = "MONITOR"
    else:
        decision = "SAFE"

    reasons = [f"Predicted wear probability: {probability:.1%}"]
    if relevant_notes:
        reasons.append(f"{len(relevant_notes)} relevant maintenance note(s)")
    if note_is_serious:
        reasons.append("Maintenance note contains a high-concern keyword")
    return decision, probability, reasons, spoof_alerts, relevant_notes

def make_arm(shoulder, elbow, wrist):
    return {"shoulder": shoulder, "elbow": elbow, "wrist": wrist}

def normal_joint(torque=40, temp=45, vibration=2):
    return {"torque_nm": torque, "temperature_c": temp, "vibration_mms": vibration}

def main():
    dataset_path = HERE / "historical_joint_data.csv"
    log_path = HERE / "maintenance_log.txt"
    make_historical_dataset(dataset_path, rows=120)
    make_maintenance_log(log_path)

    X, y = load_training_data(dataset_path)
    model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
    model.fit(X, y)
    maintenance = parse_maintenance_log(log_path)

    # Four examples: SAFE, MONITOR, SCHEDULE_MAINTENANCE, EMERGENCY_STOP.
    examples = [
        ("Arm F", make_arm(normal_joint(35, 42, 1.5), normal_joint(40, 46, 2.0), normal_joint(32, 40, 1.2))),
        ("Arm D", make_arm(normal_joint(63, 65, 5.8), normal_joint(58, 61, 5.2), normal_joint(55, 60, 4.8))),
        ("Arm B", make_arm(normal_joint(78, 78, 7.5), normal_joint(74, 75, 7.2), normal_joint(65, 68, 5.5))),
        ("Arm G", make_arm(normal_joint(45, 48, 2), normal_joint(105, 55, 3), normal_joint(40, 45, 2))),
        # Spoof demonstration: low torque conflicts with high temperature/vibration.
        ("Arm H (spoof demo)", make_arm(normal_joint(2, 78, 9.5), normal_joint(45, 50, 2), normal_joint(42, 47, 2))),
    ]

    print("\n=== ROBOTIC ARM JOINT HEALTH REPORT ===")
    print("Safety thresholds (demo only): torque >= 100 Nm, temperature >= 90 C, vibration >= 12 mm/s")
    for arm_id, arm in examples:
        letter = arm_id.split()[1][0] if arm_id.startswith("Arm ") else ""
        notes = maintenance.get(letter, [])
        # Add a mild note for the MONITOR example; it is relevant but not a
        # high-concern keyword, so it should not force maintenance scheduling.
        if arm_id == "Arm D":
            notes = ["Arm D vibration is slightly above normal; recheck next shift."]
        decision, probability, reasons, spoof_alerts, relevant_notes = analyze_arm(
            arm_id, arm, model, notes
        )
        print(f"\n{arm_id}: {decision}")
        for reason in reasons:
            print(" -", reason)
        if spoof_alerts:
            print(" - SPOOF DEFENSE:", "; ".join(spoof_alerts))
        if relevant_notes:
            print(" - Maintenance notes:", " | ".join(relevant_notes))

if __name__ == "__main__":
    main()
