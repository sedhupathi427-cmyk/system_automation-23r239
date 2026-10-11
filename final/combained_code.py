import csv
import random
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_FILE = PROJECT_DIR / "day2" / "day2_sample.csv"
MODEL_FILE = PROJECT_DIR / "day2" / "predictive_maintenance_model.pkl"
RESULTS_FILE = BASE_DIR / "combined_live_results.csv"
FEATURES = ["temperature", "pressure", "vibration", "rpm", "current"]
INTERVAL_SECONDS = 1.0
MACHINE_COUNT = 5

RULES = [
    {
        "name": "SENSOR_FAILURE",
        "priority": 1,
        "condition": lambda reading: any(
            reading.get(sensor) is None for sensor in FEATURES
        ),
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_TEMP",
        "priority": 2,
        "condition": lambda reading: reading["temperature"] > 80,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 3,
        "condition": lambda reading: reading["pressure"] > 50,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 4,
        "condition": lambda reading: reading["vibration"] > 4.0,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 5,
        "condition": lambda reading: reading["current"] > 20,
        "action": "STOP_MACHINE",
    },
]


def train_model():
    """Train the Day 2 model and report its held-out test performance."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Training CSV was not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    required_columns = FEATURES + ["state"]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Training CSV is missing columns: {missing_columns}")
    if data[required_columns].isna().any().any():
        raise ValueError("Training CSV contains missing sensor values or states.")

    X = data[FEATURES]
    y = data["state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Day 2 predictive model evaluation (held-out test data)")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions, zero_division=0))

    # Retrain using all labeled rows for the live predictions.
    model.fit(X, y)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Saved model to: {MODEL_FILE}")
    return model, data


def generate_sensor_data(data):
    """Generate one simulated reading within the training data's sensor ranges."""
    reading = {}
    for feature in FEATURES:
        lower = float(data[feature].min())
        upper = float(data[feature].max())
        if feature == "rpm":
            reading[feature] = random.randint(int(lower), int(upper))
        else:
            reading[feature] = round(random.uniform(lower, upper), 2)
    return reading


def matching_rules(reading):
    """Return all matching Day 1 rules in priority order."""
    matched = []
    for rule in sorted(RULES, key=lambda item: item["priority"]):
        try:
            if rule["condition"](reading):
                matched.append(rule)
        except (KeyError, TypeError):
            continue
    return matched


def get_rule_decision(reading):
    matched = matching_rules(reading)
    if matched:
        return matched[0]["name"], matched[0]["action"], matched
    return "NO_RULE_FIRED", "CONTINUE_NORMAL", matched


def run_live_demo(interval_seconds=INTERVAL_SECONDS, max_cycles=None):
    """Print and log continuous simulated Day 1 and Day 2 machine results."""
    model, data = train_model()
    result_fields = [
        "timestamp",
        "machine",
        *FEATURES,
        "day1_rule",
        "day1_action",
        "day2_prediction",
    ]
    cycle = 0

    print(
        "Starting simulated live monitoring. "
        "Press Ctrl+C to stop; results are written to combined_live_results.csv."
    )
    try:
        with RESULTS_FILE.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=result_fields)
            writer.writeheader()

            while max_cycles is None or cycle < max_cycles:
                cycle += 1
                timestamp = datetime.now().isoformat(timespec="seconds")
                attention_candidates = []
                print(f"\n[{timestamp}] Monitoring cycle {cycle}")

                for machine_number in range(1, MACHINE_COUNT + 1):
                    machine = f"Machine {machine_number}"
                    reading = generate_sensor_data(data)
                    rule_name, action, matched = get_rule_decision(reading)
                    prediction = model.predict(
                        pd.DataFrame([reading], columns=FEATURES)
                    )[0]

                    print(
                        f"{machine}: {reading} | "
                        f"Day 1: {rule_name} -> {action} | "
                        f"Day 2 prediction: {prediction}"
                    )

                    if matched:
                        attention_candidates.append(
                            (
                                -len(matched),
                                matched[0]["priority"],
                                machine,
                                matched[0]["name"],
                            )
                        )

                    writer.writerow(
                        {
                            "timestamp": timestamp,
                            "machine": machine,
                            **reading,
                            "day1_rule": rule_name,
                            "day1_action": action,
                            "day2_prediction": prediction,
                        }
                    )

                output.flush()
                if attention_candidates:
                    _, _, machine, rule_name = min(attention_candidates)
                    print(
                        f"Day 1 priority: {machine} needs immediate attention "
                        f"({rule_name})."
                    )
                else:
                    print("Day 1 priority: No machine needs immediate attention.")

                if max_cycles is None or cycle < max_cycles:
                    time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nLive monitoring stopped.")


if __name__ == "__main__":
    run_live_demo()