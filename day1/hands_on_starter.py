"""
DAY 1 · HANDS-ON — Build Your Rule Engine
23IF02 System Automation

Work through the TODOs in order. Each one matches a step on the
"Hands-on · Build Your Rule Engine" slide. Run this file after each
step to see your progress:

    python hands_on_starter.py

Need the answer? See hands_on_solution.py — but try it yourself first.
"""
import random
import time

LOG_FILE = "day1_log.txt"


# ---------------------------------------------------------------------------
# STEP 1 — Generate simulated sensor data
# ---------------------------------------------------------------------------
def generate_sensor_data():
    """Return realistic simulated machine sensor readings."""
    return {
        "temperature": round(random.uniform(50, 95), 2),
        "pressure": round(random.uniform(20, 60), 2),
        "vibration": round(random.uniform(0.1, 5.0), 2),
        "current": round(random.uniform(5, 25), 2),
    }


# ---------------------------------------------------------------------------
# STEP 2, STEP 7, STEP 8 — Automation rules
# Lower priority number means the rule is checked first.
# ---------------------------------------------------------------------------
RULES = [
    {
        "name": "SENSOR_FAILURE",
        "priority": 1,
        "condition": lambda r: any(
            r.get(sensor) is None
            for sensor in ("temperature", "pressure", "vibration", "current")
        ),
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_TEMP",
        "priority": 2,
        "condition": lambda r: r["temperature"] > 80,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 3,
        "condition": lambda r: r["pressure"] > 50,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 4,
        "condition": lambda r: r["vibration"] > 4.0,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 5,
        "condition": lambda r: r["current"] > 20,
        "action": "STOP_MACHINE",
    },
]


# ---------------------------------------------------------------------------
# STEP 3 — Detect machine state from the rules
# ---------------------------------------------------------------------------
def matching_rules(reading, rules):
    """Return all rules that match, ordered by priority."""
    sorted_rules = sorted(rules, key=lambda rule: rule["priority"])
    matched = []

    for rule in sorted_rules:
        try:
            if rule["condition"](reading):
                matched.append(rule)
        except (TypeError, KeyError):
            # Ignore rules that cannot be evaluated due to missing/bad data.
            continue

    return matched


def evaluate_rules(reading, rules):
    """Return the first matching rule in priority order."""
    matched = matching_rules(reading, rules)
    if matched:
        return matched[0]["name"], matched[0]["action"]

    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


# ---------------------------------------------------------------------------
# STEP 4 — Trigger an action when a rule fires
# ---------------------------------------------------------------------------
def take_action(action):
    """Return a human-readable message for the selected action."""
    actions = {
        "STOP_MACHINE": "WARNING! Machine stopped due to an unsafe condition.",
        "REDUCE_PRESSURE": "High pressure detected. Reduce system pressure.",
        "CHECK_VIBRATION": "High vibration detected. Inspect machine components.",
        "REDUCE_LOAD": "High current detected. Reduce the machine load.",
        "CHECK_SENSOR": "Sensor failure detected. Inspect the sensors.",
        "CONTINUE_NORMAL": "All readings normal. Machine continues operating.",
    }

    return actions.get(action, f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# STEP 5 — Log every decision made
# ---------------------------------------------------------------------------
def log_decision(reading, rule_name, action):
    """Append each sensor reading and decision to the log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"{timestamp} | "
            f"Reading: {reading} | "
            f"Rule: {rule_name} | "
            f"Action: {action}\n"
        )


# ---------------------------------------------------------------------------
# STEP 6 — Add a second sensor: current
# Current is included in generate_sensor_data() and HIGH_CURRENT rule.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PROBLEM 2 — Identify which of five machines needs immediate attention
# ---------------------------------------------------------------------------
def identify_machine_needing_attention(machine_readings):
    """Return the machine matching the most rules, if any."""
    candidates = []

    for machine_name, reading in machine_readings.items():
        matched = matching_rules(reading, RULES)
        if matched:
            rule_name = matched[0]["name"]
            action = matched[0]["action"]
            candidates.append(
                (-len(matched), matched[0]["priority"], machine_name,
                 rule_name, action, len(matched))
            )

    if not candidates:
        return None

    _, _, machine_name, rule_name, action, condition_count = min(
        candidates, key=lambda candidate: (candidate[0], candidate[1])
    )
    return machine_name, rule_name, action, condition_count


def check_factory_machines():
    """Generate readings for five machines and report the most urgent one."""
    machines = {
        f"Machine {number}": generate_sensor_data()
        for number in range(1, 6)
    }

    print("Factory machine check:")
    for machine_name, reading in machines.items():
        matched = matching_rules(reading, RULES)
        rule_names = ", ".join(rule["name"] for rule in matched)
        print(f"  {machine_name}: {reading} -> {rule_names or 'NO_RULE_FIRED'}")

    attention = identify_machine_needing_attention(machines)
    if attention is None:
        print("No machine needs immediate attention.\n")
    else:
        machine_name, rule_name, action, condition_count = attention
        print(
            f"Immediate attention: {machine_name} "
            f"({condition_count} conditions triggered; {rule_name}: "
            f"{take_action(action)})\n"
        )


# ---------------------------------------------------------------------------
# MAIN PROGRAM — Run the automation system
# ---------------------------------------------------------------------------
def main():
    for tick in range(1, 11):
        reading = generate_sensor_data()

        rule_name, action = evaluate_rules(reading, RULES)
        outcome = take_action(action)

        print(f"[Tick {tick:02d}] SENSE -> {reading}")
        print(f"           THINK -> {rule_name}")
        print(f"           ACT   -> {outcome}\n")

        log_decision(reading, rule_name, action)

        time.sleep(0.2)

    check_factory_machines()


if __name__ == "__main__":
    main()