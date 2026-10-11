import csv
import random
from pathlib import Path


NUM_RECORDS = 200
OUTPUT_FILE = Path(__file__).with_name("history_data.csv")

# Ensure reproducible random data
random.seed(42)


def get_state(temperature, pressure, vibration, rpm, current):
    if (
        temperature > 100
        or pressure > 10.0
        or vibration > 15.0
        or rpm > 5000
        or current > 15.0
    ):
        return "failure"

    if (
        temperature > 90
        or pressure > 5.0
        or vibration > 10.0
        or rpm > 2000
        or current > 2.0
    ):
        return "warning"

    return "normal"


def generate_data(num_records=NUM_RECORDS):
    records = []

    for _ in range(num_records):
        temperature = round(random.uniform(0, 120), 2)
        pressure = round(random.uniform(0, 12), 2)
        vibration = round(random.uniform(0.1, 20), 2)
        rpm = random.randint(1, 6000)
        current = round(random.uniform(0, 20), 2)

        records.append(
            {
                "temperature": temperature,
                "pressure": pressure,
                "vibration": vibration,
                "rpm": rpm,
                "current": current,
                "state": get_state(
                    temperature, pressure, vibration, rpm, current
                ),
            }
        )

    return records


def main():
    records = generate_data()

    fieldnames = [
        "temperature",
        "pressure",
        "vibration",
        "rpm",
        "current",
        "state",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} records in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()