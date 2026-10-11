import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Define your file paths
CSV_PATH = (
    r"C:\Users\pl\system_automation-23r239-1\day2\day2_sample.csv"
)
MODEL_PATH = (
    r"C:\Users\pl\system_automation-23r239-1\day2\predictive_maintenance_model.pkl"
)

# 2. Load the data and the trained model
df = pd.read_csv(r"C:\Users\pl\system_automation-23r239-1\day2\day2_sample.csv")
model = joblib.load(r"C:\Users\pl\system_automation-23r239-1\day2\predictive_maintenance_model.pkl"
)

# Clean up column names just in case there are hidden spaces
df.columns = df.columns.str.strip()

# 3. Extract the features and target variable
# Note: Update "temperature" spelling here if your CSV used a different spelling
X = df[["temperature", "pressure", "vibration", "rpm", "current"]]
y = df["state"]

# 4. Split the data exactly like we did during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Generate predictions using the testing dataset
y_pred = model.predict(X_test)

# 6. Calculate and print the accuracy metrics
accuracy = accuracy_score(y_test, y_pred)

print("========================================")
print("       MODEL EVALUATION REPORT          ")
print("========================================")
print(f"Overall Accuracy Score: {accuracy * 100:.2f}%\n")
print("Detailed Performance Breakdown:")
print(classification_report(y_test, y_pred))
print("========================================")
