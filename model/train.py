import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load churn dataset
data = pd.read_csv("data/processed_data.csv")

# Features & target (assuming "Churn" is the target column)
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create directories
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/churn_model.pkl")

print("✅ Model training complete. Saved as model/churn_model.pkl")
