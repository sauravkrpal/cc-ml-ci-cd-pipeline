import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load churn dataset
data = pd.read_csv("data/processed_data.csv")

# Drop customerID (not useful for prediction)
if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

# Target column
y = data["Churn"].map({"Yes": 1, "No": 0})  # convert Yes/No → 1/0
X = data.drop("Churn", axis=1)

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create directories
os.makedirs("model", exist_ok=True)
# Save feature columns for later
joblib.dump(X.columns, "model/trained_columns.pkl")
# Save model
joblib.dump(model, "model/churn_model.pkl")

print("✅ Model training complete. Saved as model/churn_model.pkl")
print(f"Training accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
