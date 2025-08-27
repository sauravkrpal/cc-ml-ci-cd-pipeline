import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Load dataset
data = pd.read_csv("data/processed_data.csv")

# Drop customerID
if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

# Target column
y = data["Churn"].map({"Yes": 1, "No": 0})
X = data.drop("Churn", axis=1)

# One-hot encode categorical variables (same as train.py)
X = pd.get_dummies(X)

# Load trained model
model = joblib.load("model/churn_model.pkl")

# Align features between train and evaluate
# (in case columns differ due to dummy encoding)
model_features = model.n_features_in_
if X.shape[1] != model_features:
    # Ensure same columns as training
    # Load the feature list from training
    trained_columns = joblib.load("model/trained_columns.pkl")
    X = X.reindex(columns=trained_columns, fill_value=0)

# Predict
y_pred = model.predict(X)

# Evaluate
print("✅ Evaluation Results:")
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))
print("\nClassification Report:")
print(classification_report(y, y_pred))
