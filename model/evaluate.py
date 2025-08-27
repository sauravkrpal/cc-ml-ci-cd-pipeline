import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the processed churn dataset (make sure this matches your train script preprocessing)
data = pd.read_csv("data/processed_data.csv")

# Separate features and target
X = data.drop("Churn", axis=1)   # assuming your target column is "Churn"
y = data["Churn"]

# Load the trained model
model = joblib.load("model/customer_segmentation.pkl")

# Make predictions
y_pred = model.predict(X)

# Evaluate model
print("📊 Model Evaluation on Entire Dataset:")
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall: {recall_score(y, y_pred):.4f}")
print(f"F1-score: {f1_score(y, y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nDetailed Classification Report:")
print(classification_report(y, y_pred))
