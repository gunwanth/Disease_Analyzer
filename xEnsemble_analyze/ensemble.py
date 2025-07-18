import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure output directory exists
output_dir = "xEnsemble_analyze/ensemble_outputs"
os.makedirs(output_dir, exist_ok=True)

# ✅ Load Dataset
data_path = "data/augmented_dataset_50000.csv"
df = pd.read_csv(data_path)

# ✅ Load LabelEncoder
encoder_path = "src/models/savedmodels/label_encoder.pkl"
label_encoder = joblib.load(encoder_path)
df['Disease'] = label_encoder.transform(df['Disease'])

# ✅ Split Dataset
X = df.drop('Disease', axis=1)
y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Load Ensemble Model
ensemble_model_path = "src/models/savedmodels/ensemble_voting.pkl"
if not os.path.exists(ensemble_model_path):
    raise FileNotFoundError(f"❌ Model file not found: {ensemble_model_path}")

ensemble_model = joblib.load(ensemble_model_path)

# ✅ If model is a list, extract the first model
if isinstance(ensemble_model, list):
    ensemble_model = ensemble_model[0]

# ✅ Make Predictions
y_pred = ensemble_model.predict(X_test)

# ✅ Ensure `y_pred` and `y_test` have the same format
if isinstance(y_test.iloc[0], str):  # If y_test contains strings, convert to numerical labels
    y_test = label_encoder.transform(y_test)

if isinstance(y_pred[0], str):  # If y_pred contains strings, convert to numerical labels
    y_pred = label_encoder.transform(y_pred)

# ✅ Compute Model Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

# ✅ Print Metrics
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

# ✅ Save Metrics as CSV
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [accuracy, precision, recall, f1]
})
metrics_df.to_csv(f"{output_dir}/ensemble_metrics.csv", index=False)

# ✅ Disease Distribution
disease_counts = df["Disease"].value_counts()
disease_counts.to_csv(f"{output_dir}/disease_distribution.csv")

plt.figure(figsize=(10, 5))
plt.bar(disease_counts.index[:15], disease_counts.values[:15], color='orange')
plt.xticks(rotation=90)
plt.xlabel("Diseases")
plt.ylabel("Number of Cases")
plt.title("Disease Distribution (Top 15)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"{output_dir}/disease_distribution.png")
plt.close()

# ✅ Symptom Weight Distribution
symptom_weights = df.drop(columns=["Disease"]).sum().sort_values(ascending=False)
symptom_weights.to_csv(f"{output_dir}/symptom_weight_distribution.csv")

plt.figure(figsize=(10, 5))
plt.bar(symptom_weights.index[:15], symptom_weights.values[:15], color='blue')
plt.xticks(rotation=90)
plt.xlabel("Symptoms")
plt.ylabel("Weights")
plt.title("Symptom Weight Distribution (Top 15)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"{output_dir}/symptom_weight_distribution.png")
plt.close()

# ✅ Cross-Validation Scores
try:
    cross_val_scores = cross_val_score(ensemble_model, X, y, cv=5)
    cv_df = pd.DataFrame({"Fold": list(range(1, 6)), "Accuracy": cross_val_scores})
    cv_df.to_csv(f"{output_dir}/ensemble_cross_val_scores.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, 6), cross_val_scores, color='green')
    plt.xticks(range(1, 6))
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Cross-Validation Scores")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/ensemble_cross_val_scores.png")
    plt.close()
except Exception as e:
    print(f"❌ Cross-validation error: {e}")

# ✅ Save Predictions
predictions_df = pd.DataFrame({
    "Actual": label_encoder.inverse_transform(y_test),
    "Predicted": label_encoder.inverse_transform(y_pred)
})
predictions_df.to_csv(f"{output_dir}/ensemble_predictions.csv", index=False)

print(f"✅ Ensemble Model Analysis Completed! Check the '{output_dir}' folder.")