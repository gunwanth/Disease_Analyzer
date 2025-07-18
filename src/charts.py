import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.inspection import permutation_importance  # For models without built-in feature importance

# Load models
rf_model = joblib.load("src/models/savedmodels/random_forest.pkl")
svm_model = joblib.load("src/models/savedmodels/svm_model.pkl")
nv_model = joblib.load("src/models/savedmodels/naive_bayes_optimized.pkl")
knn_model = joblib.load("src/models/savedmodels/knn_model.pkl")
dt_model = joblib.load("src/models/savedmodels/decision_tree.pkl")
en_model = joblib.load("src/models/savedmodels/ensemble_voting.pkl")

# Load dataset features
file_path = 'data/augmented_dataset_50000.csv'
df = pd.read_csv(file_path)
features = df.columns[1:]
X = df.iloc[:, 1:]  # Features
y = df.iloc[:, 0]    # Target

# ------------------------------
# 1️⃣ Random Forest Feature Importance
# ------------------------------
if hasattr(rf_model, "feature_importances_"):
    rf_importance = rf_model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rf_importance, y=features)
    plt.xlabel("Feature Importance")
    plt.ylabel("Symptoms")
    plt.title("Feature Importance - Random Forest")
    plt.show()
else:
    print("Random Forest does not provide feature importance.")

# ------------------------------
# 2️⃣ Decision Tree Feature Importance
# ------------------------------
if hasattr(dt_model, "feature_importances_"):
    dt_importance = dt_model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=dt_importance, y=features)
    plt.xlabel("Feature Importance")
    plt.ylabel("Symptoms")
    plt.title("Feature Importance - Decision Tree")
    plt.show()
else:
    print("Decision Tree does not provide feature importance.")

# ------------------------------
# 3️⃣ SVM Weights (Only for Linear Kernel)
# ------------------------------
if hasattr(svm_model, "coef_"):
    svm_weights = svm_model.coef_[0]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=svm_weights, y=features)
    plt.xlabel("Weight")
    plt.ylabel("Symptoms")
    plt.title("SVM Weights (Linear Kernel)")
    plt.show()
else:
    print("SVM model does not have explicit feature weights (likely using RBF kernel).")

# ------------------------------
# 4️⃣ Naive Bayes (Log Probabilities as Importance)
# ------------------------------
if hasattr(nv_model, "feature_log_prob_"):
    nb_importance = np.mean(nv_model.feature_log_prob_, axis=0)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=nb_importance, y=features)
    plt.xlabel("Importance (Log Probability)")
    plt.ylabel("Symptoms")
    plt.title("Feature Importance - Naive Bayes (Log Probabilities)")
    plt.show()
else:
    print("Naive Bayes does not provide feature importances.")

# ------------------------------
# 5️⃣ KNN Feature Importance (Using Permutation)
# ------------------------------
knn_importance = permutation_importance(knn_model, X, y, scoring="accuracy", n_repeats=10, random_state=42).importances_mean
plt.figure(figsize=(10, 6))
sns.barplot(x=knn_importance, y=features)
plt.xlabel("Permutation Importance")
plt.ylabel("Symptoms")
plt.title("Feature Importance - KNN (Permutation Importance)")
plt.show()

# ------------------------------
# 6️⃣ Ensemble Model Feature Importance (Using Permutation)
# ------------------------------
en_importance = permutation_importance(en_model, X, y, scoring="accuracy", n_repeats=10, random_state=42).importances_mean
plt.figure(figsize=(10, 6))
sns.barplot(x=en_importance, y=features)
plt.xlabel("Permutation Importance")
plt.ylabel("Symptoms")
plt.title("Feature Importance - Ensemble Model (Permutation Importance)")
plt.show()