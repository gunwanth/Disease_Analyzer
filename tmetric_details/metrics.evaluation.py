import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ensure 'static/graphs' directory exists
graphs_dir = "app/static/graphs"
os.makedirs(graphs_dir, exist_ok=True)

# Load dataset
data_path = "data/augmented_dataset_50000.csv"  
df = pd.read_csv(data_path)

# Load LabelEncoder
encoder_path = "src/models/savedmodels/label_encoder.pkl"
label_encoder = joblib.load(encoder_path)
df['Disease'] = label_encoder.transform(df['Disease'])

# Split dataset
X = df.drop('Disease', axis=1)
y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model paths
model_paths = {
    "Decision Tree": "src/models/savedmodels/decision_tree.pkl",
    "Random Forest": "src/models/savedmodels/random_forest.pkl",
    "Naive Bayes": "src/models/savedmodels/naive_bayes.pkl",
    "SVM": "src/models/savedmodels/svm.pkl",
    "KNN": "src/models/savedmodels/knn.pkl"
}

# Function to evaluate and save a model's metrics graph
def evaluate_and_save_model(model, model_name):
    y_pred = model.predict(X_test)
    
    # Convert predictions back to label encoding
    y_pred = label_encoder.transform(label_encoder.inverse_transform(y_pred))

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

    print(f"\n🔹 Model: {model_name}")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"📈 Recall: {recall:.4f}")
    print(f"🏆 F1-score: {f1:.4f}")

    # Plot Metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = [accuracy, precision, recall, f1]

    plt.figure(figsize=(8, 5))
    plt.bar(metrics, scores, color=['blue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title(f"Model Performance: {model_name}")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the graph instead of displaying
    graph_filename = f"{model_name.lower().replace(' ', '_')}.png"
    graph_path = os.path.join(graphs_dir, graph_filename)
    plt.savefig(graph_path)
    plt.close()

    print(f"📊 Graph saved at: {graph_path}")

# Load and evaluate each model
for model_name, model_path in model_paths.items():
    try:
        model = joblib.load(model_path)
        evaluate_and_save_model(model, model_name)
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
