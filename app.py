from flask import Flask, render_template, request, jsonify , url_for , redirect , session , flash
import joblib
import numpy as np
import pandas as pd
import os
import re
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

app = Flask(__name__)

# Ensure 'static/graphs' directory exists
graphs_dir = "static/graphs"
os.makedirs(graphs_dir, exist_ok=True)

# Load dataset
data_path = "data/augmented_dataset_50000.csv"
df = pd.read_csv(data_path)

# Load LabelEncoder
encoder_path = "src/models/savedmodels/label_encoder.pkl"
label_encoder = joblib.load(encoder_path)
df['Disease'] = label_encoder.transform(df['Disease'])

# Extract all symptoms from dataset columns
all_symptoms = list(df.columns)
all_symptoms.remove("Disease")  # Remove the target column

# Split dataset
X = df.drop('Disease', axis=1)
y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models
model_paths = {
    "Decision Tree": "src/models/savedmodels/decision_tree.pkl",
    "Random Forest": "src/models/savedmodels/random_forest.pkl",
    "KNN": "src/models/savedmodels/knn_model.pkl",
    "Naive Bayes": "src/models/savedmodels/naive_bayes_optimized.pkl",
    "SVM": "src/models/savedmodels/svm_model.pkl",
    "Ensemble": "src/models/savedmodels/ensemble_voting.pkl"
}

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            if isinstance(model, list):  # Fix for ensemble model issue
                model = model[0]  # Extract model if stored as a list
            if isinstance(model, BaseEstimator):  # Ensure it's a valid scikit-learn model
                models[name] = model
                print(f"✅ Loaded model: {name}")
            else:
                print(f"⚠️ Skipping {name}: Not a valid scikit-learn model.")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")

# Function to ensure input size matches the model
def get_input_vector(user_symptoms, model):
    feature_size = model.n_features_in_
    input_vector = np.zeros(feature_size)

    for symptom in user_symptoms:
        if symptom in X.columns:
            index = X.columns.get_loc(symptom)
            if index < feature_size:
                input_vector[index] = 1

    return input_vector


@app.route("/")
def features():
    return render_template("features.html")

# Login Page
@app.route("/login")
def front():
    return render_template("login.html")


app.secret_key = os.urandom(24)  # Secret key for session management

# Initialize SQLite Database
def init_db():
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()

init_db()  # Create database if not exists

# Home Page (Redirects to Login)
@app.route("/")
def welcome():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")  # Get email from form
        if email:  # If email is entered, proceed
            session["user"] = email  # Store email in session
            return redirect(url_for("page"))  # Redirect to homepage
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        return redirect(url_for("login"))  # Redirect to login after sign-up
    return render_template("signup.html")

@app.route("/page")
def page():
    if "user" in session:  # Check if user is logged in
        return render_template("hompage.html", user=session["user"])
    return redirect(url_for("login"))  # Redirect if not logged in

@app.route("/logout")
def logout():
    session.pop("user", None)  # Clear session
    return redirect(url_for("login"))


@app.route('/')
def homepage():
    return render_template('hompage.html')  # Ensure correct file name

@app.route('/index')
def predict_page():
    return render_template('index.html', models=models.keys(), symptoms=all_symptoms)

@app.route('/')
def home():
    return render_template('index.html', models=models.keys(), symptoms=all_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    if request.content_type != 'application/json':
        return jsonify({"error": "Invalid content type. Expecting application/json"}), 415

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    selected_model_name = data.get("model")
    user_symptoms = data.get("symptoms", [])

    if not selected_model_name or selected_model_name not in models:
        return jsonify({"error": "Model not found"}), 400

    selected_model = models[selected_model_name]
    input_vector = get_input_vector(user_symptoms, selected_model)

    try:
        prediction = selected_model.predict([input_vector])

        # 🚀 Debug: Print raw predictions
        print(f"🚀 Raw Model Prediction ({selected_model_name}): {prediction}")

        # 🛠 Handle different model outputs
        if isinstance(prediction, (list, np.ndarray)):
            prediction = prediction[0]  # Extract single value if it's an array

        if isinstance(prediction, np.generic):
            prediction = prediction.item()  # Convert NumPy types to Python native

        # Special handling for **Ensemble & SVM models**
        if selected_model_name in ["Ensemble", "SVM"]:
            try:
                prediction = int(prediction)  # Ensure integer conversion
                predicted_disease = label_encoder.inverse_transform([prediction])[0]
            except ValueError:
                predicted_disease = f"Unknown Disease (Label {prediction})"  
        else:
            # Default case for other models
            predicted_disease = label_encoder.inverse_transform([prediction])[0]

        print(f"✅ Predicted Disease: {predicted_disease}")

        return jsonify({
            "disease": predicted_disease,  
            "model": selected_model_name,
            "symptoms": user_symptoms
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Model Evaluation Route
@app.route('/metrics')
def show_metrics():
    model_metrics = {}

    # Evaluate each model
    for model_name, model in models.items():
        try:
            y_pred = model.predict(X_test)

            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

            # Store results
            model_metrics[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

            # Generate and save graph
            graph_path = f"{graphs_dir}/{model_name.lower().replace(' ', '_')}.png"
            plt.figure(figsize=(8, 5))
            plt.bar(["Accuracy", "Precision", "Recall", "F1 Score"],
                    [accuracy, precision, recall, f1],
                    color=['blue', 'orange', 'green', 'red'])
            plt.ylim(0, 1)
            plt.xlabel("Metrics")
            plt.ylabel("Score")
            plt.title(f"Model Performance: {model_name}")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(graph_path)
            plt.close()
            print(f"📊 Graph saved at: {graph_path}")
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")

    return render_template('metrics.html', model_metrics=model_metrics, graphs_dir=graphs_dir)

# dataset analysis

@app.route('/dataset_analysis')
def dataset_analysis():
    data_path = "data/augmented_dataset_50000.csv"
    df = pd.read_csv(data_path)

    diseases = sorted(df["Disease"].unique())
    symptoms = sorted([col for col in df.columns if col != "Disease"])

    disease_counts = df["Disease"].value_counts()
    disease_labels = disease_counts.index.tolist()
    disease_values = disease_counts.values.tolist()

    symptom_occurrences = (df.drop(columns=["Disease"]) > 0).sum().sort_values(ascending=False)
    symptom_labels = symptom_occurrences.index[:15].tolist()
    symptom_values = symptom_occurrences.values[:15].tolist()

    symptom_weights = df.drop(columns=["Disease"]).sum().sort_values(ascending=False)
    weight_labels = symptom_weights.index[:15].tolist()
    weight_values = symptom_weights.values[:15].tolist()

    return render_template('dataset_analysis.html',
                           num_diseases=len(diseases),
                           num_symptoms=len(symptoms),
                           disease_labels=disease_labels,
                           disease_values=disease_values,
                           symptom_labels=symptom_labels,
                           symptom_values=symptom_values,
                           weight_labels=weight_labels,
                           weight_values=weight_values,
                           disease_image=url_for('static', filename='graphs/disease_distribution.png'),
                           symptom_image=url_for('static', filename='graphs/symptom_distribution.png'),
                           weight_image=url_for('static', filename='graphs/weight_distribution.png'))


ensemble_output_dir = "xEnsemble_analyze/ensemble_outputs"

@app.route('/ensemble_analysis')
def ensemble_analysis():
    """Load the ensemble model analysis results dynamically"""

    # ✅ Read Model Performance Metrics
    metrics_path = os.path.join(ensemble_output_dir, "ensemble_metrics.csv")
    if os.path.exists(metrics_path):
        model_metrics = pd.read_csv(metrics_path).set_index("Metric").to_dict()["Value"]
    else:
        model_metrics = {}

    # ✅ Read Disease Distribution
    disease_path = os.path.join(ensemble_output_dir, "disease_distribution.csv")
    if os.path.exists(disease_path):
        df_disease = pd.read_csv(disease_path)
        if "Disease" in df_disease.columns and "Count" in df_disease.columns:
            disease_counts = df_disease.set_index("Disease")["Count"].to_dict()
        else:
            print("⚠ Warning: Missing columns in disease distribution CSV.")
            disease_counts = {}
    else:
        disease_counts = {}

    # ✅ Read Symptom Weight Distribution
    symptom_path = os.path.join(ensemble_output_dir, "symptom_weight_distribution.csv")
    if os.path.exists(symptom_path):
        df_symptom = pd.read_csv(symptom_path)
        if "Symptom" in df_symptom.columns and "Weight" in df_symptom.columns:
            symptom_weights = df_symptom.set_index("Symptom")["Weight"].to_dict()
        else:
            print("⚠ Warning: Missing columns in symptom weight CSV.")
            symptom_weights = {}
    else:
        symptom_weights = {}

    # ✅ Read Cross Validation Scores
    cv_path = os.path.join(ensemble_output_dir, "ensemble_cross_val_scores.csv")
    if os.path.exists(cv_path):
        cross_val_scores = pd.read_csv(cv_path).set_index("Fold").to_dict()["Accuracy"]
    else:
        cross_val_scores = {}

    # ✅ Read Predictions
    predictions_path = os.path.join(ensemble_output_dir, "ensemble_predictions.csv")
    if os.path.exists(predictions_path):
        df_predictions = pd.read_csv(predictions_path)
        predictions = df_predictions.to_dict(orient="records")
    else:
        predictions = []

    return render_template("ensemble_analysis.html",
                           model_metrics=model_metrics,
                           disease_counts=disease_counts,
                           symptom_weights=symptom_weights,
                           cross_val_scores=cross_val_scores,
                           predictions=predictions)

if __name__ == '__main__':
    print("🚀 Flask App Running on http://127.0.0.1:5000/")
    app.run(debug=True)
