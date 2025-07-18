import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report

decision_tree = joblib.load('src/models/savedmodels/decision_tree.pkl')
knn_model = joblib.load('src/models/savedmodels/knn_model.pkl')
naive_bayes = joblib.load('src/models/savedmodels/naive_bayes_optimized.pkl')
random_forest = joblib.load('src/models/savedmodels/random_forest.pkl')
svm_model = joblib.load('src/models/savedmodels/svm_model.pkl')

models = [decision_tree, knn_model, naive_bayes, random_forest, svm_model]

file_path = 'data/augmented_dataset_50000.csv'
df = pd.read_csv(file_path)
X = df.drop('Disease', axis=1)
y = df['Disease']

# Use a smaller test size for faster evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Function for majority voting
def majority_vote(predictions):
    return Counter(predictions).most_common(1)[0][0]

predictions = np.array([model.predict(X_test) for model in models])
final_predictions = [majority_vote(predictions[:, i]) for i in range(len(X_test))]

accuracy = accuracy_score(y_test, final_predictions)
report = classification_report(y_test, final_predictions)

print(f'Ensemble Model Accuracy: {accuracy * 100:.2f}%')
print(report)

ensemble_model_path = "src/models/savedmodels/ensemble_voting.pkl"
joblib.dump(models, ensemble_model_path)
print(f"Ensemble model saved at {ensemble_model_path}")

def predict_disease():
    user_symptoms = input("Enter your symptoms separated by commas: ").split(',')
    user_symptoms = [symptom.strip() for symptom in user_symptoms]

    all_symptoms = X.columns.tolist()
    valid_symptoms = [s for s in user_symptoms if s in all_symptoms]
    invalid_symptoms = list(set(user_symptoms) - set(valid_symptoms))

    if invalid_symptoms:
        print(f"⚠️ Unknown symptoms detected: {invalid_symptoms}")

    input_vector = np.zeros(len(all_symptoms))
    for i, symptom in enumerate(all_symptoms):
        if symptom in valid_symptoms:
            input_vector[i] = 1  

    model_predictions = [model.predict([input_vector])[0] for model in models]
    final_prediction = majority_vote(model_predictions)

    print(f"Predicted Disease: {final_prediction}")

print("\nTesting predictions on random samples:")
sample_indices = np.random.choice(X_test.index, 5, replace=False)
for idx in sample_indices:
    sample = X_test.loc[idx].values.reshape(1, -1)
    model_preds = [model.predict(sample)[0] for model in models]
    final_pred = majority_vote(model_preds)
    print(f"Actual: {y_test.loc[idx]}, Predicted: {final_pred}")

predict_disease()