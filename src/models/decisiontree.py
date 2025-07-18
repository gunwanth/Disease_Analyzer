import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import joblib

file_path = 'data/augmented_dataset_50000.csv'
df = pd.read_csv(file_path)

with open('data/corrected_disease_symptoms.json') as f:
    disease_symptoms = json.load(f)

symptom_weights = {}
all_symptoms = df.columns[1:]

for symptom in all_symptoms:
    count = sum([1 for disease, symptoms in disease_symptoms.items() if symptom in symptoms])
    symptom_weights[symptom] = 1 / count if count > 0 else 0

for symptom in all_symptoms:
    df[symptom] = df[symptom] * symptom_weights[symptom]

X = df.drop('Disease', axis=1)
y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

model_path = "src/models/savedmodels/decision_tree.pkl"
joblib.dump(model, model_path)
print(f"Model saved at {model_path}")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(report)

def predict_disease():
    user_symptoms = input("Enter your symptoms separated by commas: ").split(',')
    user_symptoms = [symptom.strip() for symptom in user_symptoms]

    input_vector = np.zeros(len(all_symptoms))
    for i, symptom in enumerate(all_symptoms):
        if symptom in user_symptoms:
            input_vector[i] = symptom_weights[symptom]

    prediction = model.predict([input_vector])
    print(f"Predicted Disease: {prediction[0]}")

print("\nTesting predictions on random samples:")
sample_indices = np.random.choice(X_test.index, 5, replace=False)
for idx in sample_indices:
    sample = X_test.loc[idx].values.reshape(1, -1)
    prediction = model.predict(sample)
    print(f"Actual: {y_test.loc[idx]}, Predicted: {prediction[0]}")

predict_disease()