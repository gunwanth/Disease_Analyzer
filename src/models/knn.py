import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])

scaler = StandardScaler()
X = scaler.fit_transform(df.drop('Disease', axis=1))
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 5  
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

model_path = "src/models/savedmodels/knn_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved at {model_path}")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(report)

# Function to predict disease from user input
def predict_disease():
    user_symptoms = input("Enter your symptoms separated by commas: ").split(',')
    user_symptoms = [symptom.strip() for symptom in user_symptoms]

    valid_symptoms = [s for s in user_symptoms if s in all_symptoms]
    invalid_symptoms = list(set(user_symptoms) - set(valid_symptoms))

    if invalid_symptoms:
        print(f"⚠ Unknown symptoms detected: {invalid_symptoms}")

    input_vector = np.zeros(len(all_symptoms))
    for i, symptom in enumerate(all_symptoms):
        if symptom in valid_symptoms:
            input_vector[i] = symptom_weights[symptom]

    input_vector = scaler.transform([input_vector])
    probabilities = model.predict_proba(input_vector)[0]

    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_diseases = label_encoder.inverse_transform(top_3_indices)

    print("\nTop 3 Predicted Diseases:")
    for i, disease in enumerate(top_3_diseases):
        confidence = probabilities[top_3_indices[i]] * 100
        print(f"{i+1}. {disease} - {confidence:.2f}% confidence")

print("\nTesting predictions on random samples:")
sample_indices = np.random.choice(X_test.shape[0], 5, replace=False)
for idx in sample_indices:
    sample = X_test[idx].reshape(1, -1)
    prediction = model.predict(sample)
    print(f"Actual: {label_encoder.inverse_transform([y_test.iloc[idx]])[0]}, Predicted: {label_encoder.inverse_transform(prediction)[0]}")

predict_disease()