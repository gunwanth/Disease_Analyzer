import numpy as np
import joblib
from collections import Counter
import pandas as pd

ensemble_model_path = "src/models/savedmodels/ensemble_voting.pkl"
models = joblib.load(ensemble_model_path)

file_path = 'data/augmented_dataset_50000.csv'
df = pd.read_csv(file_path, nrows=1)  
all_symptoms = df.drop('Disease', axis=1).columns.tolist()

# Function for majority voting
def majority_vote(predictions):
    return Counter(predictions).most_common(1)[0][0]

def predict_disease(user_symptoms, selected_models=None):
    user_symptoms = [symptom.strip() for symptom in user_symptoms]

    valid_symptoms = [s for s in user_symptoms if s in all_symptoms]
    invalid_symptoms = list(set(user_symptoms) - set(valid_symptoms))

    if invalid_symptoms:
        print(f"⚠️ Unknown symptoms detected: {invalid_symptoms}")

    input_vector = np.zeros(len(all_symptoms))
    for i, symptom in enumerate(all_symptoms):
        if symptom in valid_symptoms:
            input_vector[i] = 1 

    selected_models = selected_models or models 
    model_predictions = [model.predict([input_vector])[0] for model in selected_models]
    final_prediction = majority_vote(model_predictions)

    return final_prediction

if __name__ == "__main__":
    user_input = input("Enter your symptoms separated by commas: ").split(',')
    predicted_disease = predict_disease(user_input, selected_models=[models[0], models[3]])  
    print(f"Predicted Disease: {predicted_disease}")