import pandas as pd
import json
import os

# Define file paths
DATA_DIR = r"D:/disease analyzer project/data"
dataset_path = os.path.join(DATA_DIR, "dataset.csv")
json_output_path = os.path.join(DATA_DIR, "disease_symptoms.json")
preprocessed_dataset_path = os.path.join(DATA_DIR, "preprocessed_dataset.csv")

try:
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully with shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The dataset file was not found at {dataset_path}")
    exit()

df.columns = df.columns.str.strip()

df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

df.fillna('unknown', inplace=True)
print(f"Dataset shape after handling missing values: {df.shape}")

df_encoded = pd.get_dummies(df, columns=df.columns[1:]) 

df_encoded.to_csv(preprocessed_dataset_path, index=False)
print(f"Preprocessed dataset saved to {preprocessed_dataset_path}")

disease_symptom_mapping = {}
for _, row in df.iterrows():
    disease = row['Disease']
    symptoms = row.index[row == 1].tolist()
    disease_symptom_mapping[disease] = symptoms

with open(json_output_path, 'w') as json_file:
    json.dump(disease_symptom_mapping, json_file, indent=4)
print(f"Disease-symptom mapping saved to {json_output_path}")