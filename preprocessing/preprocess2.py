import pandas as pd
import numpy as np

# Load the original dataset
file_path = 'data/preprocessed_dataset.csv'  
output_file_path = 'data/augmented_dataset_50000.csv'

df = pd.read_csv(file_path)

symptom_weights = {}
for disease in df['Disease'].unique():
    disease_df = df[df['Disease'] == disease]
    symptom_counts = disease_df.iloc[:, 1:].sum(axis=0)
    total_symptoms = symptom_counts.sum()
    weights = symptom_counts / total_symptoms if total_symptoms > 0 else symptom_counts
    symptom_weights[disease] = weights

# Generate synthetic samples
samples_to_generate = 50000 - len(df)
new_data = []
np.random.seed(42)

for i in range(samples_to_generate):
    disease = np.random.choice(df['Disease'].unique())
    weights = symptom_weights[disease]

    synthetic_sample = np.random.rand(1, len(df.columns) - 1) < weights.values
    new_row = np.hstack([disease, synthetic_sample[0]])
    new_data.append(new_row)

new_df = pd.DataFrame(new_data, columns=df.columns)
augmented_df = pd.concat([df, new_df], ignore_index=True)
augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)

augmented_df.to_csv(output_file_path, index=False)

print(f"Augmented dataset with 50,000 samples saved to {output_file_path}")
