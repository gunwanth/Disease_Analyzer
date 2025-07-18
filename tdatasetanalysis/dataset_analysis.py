import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = "data/augmented_dataset_50000.csv"
df = pd.read_csv(data_path)

# Ensure the directory exists
analysis_dir = "app/static/dataset_analysis"
os.makedirs(analysis_dir, exist_ok=True)

# Generate summary
summary_file = os.path.join(analysis_dir, "dataset_summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("📊 Dataset Summary:\n")
    f.write(df.describe(include="all").to_string())

# Count unique symptoms and diseases
symptom_counts = df.drop(columns=["Disease"]).sum().sort_values(ascending=False)
disease_counts = df["Disease"].value_counts()

# Save symptoms table
symptoms_table = pd.DataFrame(symptom_counts, columns=["Weight"])
symptoms_table.to_csv(os.path.join(analysis_dir, "symptoms_table.csv"))

# Save diseases table
diseases_table = pd.DataFrame(disease_counts, columns=["Count"])
diseases_table.to_csv(os.path.join(analysis_dir, "diseases_table.csv"))

# Plot symptom weight distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=symptom_counts.index[:20], y=symptom_counts.values[:20], palette="coolwarm")
plt.xticks(rotation=90)
plt.xlabel("Symptoms")
plt.ylabel("Weight")
plt.title("Top 20 Symptoms by Weight")
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "symptoms_distribution.png"))
plt.close()

# Plot disease frequency distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=disease_counts.index[:20], y=disease_counts.values[:20], palette="coolwarm")
plt.xticks(rotation=90)
plt.xlabel("Diseases")
plt.ylabel("Count")
plt.title("Top 20 Diseases by Frequency")
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "disease_distribution.png"))
plt.close()

print("✅ Dataset analysis completed! Check static/dataset_analysis/")
