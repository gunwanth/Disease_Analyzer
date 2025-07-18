import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data_path = "data/augmented_dataset_50000.csv"  
df = pd.read_csv(data_path)

# Initialize and fit LabelEncoder
label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])

# Save LabelEncoder for future use
encoder_path = "src/models/savedmodels/label_encoder.pkl"
joblib.dump(label_encoder, encoder_path)
print("✅ Saved LabelEncoder for all models.")

# Split dataset
X = df.drop('Disease', axis=1)
y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Train and save each model
for model_name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    model_path = f"src/models/savedmodels/{model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, model_path)  # Save the trained model
    print(f"✅ {model_name} model saved at {model_path}")

print("✅ All models trained and saved successfully.")
