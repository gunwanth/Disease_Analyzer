# 🧠 Disease Prediction & Analysis System

A Flask-based intelligent web application that predicts diseases based on user symptoms using multiple ML models and provides detailed insights, metrics, and precautions.

## 🚀 Features

- 🔐 **Animated Login & Signup System** with session management  
- 🧾 **Symptom-Based Disease Prediction** using multiple trained ML models:
  - Decision Tree
  - Random Forest
  - KNN
  - Naive Bayes
  - SVM
  - Ensemble (Voting Classifier)
- 📊 **Model Metrics Evaluation** (Accuracy, Precision, Recall, F1-Score)
- 📈 **Data Visualizations**:
  - Disease frequency distribution
  - Symptom occurrence and weights
- 📚 **Ensemble Analysis Page**:
  - Cross-validation scores
  - Disease prediction results
  - Symptom weights from ensemble voting
- 🛡️ **Precaution & Treatment Info Page** for each predicted disease
- 🌐 **Responsive UI** with animated transitions and seamless navigation

## 🗂️ Folder Structure

```
.
├── app.py
├── data/
│   └── augmented_dataset_50000.csv
├── src/
│   └── models/
│       └── savedmodels/
│           └── [all trained .pkl models]
├── templates/
│   ├── login.html
│   ├── signup.html
│   ├── homepage.html
│   ├── index.html
│   ├── dataset_analysis.html
│   ├── ensemble_analysis.html
│   ├── disease_info.html
├── static/
│   └── graphs/
│       └── [generated graphs]
├── users.db
├── requirements.txt
└── README.md
```

## ⚙️ How It Works

1. **User logs in** or signs up using the animated frontend.
2. On the homepage, selects symptoms from the list and chooses a model.
3. The backend:
   - Converts symptoms into a feature vector.
   - Uses the selected model to predict the disease.
   - Returns results to the user.
4. Optionally, the user can:
   - See **model performance metrics**.
   - Analyze the **dataset visually**.
   - Click to view **disease information**: precautions, severity, treatment.

## 🧪 Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Models**: scikit-learn, joblib
- **Visualization**: Matplotlib, Chart.js
- **Authentication**: Flask sessions, SQLite
- **Data Handling**: Pandas, NumPy

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/disease-prediction-app.git
cd disease-prediction-app
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## 🧠 Dataset

The dataset used (`augmented_dataset_50000.csv`) is a synthetic and augmented version mapping symptoms to diseases. Label encoding and preprocessing is done before training the models.

## 📚 Future Enhancements

- Add user feedback and disease confirmation
- Add support for live disease APIs (e.g., WHO, CDC)
- Enhance visualizations using D3.js
- Include severity estimation and risk classification

## 📸 Screenshots

> Include these if you want (just drop image links or uploads):
- Login page
- Symptom input page
- Predictions and visualizations
- Disease info display

## 📄 License

MIT License. Feel free to use and modify this project for learning and research purposes.
