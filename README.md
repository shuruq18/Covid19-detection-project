
# 🦠 COVID-19 Detection Project

A machine learning project that predicts whether a person is likely to have COVID-19 based on their **reported symptoms and exposure history**, using classic tabular ML (not medical imaging). The trained model is deployed through an interactive **Streamlit** web app.

> ⚠️ **Disclaimer:** This project is for **educational purposes only**. It is not a diagnostic tool and must not be used as a substitute for professional medical advice, testing, or diagnosis.

---

## 📌 Overview

Given a patient's self-reported symptoms (e.g., fever, dry cough, breathing problems) and exposure/risk factors (e.g., travel history, contact with a COVID-19 patient), the model predicts a binary outcome: **COVID-19 Positive** or **COVID-19 Negative**.

The workflow covers the full pipeline:
1. Data cleaning and preprocessing
2. Outlier removal
3. Feature encoding and scaling
4. Model training with hyperparameter tuning
5. Model evaluation
6. Deployment via a Streamlit web app

---

## 🗂️ Repository Contents

| File | Description |
|---|---|
| `Covid_19_detection_proj.ipynb` | Main notebook: data preprocessing, model training, evaluation, and exporting the trained artifacts |
| `streamlit_covid (1).py` | Streamlit web app that loads the saved model and serves an interactive symptom-checker UI |
| `README.md` | Project documentation (this file) |
| `Classwork for ML for Bioinformatics School _ Feb. 2026 - Classroom.pdf` | Supporting classwork/assignment material |

---

## 📊 Dataset

The project uses a symptoms-and-exposure dataset (`Covid Dataset.csv`) where each row is a patient record with 20 binary (**Yes/No**) features and a binary target column, `COVID-19`.

**Symptom features:**
- Breathing Problem
- Fever
- Dry Cough
- Sore throat
- Running Nose
- Headache
- Fatigue
- Gastrointestinal

**Pre-existing conditions:**
- Asthma
- Chronic Lung Disease
- Heart Disease
- Diabetes
- Hyper Tension

**Exposure / risk factors:**
- Abroad travel
- Contact with COVID Patient
- Attended Large Gathering
- Visited Public Exposed Places
- Family working in Public Exposed Places
- Wearing Masks
- Sanitization from Market

---

## 🧠 Methodology

### 1. Data Preprocessing
- Removed duplicate records
- Filled missing numeric values with the column median
- Label-encoded categorical (Yes/No) columns
- Removed outliers using the **IQR (Interquartile Range)** method
- Split data into features (`X`) and target (`y`, the `COVID-19` column)
- Standardized features using `StandardScaler`
- Split into training and test sets (80/20, stratified by target)

### 2. Model Training
- **Random Forest Classifier**, tuned via `GridSearchCV` (5-fold cross-validation) over:
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5]
- The best estimator from the grid search is selected as the final model

### 3. Evaluation
- Accuracy score
- Full classification report (precision, recall, F1-score)

### 4. Exporting Artifacts
The trained pipeline is saved with `joblib` for reuse in the web app:
- `covid_model.pkl` – trained Random Forest model
- `scaler.pkl` – fitted `StandardScaler`
- `encoders.pkl` – fitted `LabelEncoder`s for categorical columns

---

## 🖥️ Streamlit App

`streamlit_covid (1).py` loads the saved model, scaler, and encoders, then presents a simple form where a user selects **Yes/No** for each of the 20 symptom/risk factors. On clicking **Predict COVID-19**, the app:
1. Encodes the inputs using the saved encoders
2. Scales them using the saved scaler
3. Runs them through the trained model
4. Displays the result: ✅ **COVID-19 Negative** or ⚠️ **COVID-19 Positive**

---

## ⚙️ Installation & Usage

### Requirements
```bash
pip install pandas numpy scikit-learn joblib streamlit
```

### 1. Train the model (notebook)
Open `Covid_19_detection_proj.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/github/shuruq18/Covid19-detection-project/blob/main/Covid_19_detection_proj.ipynb) and run all cells. Make sure `Covid Dataset.csv` is in the same directory. This will produce `covid_model.pkl`, `scaler.pkl`, and `encoders.pkl`.

### 2. Run the web app
Place `covid_model.pkl`, `scaler.pkl`, and `encoders.pkl` in the same folder as the app script, then run:
```bash
streamlit run "streamlit_covid (1).py"
```

Open the local URL shown in the terminal (usually `http://localhost:8501`) to use the symptom-checker interface.

---

## 🛠️ Tech Stack
- **Python**
- **pandas**, **numpy** – data handling
- **scikit-learn** – preprocessing, modeling, evaluation
- **joblib** – model persistence
- **Streamlit** – web app / UI

---

## 📈 Future Improvements
- Add model comparison (e.g., Gradient Boosting, XGBoost, Logistic Regression) with cross-model benchmarking
- Add SHAP/feature-importance explainability to the app
- Add input validation and confidence scores to predictions
- Package dependencies in a `requirements.txt`

---

## 👩‍💻 Author
**Shuruq Alghamdi**
GitHub: [@shuruq18](https://github.com/shuruq18)

---

## 📄 License
No license specified yet. Consider adding one (e.g., MIT) if you'd like others to reuse this work.
