# ML-Based Diabetes Risk Prediction Using Clinical Data

> **Can we predict diabetes risk early using simple clinical data?**
> This project explores that question using machine learning, domain knowledge, and a fully reproducible pipeline.

![Status](https://img.shields.io/badge/Status-Research%20Project-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

---

## Why This Project Matters

Diabetes is one of the fastest-growing chronic diseases worldwide, especially in regions where access to early diagnostic tools is limited.

This project was built with a simple goal:

>  **Use machine learning to assist early diabetes risk detection using easily available clinical data.**

By combining healthcare knowledge with machine learning, this system aims to support:

* Early screening
* Risk awareness
* Preventive healthcare strategies

---

##  What This Project Does

This project builds a complete machine learning pipeline to predict diabetes risk using the **Pima Indian Diabetes Dataset**.

It includes:

* Medical-aware data preprocessing
* Feature engineering (8 → 24+ features)
* Comparison of 7 ML models
* Threshold optimization (sensitivity vs specificity)
* Performance evaluation (ROC, confusion matrix)
* Reproducible training pipeline
* Production-ready prediction function

The focus is not just accuracy—but **interpretability, reliability, and real-world usability**.

---

## System Architecture

![Architecture](https://raw.githubusercontent.com/binodsth2/ML-Based_Diabetes_Risk_Prediction_Using_Clinical_Data/7163cdd10033d46469a6279e68f680e0c9930b72/overall_architecture.svg)

---

## 📊 Key Results

The model demonstrates strong predictive performance on unseen data:

* 🎯 **Accuracy:** 82.1%
* ⚖️ **F1-Score:** 0.798
* 📈 **ROC-AUC:** 0.887

👉 This shows the model can effectively distinguish between diabetic and non-diabetic cases, making it useful for **screening-level applications**.

---

## 📊 Visual Results

### Model Performance

![Performance]([images/model_performance.png](https://raw.githubusercontent.com/binodsth2/ML-Based_Diabetes_Risk_Prediction_Using_Clinical_Data/810242870bf04d1e029302c91a01fb7cf1f12edb/model_performance_comparison.svg))

### ROC Curve

![ROC](https://raw.githubusercontent.com/binodsth2/ML-Based_Diabetes_Risk_Prediction_Using_Clinical_Data/7163cdd10033d46469a6279e68f680e0c9930b72/roc_curve.svg)

### Confusion Matrix

![Confusion]([images/confusion_matrix.png](https://raw.githubusercontent.com/binodsth2/ML-Based_Diabetes_Risk_Prediction_Using_Clinical_Data/19e50c69724c897f81d3d5ca1fb6e3a19b79e38f/confusion_matrix.svg))

---

## 🧠 Feature Engineering (Why it Matters)

Instead of relying only on raw clinical data, additional features were engineered based on medical intuition:

* BMI categories → obesity risk levels
* Age groups → population segmentation
* Interaction features → capture non-linear relationships
* Clinical ratios → improve prediction quality

👉 This significantly improved model performance and interpretability.

---

## ⚙️ Installation

```bash
git clone https://github.com/binodsth2/ML-Based_Diabetes_Risk_Prediction_Using_Clinical_Data
cd ML-Based_Diabetes_Risk_Prediction_Using_Clinical_Data
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run Full Analysis

```bash
jupyter notebook Untitled12.ipynb
```

### Quick Prediction

```python
from joblib import load

model = load('best_diabetes_model.pkl')
scaler = load('scaler.pkl')

patient = {
    'Pregnancies': 2,
    'Glucose': 120,
    'BloodPressure': 80,
    'SkinThickness': 30,
    'Insulin': 100,
    'BMI': 28.5,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 35
}

result = predict_diabetes_risk_enhanced(patient)
print(result)
```

---

## 📁 Project Structure

```
├── data/
├── images/
├── notebooks/
├── models/
├── Untitled12.ipynb
├── best_diabetes_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## 🧪 Model Performance

```
Accuracy:   82.1%
Precision:  79.5%
Recall:     80.3%
F1-Score:   0.798
ROC-AUC:    0.887
```

✔ Low overfitting
✔ Stable across validation sets
✔ Suitable for real-world screening

---

## ⚖️ Threshold Optimization

| Threshold | Sensitivity | Specificity |
| --------- | ----------- | ----------- |
| 0.40      | 89%         | 62%         |
| 0.50      | 78%         | 75%         |
| 0.60      | 68%         | 85%         |


---

##  Limitations

* Dataset is population-specific (Pima Indian dataset)
* Limited clinical features (no lifestyle or medication data)

##  What I Learned

This project helped me understand:

* Data quality matters more than model complexity
* Feature engineering is critical in healthcare ML
* Model interpretability is as important as accuracy
* Small datasets require careful validation

It strengthened my interest in applying AI to **healthcare and real-world impact problems**.

---

##  Future Improvements

* Add explainable AI (SHAP / LIME)
* Use larger and more diverse datasets
* Deploy as a web application (Streamlit)
* Integrate real-time prediction API

---

## Research Paper



##  Author

**Binod Shrestha**
📧 [binods021419@nec.edu.np](mailto:binods021419@nec.edu.np)

---

## ⭐ If you find this useful

Consider giving it a ⭐ on GitHub — it helps a lot!
