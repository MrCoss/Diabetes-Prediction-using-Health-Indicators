# Diabetes Prediction using Health Indicators

## Overview
This project builds and evaluates **Machine Learning models** to predict the likelihood of an individual having **diabetes** based on demographic, lifestyle, and health-related attributes.  
Developed as the **final capstone project** for the **IBM Course – Supervised Machine Learning: Classification**, this analysis demonstrates how predictive modeling can support **preventive healthcare**, enabling early screening and risk assessment at scale.

---

## Objective
To design a **classification model** that can identify individuals at high risk for diabetes using survey data.  
The project balances **predictive accuracy**, **interpretability**, and **public health relevance**, focusing on transparent, data-driven decision support.

---

## Dataset
**Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS 2015)  
**File:** `diabetes_binary_health_indicators_BRFSS2015.csv`  
- **Records:** 253,680  
- **Features:** 21 predictors + 1 binary target (`Diabetes_binary`)  
- **Target Distribution:**  
  - `1` → Respondent has diabetes (13.9%)  
  - `0` → Respondent does not have diabetes (86.1%)  

### Key Attributes
- **Health Factors:** `BMI`, `HighBP`, `HighChol`, `GenHlth`, `PhysHlth`, `MentHlth`
- **Lifestyle Indicators:** `Smoker`, `Fruits`, `Veggies`, `PhysActivity`, `HvyAlcoholConsump`
- **Demographics:** `Age`, `Sex`, `Education`, `Income`

---

## Data Preparation & Feature Engineering
- **Outliers:** Detected in BMI and health scores; retained after normalization.
- **Imbalance Handling:** Applied **SMOTE (Synthetic Minority Oversampling Technique)**.
- **Split:** 80% training / 20% testing.
- **Final Training Balance:** Class 0 – 104,863 | Class 1 – 52,431.
- **Feature Correlation:** `GenHlth`, `HighBP`, `DiffWalk`, `BMI`, `HighChol`, and `Age` showed strongest correlation with diabetes.

---

## Model Development
Three classification models were trained and compared using **GridSearchCV** with 5-fold cross-validation and **ROC-AUC** as the main metric.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|:------|:---------:|:----------:|:------:|:--:|:--------:|
| Logistic Regression | 0.8126 | 0.3806 | 0.5506 | 0.4501 | 0.8181 |
| Random Forest | 0.8623 | 0.5138 | 0.2130 | 0.3012 | 0.8147 |
| **Gradient Boosting (Best)** | **0.8620** | **0.5105** | **0.2334** | **0.3203** | **0.8234** |

**Best Model:** Gradient Boosting Classifier — due to highest ROC-AUC and strong generalization.

---

## Insights
- **Top Predictive Features:**  
  `GenHlth`, `HighBP`, `BMI`, `HighChol`, `Age`, `DiffWalk`.
- **Behavioral Indicators:**  
  Lower physical activity, poor nutrition, and limited mobility increased diabetes risk.
- **Interpretability:**  
  Model explanations (via feature importance) aligned with established medical literature.

---

## Key Visualizations
*(Include images if available)*  
- `feature_importances.png` – Feature influence ranking  
- `roc_curve_GradientBoosting.png` – ROC Curve for best model  
- `confusion_matrix_GradientBoosting.png` – Confusion Matrix  

---

## Results Summary
- **ROC-AUC:** 0.823  
- **Accuracy:** 86.2%  
- **Balanced Model Performance** with solid interpretability.  
- Highlights the potential of **AI in preventive healthcare analytics**.

---

## Limitations & Future Work
**Current Limitations**
- Class imbalance affects recall.
- Self-report survey data may introduce bias.
- Missing laboratory metrics (e.g., glucose levels).

**Next Steps**
- Integrate richer clinical datasets.  
- Experiment with **ADASYN** and **SMOTE-Tomek** for improved recall.  
- Implement **Explainable AI (XAI)** tools (LIME, SHAP).  
- Deploy as a **Flask/FastAPI health-risk dashboard**.

---

## Tech Stack
- **Language:** Python  
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Imbalanced-Learn  
- **Tools:** Jupyter Notebook, VS Code, GitHub  
- **Data Source:** CDC BRFSS 2015  

---

## Certification
This project was completed as part of the **IBM Course – Supervised Machine Learning: Classification**  
**Verified Certificate:** [https://coursera.org/verify/ALM4GD80QY7C](https://coursera.org/verify/ALM4GD80QY7C)  
**Instructor:** Yan Luo, Mark Grover, Miguel Maldonado – IBM  

---

## Author
**Pinto Costas Antony**  
Master of Computer Application | AI & ML Enthusiast  
[Your Email Here]  
[LinkedIn Profile](https://www.linkedin.com/in/mrcoss)  
[GitHub Repository](https://github.com/MrCoss/Diabetes-Prediction-using-Health-Indicators)

---

## 💬 Acknowledgments
Special thanks to **IBM** and **Coursera** for designing an impactful course bridging theory and application, and to the **CDC BRFSS** for open-access data enabling AI-driven healthcare research.
