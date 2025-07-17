# Predicting-Online-Advertising-Click-Through-Rates-CTR-Using-Machine-Learning-and-NLP

## Overview

This project focuses on predicting whether a user will click on an online advertisement using structured features (e.g., demographics, browsing behavior) and unstructured text data (ad headlines). Accurate CTR prediction is critical for optimizing ad delivery, improving user experience, and increasing ROI for advertisers.

This project applies diverse set of machine learning models, enhance them with feature engineering (timestamp extraction, TF-IDF on ad content), and validate performance using rigorous evaluation metrics. The final model is a weighted soft voting ensemble, achieving an accuracy of **0.86** and demonstrating excellent generalization and interpretability.

![CTR](https://github.com/user-attachments/assets/22f485db-4d1d-49a9-9183-121fecdb3f62)


---

## Problem Statement

- **Task:** Predict the likelihood (`click = 1 or 0`) that a user will click on an online advertisement.
- **Dataset:** 10,000 samples with 9 predictors including demographics, internet usage, ad topic, and timestamp.
- **Target:** `click` (binary classification)

---

## Key Features

- **Balanced Dataset:** 49% click, 51% no-click — no resampling needed  
- **High-Cardinality Features:** Handled via frequency encoding (e.g., `City`, `Country`)  
- **Temporal Features:** Extracted from timestamp (e.g., `Hour_of_Day`, `Weekend`)  
- **Textual Features:** Processed using TF-IDF followed by PCA (from `Ad_Topic_Line`)  

---

## Models Implemented

| Model                    | Accuracy | F1-Score (Click=1) | AUC   |
|--------------------------|----------|--------------------|-------|
| Logistic Regression (L2) | 0.805    | 0.790              | 0.896 |
| KNN                      | 0.814    | 0.807              | 0.879 |
| Random Forest            | 0.837    | 0.831              | 0.923 |
| SVM (RBF Kernel)         | 0.811    | 0.801              | 0.894 |
| Feedforward NN (FNN)     | 0.812    | 0.803              | 0.872 |
| XGBoost                  | 0.850    | 0.844              | 0.894 |
| **Ensemble (Final)**     | **0.852**| **0.847**          | **0.918** |

> 🔧 Ensemble model uses **weighted soft voting** (XGBoost: 0.6, RF: 0.35, KNN: 0.05)

---

## Methodology

### Data Preprocessing

- **Standardization:** For numerical features  
- **Encoding:** Frequency encoding (categorical), TF-IDF + PCA (text), timestamp decomposition  
- **Handling Sparsity:** Avoided one-hot for high-cardinality fields  

### Training & Validation

- Stratified split: 60% train / 20% validation / 20% test  
- Cross-validation + GridSearch for hyperparameter tuning  
- Metrics: Accuracy, Precision, Recall, F1-Score, AUC, Learning Curve  

---

## Key Insights

- **Temporal Trends:** CTR spikes around 8–9 PM and on the 4th, 17th, and 27th of the month  
- **User Demographics:** Older users and users from higher income areas are more likely to click  
- **Text Signals:** TF-IDF + PCA improved performance by extracting useful information from ad headlines  

---

## Related Techniques

- Logistic Regression with L2 for interpretability  
- SVM with RBF kernel for nonlinear user behaviors  
- Tree-based models (RF, XGBoost) for robustness and feature interaction  
- Feedforward Neural Network for structured + textual data  
- Soft Voting Ensemble for performance aggregation  

---

## Visualization Excerpts

<img width="624" height="492" alt="Screenshot 2025-07-16 at 21 03 01" src="https://github.com/user-attachments/assets/a2c4825a-36ab-4157-81ef-45c79d418bd5" />
<img width="640" height="344" alt="Screenshot 2025-07-16 at 21 03 27" src="https://github.com/user-attachments/assets/f384f7a3-8bdb-4bc8-b058-d1ccbbc994c8" />
<img width="558" height="380" alt="Screenshot 2025-07-16 at 21 03 46" src="https://github.com/user-attachments/assets/38f277b4-58a8-4083-b4c7-6473273de849" />


