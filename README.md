# **Experiment-Based-on-Tam-Dataset**

## **Project Overview**
This repository explores predictive models for vaccine responses based on the dataset from Tam et al. (2016). The analysis focuses on the four vaccination schemes: **Exp-inc**, **Exp-dec**, **Constant**, and **Bolus**. Multiple modeling approaches, including Linear Regression, ARIMA, and Gradient Boosting Machine (GBM), were evaluated for their ability to forecast immune response dynamics.

---

## **Table of Contents**
1. [Background](#background)
2. [Methods](#methods)
    - [Linear Regression](#linear-regression)
    - [ARIMA](#arima)
    - [Gradient Boosting Machine](#gradient-boosting-machine)
3. [Results](#results)
4. [Limitations](#limitations)
5. [Future Directions](#future-directions)
6. [Repository Structure](#repository-structure)
7. [References](#references)

---

## **Background**
This project builds upon the work of Tam et al. (2016), which explored how antigen kinetics and availability during vaccination influence immune responses. The primary goal of this analysis is to predict response dynamics beyond the observed time range, identify patterns linking vaccination schedules to outcomes, and assess the performance of different predictive models.

---

## **Methods**

### **Linear Regression**
A baseline model used to assess the linear relationships between vaccination schemes and immune responses. While interpretable, this model struggled to capture complex patterns in the dataset.

### **ARIMA**
A time-series model employed to forecast vaccine responses. Outlier clipping was applied for preprocessing, and hyperparameters (p, d, q) were optimized using AIC. ARIMA captured temporal trends but sometimes produced biologically implausible predictions.

### **Gradient Boosting Machine**
A machine learning approach to uncover nonlinear relationships in the data. Using a sliding window, GBM was trained with optimized hyperparameters and performed recursive forecasting for test and future values.

---

## **Results**
### Model Performance Summary

| Vaccine Scheme | RMSE   | MAE   | Slope | Intercept |
|----------------|--------|-------|-------|-----------|
| **Exp-Inc**    | 113.63 | 29.84 | 0.16  | 15.54*    |
| **Exp-Dec**    | 38.63  | 7.84  | 0.04  | 4.11      |
| **Constant**   | 6.46   | 6.93  | 0.04  | 7.99*     |
| **Bolus**      | 13.04  | 9.97* | 0.07* | 2.68      |

- **Linear Regression:** Weak fits with limited predictability.
- **ARIMA:** Captured general trends but failed to enforce biological constraints.
- **GBM:** More accurate in stable scenarios but struggled with dynamic patterns.

---

## **Limitations**
1. Lack of detailed dose information for Exp-inc and Exp-dec vaccination schemes reduces analysis precision.
2. The dataset size is relatively small, limiting the performance of machine learning models.

---

## **Future Directions**
- Transform vaccination schemes into sequence-to-sequence datasets for more granular predictions.
- Explore reinforcement learning, such as Q-learning, to optimize vaccination schedules.

---

## **Repository Structure**
```plaintext
📂 Experiment-Based-on-Tam-Dataset/
├── README.md                     # Project documentation
├── final_model.py                # Final Python script for analysis
├── run_code.py                   # Script to run models and generate results
├── Linear_model.R                # R script for Linear Regression
├── Result_Plots/                 # Folder containing generated plots
│   ├── arima_predictions.png
│   ├── gbm_predictions.png
│   ├── model_performance.png
└── .gitignore                    # Git ignore file
