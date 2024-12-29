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
- **Linear Regression:** Weak fits with limited predictability.
- **ARIMA:** Captured general trends but failed to enforce biological constraints.
- **GBM:** More accurate in stable scenarios but struggled with dynamic patterns.

---

## **Limitations**
1. Lack of detailed dose information for Exp-inc and Exp-dec vaccination schemes reduces analysis precision.
2. The dataset size is relatively small, limiting the performance of machine learning models.

---

## **Future Directions**
- Transform vaccination schemes into sequence-to-sequence datasets for more precise predictions.
- Explore reinforcement learning, such as Q-learning, to optimize vaccination schedules.

---

## **Repository Structure**
- **Linear_model.R**: R script for linear regression modeling and visualization.
- **final_model.py**: Python script implementing ARIMA and GBM models.
- **run_code.py**: Workflow for running the entire analysis pipeline.
- **Result_Plots/**: Generated plots comparing measured vs. predicted responses.
- **README.md**: Project overview and documentation.

---

## **References**
Tam, H. H., Melo, M. B., Kang, M., Pelet, J. M., Ruda, V. M., Foley, M. H., Hu, J. K., Kumari, S., Crampton, J., Baldeon, A. D., et al. (2016). Sustained antigen availability during germinal center initiation enhances antibody responses to vaccination. *PNAS, 113*(43). [https://doi.org/10.1073/pnas.1606050113](https://doi.org/10.1073/pnas.1606050113)
