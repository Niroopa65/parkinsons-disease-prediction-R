# Parkinson’s Disease Prediction Using Machine Learning in R

## Project Overview
This project demonstrates the use of **machine learning algorithms** to predict **Parkinson’s disease** based on voice measurements from the **kaggle Parkinson’s dataset**. The goal is to build, evaluate, and compare multiple models to identify the most accurate and reliable for early detection.

## Dataset
- Source: Kaggle dataset
- Features: 22 biomedical voice measurements (e.g., MDVP.Fo.Hz., Jitter, Shimmer)
- Target: `status` (0 = Healthy, 1 = Parkinson’s)

## Tools & Technologies
- **R Programming Language**
- **R Packages:** `caret`, `randomForest`, `rpart`, `rpart.plot`, `e1071`, `kernlab`, `ggplot2`, `GGally`, `corrplot`
- **IDE:** RStudio

## Methodology
1. **Data Preprocessing:** Cleaning, factor conversion, train-test split (80%-20%)
2. **Exploratory Data Analysis (EDA):** Summary stats, correlation heatmap, feature visualization
3. **Model Building:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN
4. **Model Evaluation:** Accuracy computation, ROC curves, feature importance, visualization
5. **Insights:** Identification of key predictive features and best-performing models

## Performance
- Random Forest and SVM generally achieved the highest accuracy.
- Important features: `MDVP.Fo.Hz.`, `MDVP.Jitter...`, `MDVP.Shimmer`
- Visualizations include bar plots for accuracy, decision tree plots, and feature importance graphs.


