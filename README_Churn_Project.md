# Customer Churn — EDA, PCA, ROC & Multi-Model ML

This project demonstrates an end-to-end **binary classification** workflow on a synthetic **Customer Churn** dataset.

## What’s included
- **Dataset**: `churn_dataset.csv` (3,000 rows, numeric + categorical features, 5% missingness in some columns).
- **Google Colab Notebook**: `Churn_EDA_PCA_ROC_ML.ipynb` — EDA, PCA visualization, multiple ML models, ROC curves, model selection & export.
- **Artifacts**: The notebook saves `best_churn_model.joblib` and `model_results.csv`.

## How to run (Google Colab)
1. Open Google Colab and upload both files:
   - `churn_dataset.csv`
   - `Churn_EDA_PCA_ROC_ML.ipynb`
2. Open the notebook in Colab.
3. (Optional) Run the first cell to install requirements if needed.
4. Update the `csv_path` variable in the **Load Data** cell to point to your uploaded `churn_dataset.csv` (or keep the default if the path matches).
5. Run all cells (Runtime → Run all).

## Models Trained
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (RBF)
- K-Nearest Neighbors
- Naive Bayes
- Multi-layer Perceptron (Neural Net)

## Evaluation
- Metrics: Accuracy, Precision, Recall, F1, **ROC-AUC**
- **ROC Curves** plotted for each model.
- Best model chosen by ROC-AUC and exported via `joblib`.

## PCA
- 2D visualization of the preprocessed training data.
- Cumulative explained variance to assess dimensionality.

## Project Structure
```
/ (root)
├── churn_dataset.csv
├── Churn_EDA_PCA_ROC_ML.ipynb
└── (generated in notebook)
    ├── best_churn_model.joblib
    └── model_results.csv
```

## Resume Bullets (ATS-friendly)
- Performed **end-to-end EDA** and feature engineering on a **3,000-row churn dataset**; handled missing values, encoded categorical variables, and standardized numeric features.
- Built and compared **7 ML models** (Logistic Regression, SVM, Random Forest, Gradient Boosting, KNN, Naive Bayes, MLP), using **ROC-AUC** and **confusion matrices** to select the best model; exported a reusable **pipeline** artifact.
