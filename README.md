# Model-Evaluation-Refinement-Project
This project details the development and refinement of a profit prediction model using the Superstore dataset. It covers data exploration, feature engineering, outlier treatment, and the evaluation of various regression models (Ridge, Gradient Boosting, XGBoost) to achieve a highly accurate predictive solution for business profitability.

# Profit Prediction Model Evaluation and Refinement

## Project Overview

This project focuses on building and refining a predictive model for `Profit` using the `sample_superstore.csv` dataset. The goal was to overcome initial model underfitting by systematically applying data preprocessing, feature engineering, and evaluating different regression algorithms.

## Key Steps Performed

1.  **Data Loading and Initial Inspection:**
    *   Loaded `sample_superstore.csv` using `latin1` encoding.
    *   Inspected data types (`df.info()`) and summary statistics (`df.describe()`).
    *   Converted `Order Date` and `Ship Date` to datetime objects.
    *   Confirmed no missing values in the dataset.

2.  **Initial Model Evaluation (Ridge Regression & Gradient Boosting Regressor):**
    *   Split data into training (80%) and testing (20%) sets.
    *   Used `ColumnTransformer` for preprocessing (one-hot encoding categorical features).
    *   **Ridge Regression:** Both initial and hyperparameter-tuned models (via Grid Search) showed severe underfitting, with R-squared values around -0.80 to -0.71 on the test set.
    *   **Gradient Boosting Regressor:** Initial and tuned models also suffered from significant underfitting, with R-squared values around -0.02 to -0.24 on the test set.
    *   Visualizations (Actual vs. Predicted, Residual Plots) clearly depicted these models' inability to capture data patterns.

3.  **Deeper Data Exploration & Feature Engineering:**
    *   **Date-based Features:** Extracted `Order_Year`, `Order_Month`, `Order_Day`, `Order_DayOfWeek`, `Ship_Year`, `Ship_Month`, `Ship_Day`, `Ship_DayOfWeek`, and `Ship_Duration`.
    *   **Distribution Analysis:** Visualizations of `Profit` and `Sales` revealed heavily skewed distributions and numerous extreme outliers, identified as a primary cause for model underfitting.

4.  **Outlier Treatment (Winsorization) & Advanced Models:**
    *   **Winsorization:** Applied Winsorization to `Profit` and `Sales` at the 1st and 99th percentiles to mitigate the impact of extreme outliers.
    *   **Retrained Gradient Boosting Regressor (with Winsorized + FE data):** This resulted in a dramatic improvement, achieving an R-squared of **0.85** on the test set. Visualizations showed a clear alignment of predictions with actuals and well-distributed residuals.
    *   **XGBoost Regressor:** Introduced XGBoost, a powerful boosting algorithm, to the preprocessed data. Achieved an outstanding R-squared of **0.93** on the test set without extensive tuning.

## Results & Conclusions

Our analysis successfully identified and overcame challenges posed by extreme outliers and skewed distributions in the profit data. Through systematic data preprocessing, intelligent feature engineering, and the critical application of Winsorization, we transformed models from severe underfitting to high predictive accuracy.

**The XGBoost Regressor, utilizing the winsorized 'Profit' as the target and winsorized 'Sales' along with engineered date features, stands out as the best-performing model with an R-squared of 0.93.** This model is capable of providing highly reliable insights into factors driving profitability and supports better business decision-making.

## Visualizations

Key visualizations illustrating model performance (Actual vs. Predicted, Residual Plots) are included within the Jupyter notebook itself. You can view these directly by opening the `.ipynb` file on GitHub or in a Jupyter environment.

*   **Link to Jupyter Notebook:** [Model_Evaluation_and_Refinement.ipynb](Model_Evaluation_and_Refinement.ipynb)

## How to Run the Notebook

To reproduce the analysis:

1.  Clone this repository to your local machine.
2.  Ensure you have Python (3.7+) and necessary libraries installed (e.g., `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`).
3.  Open the `.ipynb` file in a Jupyter environment (e.g., Jupyter Lab, VS Code with Python extension, Google Colab).
4.  Run all cells in sequence.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hadi02-gif/Model_Evaluation_Refinement_Project/blob/main/Model_Evaluation_and_Refinement.ipynb)
