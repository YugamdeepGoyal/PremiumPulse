# Premium Pulse - Medical Insurance Cost Prediction

A supervised machine learning project that predicts **medical insurance charges** using the **Medical Insurance Cost Dataset** from Kaggle.

The project benchmarks **10 regression models**, ranging from simple linear regression to advanced gradient boosting ensembles, while maintaining consistent preprocessing pipelines and systematic hyperparameter tuning.

---

## Project Overview

Insurance charges depend on multiple demographic and lifestyle factors such as age, BMI, smoking habits, and family size.

This project aims to:

- Predict individual medical insurance costs.
- Compare the performance of multiple regression algorithms.
- Analyze the impact of feature engineering on model performance.
- Demonstrate that domain knowledge can significantly improve simpler models.

---

## Project Structure

```text
Medical-Insurance-Prediction/
│
├── load_data.py                     # Dataset loading and train/test split
├── pipeline.py                      # Preprocessing pipeline for linear models
├── pipeline_tree_model.py           # Preprocessing pipeline for tree-based models
│
├── eda.ipynb                        # Exploratory Data Analysis
├── linear_regression.ipynb          # Linear Regression model
├── ridgecv.ipynb                    # Ridge Regression with cross-validation
├── lasso_cv.ipynb                   # Lasso Regression with cross-validation
├── elastic_net.ipynb                # Elastic Net Regression
├── elastic_net_grid_searchcv.ipynb  # Elastic Net hyperparameter tuning
├── svm.ipynb                        # Support Vector Regression
├── decision_trees.ipynb             # Decision Tree Regressor
├── random_forest.ipynb              # Random Forest Regressor
├── adaboost.ipynb                   # AdaBoost Regressor
├── gradient_boosting.ipynb          # Gradient Boosting Regressor
├── xgboost.ipynb                    # XGBoost Regressor
│
├── models/
│   ├── linear_regression_pipeline.pkl   # Trained Linear Regression pipeline
│   └── xgboost_pipeline.pkl             # Trained XGBoost pipeline
│
├── images/                         # EDA plots, model evaluation graphs, and visualizations
│
└── README.md
```

---

## Dataset

- **Dataset:** Medical Insurance Cost Prediction Dataset
- **Source:** Kaggle
- **Dataset Link**: https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset
- **Target Variable:** `charges`

### Features

- Age
- Sex
- BMI
- Children
- Smoker
- Region

---

## Models Implemented

- Linear Regression
- Ridge Regression (RidgeCV)
- Lasso Regression (LassoCV)
- Elastic Net (CV)
- Elastic Net (GridSearchCV)
- Support Vector Regression (SVR)
- Decision Tree Regressor
- Random Forest Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

## Feature Engineering

One of the major findings of this project is that **domain-specific interaction features dramatically improved linear model performance.**

### Engineered Features

### `smoker_bmi`

```python
smoker_bmi = smoker × BMI
```

Captures the increased impact of BMI among smokers.

### `smoker_obese`

```python
smoker_obese = smoker AND BMI >= 30
```

Represents smokers who are also clinically obese.

These features model the real-world compounding effect of smoking and obesity on medical insurance costs.

---

## Results

| Model | Test MAE | Test R² |
|-------------------------------|----------:|---------:|
| **XGBoost** | **~1416** | **~0.8629** |
| Gradient Boosting | ~1611 | ~0.8595 |
| SVM | ~1724 | ~0.8596 |
| Decision Tree | ~1875 | ~0.8527 |
| Random Forest | ~1977 | ~0.8621 |
| **Linear Regression** | **~2391** | **~0.8690** |
| Ridge CV | ~2391 | ~0.8690 |
| Elastic Net (GridSearchCV) | ~2391 | ~0.8690 |
| Lasso CV | ~2400 | ~0.8689 |
| Elastic Net CV | ~2486 | ~0.8679 |
| AdaBoost | ~2551 | ~0.8576 |

## Key Insights

- Carefully engineered interaction features allowed **linear regression to outperform several ensemble methods in R².**
- XGBoost achieved the **lowest Mean Absolute Error (MAE)**, making it the best model for minimizing prediction error.
- The data naturally separates into three distinct groups:
  - Non-smokers
  - Smokers with normal BMI
  - Smokers with high BMI

- These groups explain the structured residual patterns observed across nearly every model.
> **Evaluation Metric:** Models were trained and tuned using cross-validation with `neg_mean_absolute_error` because target column is having outliers.
---

## Feature Importance Graph of XGBoost
![Feature Importance](images/xgboost_feature_importances.png)


## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/YugamdeepGoyal/PremiumPulse.git
```

Navigate to the project:

```bash
cd PremiumPulse
```

Open any notebook.

The dataset will be downloaded automatically using **kagglehub**.

---

## Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Scikit-learn Pipelines
- Column Transformers
- Hyperparameter Tuning
- GridSearchCV
- Cross Validation
- Model Evaluation
- Regression Analysis
- Ensemble Learning
- XGBoost
- Machine Learning Workflow

---

## License
This project is licensed under the **MIT License**.
This project is intended for educational and portfolio purposes.