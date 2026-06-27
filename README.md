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
├── load_data.py                  # Dataset loading and train/test split
├── pipeline.py                   # Preprocessing pipeline for linear models
├── pipeline_tree_model.py        # Preprocessing pipeline for tree-based models
├── eda.ipynb                     # Exploratory Data Analysis
├── linear_regression.ipynb
├── ridgecv.ipynb
├── lasso_cv.ipynb
├── elastic_net.ipynb
├── elastic_net_grid_searchcv.ipynb
├── svm.ipynb
├── decision_trees.ipynb
├── random_forest.ipynb
├── adaboost.ipynb
├── gradient_boosting.ipynb
└── xgboost.ipynb
```

---

## Dataset

- **Dataset:** Medical Insurance Cost Prediction Dataset
- **Source:** Kaggle
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

| Model | Test R² | Test MAE |
|------|---------:|---------:|
| **Linear Regression** | **~0.8690** | **~2391** |
| Ridge CV | ~0.8690 | ~2391 |
| Elastic Net (GridSearchCV) | ~0.8690 | ~2391 |
| Lasso CV | ~0.8689 | ~2400 |
| Elastic Net CV | ~0.8679 | ~2486 |
| **XGBoost** | **~0.8629** | **~1416** |
| Random Forest | ~0.8621 | ~1977 |
| SVM | ~0.8596 | ~1724 |
| Gradient Boosting | ~0.8595 | ~1611 |
| AdaBoost | ~0.8576 | ~2551 |
| Decision Tree | ~0.8527 | ~1875 |

---

## Key Insights

- Carefully engineered interaction features allowed **linear regression to outperform several ensemble methods in R².**
- XGBoost achieved the **lowest Mean Absolute Error (MAE)**, making it the best model for minimizing prediction error.
- The data naturally separates into three distinct groups:
  - Non-smokers
  - Smokers with normal BMI
  - Smokers with high BMI
- These groups explain the structured residual patterns observed across nearly every model.

---

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost kagglehub
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

This project is intended for educational and portfolio purposes.