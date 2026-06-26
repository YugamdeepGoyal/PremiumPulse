import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from load_data import get_dataset

pass_through_cols = ["age", "bmi", "children", "smoker_bmi", "smoker_obese"]
cols_to_encode = ["sex", "smoker", "region"]


def add_feature(X):
    X["smoker_bmi"] = (X["smoker"] == "yes").astype(int) * X["bmi"]
    X["smoker_obese"] = ((X["smoker"] == "yes") & (X["bmi"] >= 30)).astype(int)
    return X


def get_preprocessor():
    encoder = Pipeline(
        [
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            )
        ]
    )
    preprocessor = Pipeline(
        [
            ("feature_engineer", FunctionTransformer(add_feature)),
            ("column_transformer", ColumnTransformer([
                ("passthrough", "passthrough", pass_through_cols),
                ("encoder", encoder, cols_to_encode)
            ])),
        ]
    )
    return preprocessor
