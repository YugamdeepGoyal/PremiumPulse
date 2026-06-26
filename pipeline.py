import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.linear_model import LinearRegression, ElasticNetCV
from load_data import get_dataset

pass_through_cols = ["age", "bmi", "children"]
cols_to_encode = ["sex", "smoker", "region"]


def get_preprocessor():
    pass_through_pipe = Pipeline([("scaler", StandardScaler())])
    encoder = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(
        [
            ("passthrough", pass_through_pipe, pass_through_cols),
            ("encoder", encoder, cols_to_encode),
        ]
    )
    return preprocessor
