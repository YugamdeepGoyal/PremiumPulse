from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def add_features(X):
    X = X.copy()
    X["smoker_bmi"] = (X["smoker"] == "yes").astype(int) * X["bmi"]
    X["smoker_obese"] = ((X["smoker"] == "yes") & (X["bmi"] >= 30)).astype(int)
    return X

pass_through_cols = ["age", "bmi", "children", "smoker_bmi", "smoker_obese"]
cols_to_encode = ["sex", "smoker", "region"]


def get_preprocessor():
    pass_through_pipe = Pipeline([("scaler", StandardScaler())])
    encoder = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])
    
    column_transformer = ColumnTransformer([
        ("passthrough", pass_through_pipe, pass_through_cols),
        ("encoder", encoder, cols_to_encode),
    ])
    
    preprocessor = Pipeline([
        ("feature_engineer", FunctionTransformer(add_features)),
        ("column_transformer", column_transformer),
    ])
    return preprocessor
