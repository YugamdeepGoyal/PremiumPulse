from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

pass_through_cols = ["age", "bmi", "children", "smoker_bmi", "smoker_obese"]
cols_to_encode = ["sex", "smoker", "region"]


def add_feature(X):
    X = X.copy()
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
