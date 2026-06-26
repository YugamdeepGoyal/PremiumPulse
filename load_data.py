import pandas as pd
from sklearn.model_selection import train_test_split
import os
import kagglehub

path = kagglehub.dataset_download("mosapabdelghany/medical-insurance-cost-dataset")
csv_path = os.path.join(path, "insurance.csv")
df = pd.read_csv(csv_path)

def get_dataset():
    X = df.drop(columns=["charges"])
    y = df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test
