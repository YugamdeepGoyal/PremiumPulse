from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

def print_results(y_true, y_pred):
    print("R2 Score", r2_score(y_true, y_pred))
    print("MAE", mean_absolute_error(y_true, y_pred))
    print("MSE", mean_squared_error(y_true, y_pred))
    print("RMSE", root_mean_squared_error(y_true, y_pred))