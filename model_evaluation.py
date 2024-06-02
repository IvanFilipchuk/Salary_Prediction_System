import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, accuracy_score
import numpy as np
import pandas as pd


def evaluate_model(model_file, X_test_file, y_test_file, tolerance=5000):
    model = joblib.load(model_file)
    X_test = pd.read_csv(X_test_file)
    y_test = pd.read_csv(y_test_file)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    accuracy = np.mean(np.abs(predictions - y_test.values.ravel()) < tolerance)

    try:
        auc = roc_auc_score((y_test > y_test.mean()).astype(int), predictions)
    except ValueError:
        auc = np.nan

    loss = mse

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'Accuracy: {accuracy}')
    print(f'AUC: {auc}')
    print(f'Loss: {loss}')


if __name__ == "__main__":
    evaluate_model('model/random_forest_model.pkl', 'data/X_test.csv', 'data/y_test.csv')
