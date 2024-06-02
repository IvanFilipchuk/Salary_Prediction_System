import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
import numpy as np


def evaluate_model(model_file, X_test, y_test):
    model = joblib.load(model_file)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')


if __name__ == "__main__":
    from model_training import train_model

    model, X_test, y_test = train_model('data/processed_salaries.csv')
    evaluate_model('model/random_forest_model.pkl', X_test, y_test)
