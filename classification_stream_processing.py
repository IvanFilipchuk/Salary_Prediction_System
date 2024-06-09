import numpy as np
import pandas as pd
import joblib
from dask.distributed import Client

def categorize_salary(salary, thresholds):
    if salary < thresholds[0]:
        return "very low"
    elif thresholds[0] <= salary < thresholds[1]:
        return "low"
    elif thresholds[1] <= salary < thresholds[2]:
        return "average"
    elif thresholds[2] <= salary < thresholds[3]:
        return "good"
    elif salary >= thresholds[3]:
        return "very good"

def generate_streaming_data(data):
    for _ in range(5):
        row = data.sample(1).iloc[0]
        yield row.to_dict()

def process_streaming_data(model, row, label_encoders, scaler, thresholds):
    df = pd.DataFrame([row])
    original_category = df['salary_in_usd'].apply(lambda x: categorize_salary(x, thresholds)).values[0]
    df = df.drop(['salary_currency', 'employee_residence'], axis=1)
    for column in ['work_year', 'experience_level', 'employment_type', 'job_title', 'company_size', 'company_location']:
        df[column] = label_encoders[column].transform(df[column])
    df = df.drop(['salary_in_usd'], axis=1)
    X = scaler.transform(df)
    prediction = model.predict(X)
    prediction_labels = [list(label_encoders['salary_category'].classes_)[pred] for pred in prediction]
    return original_category, prediction_labels

if __name__ == '__main__':
    client = Client()

    data = pd.read_csv("data/ds_salaries.csv")

    gb_model = joblib.load('model/gb_classification_model.pkl')
    rf_model = joblib.load('model/rf_classification_model.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')
    scaler = joblib.load('model/classification_scaler.pkl')
    thresholds = [
        data['salary_in_usd'].quantile(0.15),
        data['salary_in_usd'].quantile(0.3),
        data['salary_in_usd'].quantile(0.5),
        data['salary_in_usd'].quantile(0.75)
    ]

    print("\nStreaming Data Predictions:")
    for i, row in enumerate(generate_streaming_data(data), start=1):
        original_category, prediction_gb = process_streaming_data(gb_model, row, label_encoders, scaler, thresholds)
        _, prediction_rf = process_streaming_data(rf_model, row, label_encoders, scaler, thresholds)
        print(f"\nData Input {i}:")
        for key, value in row.items():
            print(f"{key}: {value}")
        print(f"Original Category: {original_category}")
        print(f"Gradient Boosting Prediction: {prediction_gb[0]}")
        print(f"Random Forest Prediction: {prediction_rf[0]}")
