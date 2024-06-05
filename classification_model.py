import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, roc_auc_score
import joblib

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

def build_model(data):
    data = data.drop(['salary_currency', 'employee_residence'], axis=1)

    thresholds = [
        data['salary_in_usd'].quantile(0.15),
        data['salary_in_usd'].quantile(0.3),
        data['salary_in_usd'].quantile(0.5),
        data['salary_in_usd'].quantile(0.75)
    ]
    data['salary_category'] = data['salary_in_usd'].apply(lambda x: categorize_salary(x, thresholds))

    label_encoders = {}
    for column in ['work_year', 'experience_level', 'employment_type', 'job_title', 'company_size', 'salary_category', 'company_location']:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    data = data.drop(['salary_in_usd'], axis=1)

    X = data.drop(['salary_category'], axis=1)
    y = data['salary_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    gb_clf = GradientBoostingClassifier()
    rf_clf = RandomForestClassifier()
    gb_clf.fit(X_train_scaled, y_train)
    rf_clf.fit(X_train_scaled, y_train)

    return gb_clf, rf_clf, X_test_scaled, y_test, label_encoders, scaler, thresholds

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    loss = np.mean((predictions - y_test) ** 2)
    probabilities = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, probabilities, multi_class='ovr', average='macro')
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return accuracy, report, loss, auc, mae, rmse

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

    ddf = dd.from_pandas(data, npartitions=4)

    gb_model, rf_model, X_test_scaled, y_test, label_encoders, scaler, thresholds = build_model(ddf.compute())

    gb_accuracy, gb_report, gb_loss, gb_auc, gb_mae, gb_rmse = evaluate_model(gb_model, X_test_scaled, y_test)
    rf_accuracy, rf_report, rf_loss, rf_auc, rf_mae, rf_rmse = evaluate_model(rf_model, X_test_scaled, y_test)

    print("Gradient Boosting Classifier Metrics:")
    print("Accuracy:", gb_accuracy)
    print("Loss:", gb_loss)
    print("AUC:", gb_auc)
    print("MAE:", gb_mae)
    print("RMSE:", gb_rmse)

    print("\nRandom Forest Classifier Metrics:")
    print("Accuracy:", rf_accuracy)
    print("Loss:", rf_loss)
    print("AUC:", rf_auc)
    print("MAE:", rf_mae)
    print("RMSE:", rf_rmse)

    joblib.dump(gb_model, 'gb_classification_model.pkl')
    joblib.dump(rf_model, 'rf_classification_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')

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
