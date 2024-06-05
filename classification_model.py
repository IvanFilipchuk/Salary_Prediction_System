import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, \
    roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import time

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
    data = data.drop(['salary_currency', 'employee_residence', 'company_location'], axis=1)

    thresholds = [
        data['salary_in_usd'].quantile(0.15),
        data['salary_in_usd'].quantile(0.3),
        data['salary_in_usd'].quantile(0.5),
        data['salary_in_usd'].quantile(0.75)
    ]
    data['salary_category'] = data['salary_in_usd'].apply(lambda x: categorize_salary(x, thresholds))

    label_encoders = {}
    for column in ['work_year', 'experience_level', 'employment_type', 'job_title', 'company_size', 'salary_category']:
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

    return gb_clf, rf_clf, X_test_scaled, y_test

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


data = pd.read_csv("data/ds_salaries.csv")

gb_model, rf_model, X_test_scaled, y_test = build_model(data)

gb_accuracy, gb_report, gb_loss, gb_auc, gb_mae, gb_rmse = evaluate_model(gb_model, X_test_scaled, y_test)
rf_accuracy, rf_report, rf_loss, rf_auc, rf_mae, rf_rmse = evaluate_model(rf_model, X_test_scaled, y_test)

print("Gradient Boosting Classifier Metrics:")
print("Accuracy:", gb_accuracy)
print("Loss:", gb_loss)
print("AUC:", gb_auc)
print("MAE:", gb_mae)
print("RMSE:", gb_rmse)
print("Classification Report:")
print(gb_report)

print("\nRandom Forest Classifier Metrics:")
print("Accuracy:", rf_accuracy)
print("Loss:", rf_loss)
print("AUC:", rf_auc)
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("Classification Report:")
print(rf_report)

joblib.dump(gb_model, 'gb_classification_model.pkl')
joblib.dump(rf_model, 'rf_classification_model.pkl')
