import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import joblib
data = pd.read_csv("data/ds_salaries.csv")
data = data.drop(['salary', 'salary_currency', 'employee_residence', 'company_location'], axis=1)
label_encoders = {}
for column in ['work_year', 'experience_level', 'employment_type', 'job_title', 'company_size']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])
X = data.drop('salary_in_usd', axis=1)
y = data['salary_in_usd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
gb_model = GradientBoostingRegressor()
rf_model = RandomForestRegressor()
gb_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)
gb_predictions = gb_model.predict(X_test_scaled)
rf_predictions = rf_model.predict(X_test_scaled)
gb_loss = np.mean((gb_predictions - y_test) ** 2)
rf_loss = np.mean((rf_predictions - y_test) ** 2)
gb_mae = np.mean(np.abs(gb_predictions - y_test))
rf_mae = np.mean(np.abs(rf_predictions - y_test))
gb_rmse = np.sqrt(np.mean((gb_predictions - y_test) ** 2))
rf_rmse = np.sqrt(np.mean((rf_predictions - y_test) ** 2))


print("Gradient Boosting:")
print("Loss:", gb_loss)
print("MAE:", gb_mae)
print("RMSE:", gb_rmse)

print("\nRandom Forest:")
print("Loss:", rf_loss)
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)


joblib.dump(gb_model, 'model/gb_model.pkl', protocol=4)
joblib.dump(rf_model, 'model/rf_model.pkl', protocol=4)
joblib.dump(scaler, 'model/scaler.pkl', protocol=4)
for column, le in label_encoders.items():
    joblib.dump(le, f'model/label_encoder_{column}.pkl', protocol=4)