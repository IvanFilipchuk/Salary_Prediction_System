import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


def train_model(data_file):
    df = pd.read_csv(data_file)

    X = df.drop(columns=['salary_in_usd'])
    y = df['salary_in_usd']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'model/random_forest_model.pkl')

    return model, X_test, y_test


if __name__ == "__main__":
    train_model('data/processed_salaries.csv')
