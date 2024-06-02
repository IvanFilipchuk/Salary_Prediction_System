import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(input_file):
    df = pd.read_csv(input_file)

    df['salary_in_usd'] = df['salary_in_usd'].astype(float)

    le = LabelEncoder()
    categorical_columns = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location',
                           'company_size']
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    df[categorical_columns + ['salary_in_usd']] = scaler.fit_transform(df[categorical_columns + ['salary_in_usd']])

    return df


if __name__ == "__main__":
    df = preprocess_data('data/ds_salaries.csv')
    df.to_csv('data/processed_salaries.csv', index=False)
