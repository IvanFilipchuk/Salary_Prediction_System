from dask import dataframe as dd
import joblib
import pandas as pd
from dask.distributed import Client

if __name__ == '__main__':
    gb_model = joblib.load('model/gb_model.pkl')
    rf_model = joblib.load('model/rf_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    label_encoders = {
        'work_year': joblib.load('model/label_encoder_work_year.pkl'),
        'experience_level': joblib.load('model/label_encoder_experience_level.pkl'),
        'employment_type': joblib.load('model/label_encoder_employment_type.pkl'),
        'job_title': joblib.load('model/label_encoder_job_title.pkl'),
        'company_size': joblib.load('model/label_encoder_company_size.pkl')
    }
    def preprocess_data(df):
        salary_in_usd = df['salary_in_usd']
        df = df.drop(columns=['salary', 'salary_currency', 'salary_in_usd', 'employee_residence', 'company_location'])
        for column, le in label_encoders.items():
            df[column] = le.transform(df[column])
        df_scaled = scaler.transform(df)
        return df_scaled, salary_in_usd
    schema = {
        "work_year": str,
        "experience_level": str,
        "employment_type": str,
        "job_title": str,
        "remote_ratio": int,
        "company_size": str
    }
    client = Client()
    streaming_df = dd.read_csv('data/stream/*.csv', dtype=schema)
    streaming_df_pandas = streaming_df.compute()
    preprocessed_df, salary = preprocess_data(streaming_df_pandas)
    gb_predictions = gb_model.predict(preprocessed_df)
    gb_predicted_salaries = pd.Series(gb_predictions.flatten(), name='gb_predicted_salary')
    rf_predictions = rf_model.predict(preprocessed_df)
    rf_predicted_salaries = pd.Series(rf_predictions.flatten(), name='rf_predicted_salary')
    results_df = pd.concat([streaming_df_pandas, gb_predicted_salaries, rf_predicted_salaries, salary], axis=1)
    for index, row in results_df.iterrows():
        print("Record:", row)
    gb_mae = abs(gb_predicted_salaries - salary).mean()
    rf_mae = abs(rf_predicted_salaries - salary).mean()
    if gb_mae < rf_mae:
        print("Gradient Boosting model was more precise.")
    elif gb_mae > rf_mae:
        print("Random Forest model was more precise.")
    else:
        print("Both models have the same precision.")
