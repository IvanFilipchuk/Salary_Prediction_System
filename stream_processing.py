from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType, StructType, StructField, StringType, IntegerType
import pandas as pd
import joblib


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

def preprocess_streaming_data(df):
    for column, le in label_encoders.items():
        df[column] = le.transform(df[column])
    df_scaled = scaler.transform(df)
    return df_scaled

schema = StructType([
    StructField("work_year", StringType()),
    StructField("experience_level", StringType()),
    StructField("employment_type", StringType()),
    StructField("job_title", StringType()),
    StructField("remote_ratio", IntegerType()),
    StructField("company_size", StringType())
])


spark = SparkSession.builder.appName("SalaryPredictionStream").getOrCreate()

@pandas_udf(FloatType())
def predict_salary(*cols):
    cols = [pd.Series(c) for c in cols]
    data = pd.concat(cols, axis=1)
    data.columns = ["work_year", "experience_level", "employment_type", "job_title", "remote_ratio", "company_size"]
    data_preprocessed = preprocess_streaming_data(data)
    predictions = gb_model.predict(data_preprocessed)

    return pd.Series(predictions.flatten())


df = spark.readStream \
    .schema(schema) \
    .csv('data/stream')


df = df.withColumn("predicted_salary", predict_salary("work_year", "experience_level", "employment_type", "job_title", "remote_ratio", "company_size"))


query = df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
