from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
def preprocess_streaming_data(df):
    le = LabelEncoder()
    categorical_columns = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    scaler = StandardScaler()
    df[categorical_columns + ['salary_in_usd']] = scaler.transform(df[categorical_columns + ['salary_in_usd']])
    return df

def process_stream():
    spark = SparkSession.builder.appName("StreamProcessing").getOrCreate()

    schema = StructType([
        StructField("work_year", IntegerType()),
        StructField("experience_level", StringType()),
        StructField("employment_type", StringType()),
        StructField("job_title", StringType()),
        StructField("salary", FloatType()),
        StructField("salary_currency", StringType()),
        StructField("salary_in_usd", FloatType()),
        StructField("employee_residence", StringType()),
        StructField("remote_ratio", IntegerType()),
        StructField("company_location", StringType()),
        StructField("company_size", StringType())
    ])
    model = joblib.load('model/random_forest_model.pkl')
    df = spark.readStream \
        .schema(schema) \
        .csv('data/stream')
    df_pandas = df.toPandas()
    df_processed = preprocess_streaming_data(df_pandas)
    X = df_processed.drop(columns=['salary_in_usd'])

    df = df.withColumn("prediction", model.predict(X))

    query = df.writeStream \
        .outputMode("append") \
        .format("console") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    process_stream()
