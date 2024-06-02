from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
import joblib
import pandas as pd


def preprocess_streaming_data(df):
    le = LabelEncoder()
    categorical_columns = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location',
                           'company_size']
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
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "salaries") \
        .load()

    df = df.selectExpr("CAST(value AS STRING) as value")

    df = df.select(from_json(col("value"), schema).alias("data")).select("data.*")

    df.writeStream \
        .format("console") \
        .start() \
        .awaitTermination()


if __name__ == "__main__":
    process_stream()
