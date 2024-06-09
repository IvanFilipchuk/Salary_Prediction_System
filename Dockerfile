FROM python:3.9

RUN apt-get update && \
    apt-get install -y default-jdk

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir dask scikit-learn pandas numpy dask[distributed] dask_ml

EXPOSE 9999

CMD ["python", "regression_evaluation.py && stream_processing.py && classification_evaluation.py && classification_stream_processing.py"]