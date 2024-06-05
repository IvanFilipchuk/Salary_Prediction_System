FROM python:3.9

# Install Java
RUN apt-get update && \
    apt-get install -y default-jdk

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir dask scikit-learn pandas numpy dask[distributed]

EXPOSE 9999

CMD ["python", "regression_evaluation.py"]
CMD ["python", "stream_processing.py"]
