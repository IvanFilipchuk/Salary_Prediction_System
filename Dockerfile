# Use an official Python runtime as a parent image
FROM python:3.9-slim
WORKDIR /app
COPY src/ /app/
COPY data/ /app/data/
RUN pip install pandas
RUN pip install pandas
RUN pip install pandas
RUN pip install pandas
EXPOSE 8888
ENV NAME World
CMD ["python", "stream_processing.py"]
