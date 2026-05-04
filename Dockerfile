# python 3.10 chosen to match the wheel range of the pinned numpy/scipy/sklearn/pandas
FROM python:3.10-slim

LABEL description="TF-IDF + Logistic Regression baseline for the review classifier."

WORKDIR /app

COPY requirements-baseline.txt ./
RUN pip install --no-cache-dir -r requirements-baseline.txt

COPY scripts/ ./scripts/

# data is mounted at runtime, e.g.
#   docker run --rm -v "$PWD/data:/app/data" marvin-baseline
ENTRYPOINT ["python", "scripts/main.py"]
CMD ["--data-dir", "data", "--output", "data/submission.csv"]
