# Sprint 11 - containerize the whole pipeline so it runs identically anywhere.
#
# Build (repo root):
#   docker build -t cadetx-pipeline .
#
# Run (mount data + collect results; paths inside the container):
#   docker run --rm ^
#     -v "%cd%/online_retail_II.xlsx:/app/data/input.xlsx:ro" ^
#     -v "%cd%/results:/app/results" ^
#     cadetx-pipeline --input /app/data/input.xlsx --output /app/results
#
# CSV inputs work the same way (--input /app/data/input.csv); the CLI
# converts CSV -> workbook staging before Module 1 reads it.

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first (layer cache survives code edits).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then the pipeline code, configs and versioned contracts.
COPY module1_profiling/ ./module1_profiling/
COPY module2_cleaning/*.py module2_cleaning/expected_schema.json ./module2_cleaning/
COPY module3_validation/*.py ./module3_validation/
COPY module4_pipeline/*.py module4_pipeline/config.yaml ./module4_pipeline/
COPY pipeline.py .

ENTRYPOINT ["python", "pipeline.py"]
CMD ["--help"]
