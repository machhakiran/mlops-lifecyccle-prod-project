# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
COPY requirements-docker.txt requirements.txt

# 4. Install system dependencies and Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

# 5. Copy the entire project into the image
COPY . .

# 6. Copy Model Artifacts (Prepared by prep script)
# This directory 'docker_model_context' must be created before build
COPY docker_model_context /app/model_artifacts

# 7. Configure Inference to use Local Model
# This matches the new logic in src/serving/inference.py
ENV MODEL_DIR_PATH=/app/model_artifacts
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# 8. Expose FastAPI port
EXPOSE 8000

# 9. Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 10. Run the FastAPI app
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
