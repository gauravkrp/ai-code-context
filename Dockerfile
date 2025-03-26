FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Celery and Redis
RUN pip install --no-cache-dir \
    celery[redis] \
    redis

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p logs metrics chroma_db

# Set default command
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 