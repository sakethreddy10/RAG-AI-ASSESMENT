# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Set the working directory inside the container
WORKDIR /code

# Install system dependencies (optional, but good for stability)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code
COPY . /code/

# Expose Hugging Face Space default port
EXPOSE 7860

# Start the FastAPI server using Uvicorn on port 7860
# Note: If you don't commit the 'db/' folder to git, we run ingestion at startup if './db' is missing.
CMD ["sh", "-c", "if [ ! -d './db' ]; then python src/ingest.py; fi && uvicorn main:app --host 0.0.0.0 --port $PORT"]
