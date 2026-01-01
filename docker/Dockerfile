FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Workdir INSIDE container
WORKDIR /app

# Copy requirements first (cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only face_recognition project
COPY . .

# Default command
CMD ["python", "-m", "app.train"]
