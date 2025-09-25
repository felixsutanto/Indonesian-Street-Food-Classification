# deployment/Dockerfile

# 1. Base Image - Sticking with a stable version of Debian (Bookworm)
FROM python:3.9-slim-bookworm

# 2. Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Working Directory
WORKDIR /app

# 4. Install system dependencies that might be required by CV libraries
#    - libgl1 is the modern replacement for libgl1-mesa-glx
#    - libgomp1 is added for OpenMP support, sometimes needed by CV libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copy the requirements file and install dependencies
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the API source code and the trained model
#    These paths are relative to the build context (the parent directory)
COPY ../api/ ./
COPY ../models/ ./models/

# 7. Expose Port
EXPOSE 8000

# 8. Run Application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

