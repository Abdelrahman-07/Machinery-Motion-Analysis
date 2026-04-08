# Base image: official Python 3.10 on Debian Bullseye
# Bullseye (Debian 11) is used instead of "slim" because it includes
# the C libraries that OpenCV and psycopg2 need to compile correctly
FROM python:3.10-bullseye

# Set the working directory inside the container to /app
# All subsequent commands run from this directory
# All files get copied into this directory
WORKDIR /app

# Install system-level libraries that OpenCV requires
# These are C libraries — they can't be installed with pip
# libglib2.0-0  → core Linux utility library OpenCV depends on
# libsm6        → X11 session management needed for video processing
# libxext6      → X11 extensions library
# libxrender-dev → rendering support for display operations
# libgomp1      → GNU OpenMP, enables parallel processing in PyTorch
# libgl1        → OpenGL library — fixes "libGL.so.1 not found" error
# netcat-openbsd → networking tool used to check if Kafka port is open
# "rm -rf /var/lib/apt/lists/*" deletes apt's download cache
# keeping the final Docker image smaller
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first (before the rest of the code)
# Docker caches each step — if requirements.txt hasn't changed,
# Docker skips re-running pip install on the next build (saves time)
COPY requirements.txt .

# Install all Python packages from requirements.txt
# --no-cache-dir prevents pip from saving downloads, keeping image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy all remaining Python files into /app inside the container
# This includes main.py, detector.py, motion.py, classifier.py, producer.py
COPY . .

# The command that runs when the container starts
# Equivalent to typing "python main.py" in the terminal
CMD ["python", "main.py"]