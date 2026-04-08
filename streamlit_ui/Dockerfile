FROM python:3.10-bullseye

WORKDIR /app

# Same PostgreSQL libraries as consumer_service
# because streamlit_ui also connects directly to the database
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Tell Docker this container listens on port 8501
# This is required for the port mapping in docker-compose to work
EXPOSE 8501

# Start Streamlit on port 8501
# --server.address=0.0.0.0 means accept connections from outside
# the container (i.e. your browser on Windows)
# Without 0.0.0.0 it would only accept connections from inside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]