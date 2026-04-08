# Bullseye for the same reason — psycopg2 needs PostgreSQL C libraries
FROM python:3.10-bullseye

WORKDIR /app

# Install PostgreSQL development libraries
# gcc         → C compiler needed to build psycopg2 from source
# postgresql-client → PostgreSQL command line tools
# libpq-dev   → PostgreSQL C header files psycopg2 needs to compile
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]