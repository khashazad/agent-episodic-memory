# Dockerfile for ChromaDB seeder service
# Seeds multiple collections from CSV files on first database creation

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    chromadb>=1.3.5 \
    numpy>=2.0.0 \
    tqdm>=4.66.0

# Copy the seeder script
COPY seed_collections.py /app/seed_collections.py

# Make script executable
RUN chmod +x /app/seed_collections.py

# Default command
ENTRYPOINT ["python", "/app/seed_collections.py"]
