# Use a minimal base image with Python installed
FROM python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from this directory (deployment/) into the container's /app
COPY . .

# Install system dependencies needed by some ML libraries, then Python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces expects the app to listen on port 7860
EXPOSE 7860

# Streamlit-specific environment settings for headless container deployment
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Launch the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
