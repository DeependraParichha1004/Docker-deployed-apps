# Use the base image for Python and Streamlit
FROM python:3.9-slim

# Install system dependencies required by torchaudio
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (including torchaudio)
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Streamlit app
COPY . /app

# Expose the port for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
