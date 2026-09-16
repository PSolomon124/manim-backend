# Base image with Manim, LaTeX, and Cairo pre-installed
FROM manimcommunity/manim:latest

USER root

# Install system dependencies required for Edge-TTS audio processing & FFmpeg rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    libsox-fmt-all \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files into the container
COPY . .

# Ensure storage directories exist with write permissions for FastAPI and Manim
RUN mkdir -p /app/rendered_videos /app/media && \
    chmod -R 777 /app/rendered_videos /app/media

EXPOSE 10000

# Launch FastAPI application on Render's default port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
