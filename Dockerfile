# Use official Manim image with LaTeX and FFmpeg pre-installed
FROM manimcommunity/manim:v0.18.0

USER root

# Install system dependencies needed for manim-voiceover audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    libsox-fmt-all \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set output folder permissions so FastAPI can write rendered videos
RUN mkdir -p /app/rendered_videos /app/media && \
    chmod -R 777 /app/rendered_videos /app/media

EXPOSE 10000

# Start FastAPI server on Render's default port 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
