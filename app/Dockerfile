# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /code

# Copy the dependencies file
COPY requirements.txt .

# Install dependencies
# Upgrade pip to avoid install issues
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a writable directory for cache/temp files if needed
# (Optional, but good practice for transformers cache if not pre-downloaded)
RUN mkdir -p /tmp/cache
ENV HF_HOME=/tmp/cache

# Expose the port that HuggingFace Spaces expects (7860)
EXPOSE 7860

# Run the application using Gunicorn
# Bind to 0.0.0.0:7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
