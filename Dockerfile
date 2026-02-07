# Use the official Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and set the working directory
WORKDIR /code

# Copy the requirements first to leverage Docker cache
# (I'll create a requirements.txt if it doesn't exist)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Move app files to root if necessary or adjust the command
# The app is in /app directory, but we want to run app.py
# Let's adjust the working directory for the command
WORKDIR /code/app

# Expose the port (HF Spaces uses 7860)
EXPOSE 7860

# Command to run the app
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
