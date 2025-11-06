# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app/api

# Create the application directory
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

# Install dependencies (use separate requirements files for better caching)
# Note: You'll need to create a requirements.txt in app/api/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . $APP_HOME

# Expose the port the FastAPI server runs on
EXPOSE 8000

# Command to run the application using Uvicorn
# Note: The path here is relative to the app/api/ directory
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]