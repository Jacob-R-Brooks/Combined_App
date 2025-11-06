# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV STREAMLIT_SERVER_PORT 8501
ENV APP_HOME /app/monitoring

# Create the application directory
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

# Install dependencies (use separate requirements files)
# Note: You'll need to create a requirements.txt in app/monitoring/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (app.py and IMDB Dataset.csv)
COPY . $APP_HOME

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
# We need to explicitly set the working directory for Streamlit paths to work
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]