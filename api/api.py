# -*- coding: utf-8 -*-

import json
import os
import joblib
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

# --- Configuration ---
# directory for monitoring
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 

# root dir for app
APP_ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Log File: /app/logs/prediction_logs.json
LOGS_DIR = os.path.join(APP_ROOT_DIR, "logs")

LOG_FILE = os.path.join(LOGS_DIR, "prediction_logs.json")
MODEL_PATH = "sentiment_model.pk1" 

# Ensure the logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Sentiment Prediction API",
)

# Global variable to hold the loaded model
sentiment_model = None

# --- Application Startup Event ---
@app.on_event("startup")
async def load_model():
    """Loads the model using joblib when the application starts."""
    global sentiment_model
    try:
        # Change: Use joblib.load() instead of pickle.load()
        sentiment_model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Using mock prediction.")
        sentiment_model = None
    except Exception as e:
        print(f"Error loading model: {e}. Using mock prediction.")
        sentiment_model = None

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    """Schema for the incoming prediction request."""
    text: str = Field(..., description="The text string to analyze.")
    true_label: Optional[str] = Field(None, description="Optional true sentiment label for monitoring.")

class PredictionResponse(BaseModel):
    """Schema for the outgoing prediction response."""
    predicted_sentiment: str

# --- Logging Function (runs in background) ---

def write_prediction_log(
    timestamp: str, 
    request_text: str, 
    predicted_sentiment: str, 
    true_sentiment: Optional[str]
):
    """Writes a single prediction log entry to the JSON file."""
    # Note: Log keys match the original request (timestamp, request_text, predicted_sentiment, true_sentiment)
    log_entry = {
        "timestamp": timestamp,
        "request_text": request_text,
        "predicted_sentiment": predicted_sentiment,
        "true_sentiment": true_sentiment # Logs the optional true label
    }
    
    # Write as a new line in the JSON log file (NDJSON format)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# --- Prediction Endpoint ---

@app.post(
    "/predict", 
    response_model=PredictionResponse, 
    status_code=200,
    summary="Make a sentiment prediction and log the request"
)
async def predict_sentiment(
    data: PredictionRequest, 
    background_tasks: BackgroundTasks
):
    """
    Receives text, makes a prediction using the loaded model, and logs the details asynchronously.
    """
    global sentiment_model
    
    if sentiment_model:
        # 1. **Model Prediction Logic**
        # The model is expected to take text and return a prediction label (e.g., 'Positive', 'Negative')
        try:
            # Note: Your model might require specific pre-processing (like vectorization) here.
            # Assuming the model's .predict() method works on a list containing the text.
            prediction_result = sentiment_model.predict([data.text])
            predicted_sentiment = prediction_result[0] 
        except Exception as e:
            print(f"Prediction error: {e}. Falling back to mock prediction.")
            # Fallback mock prediction if model inference fails
            predicted_sentiment = "Error/Unknown"
    else:
        print("ERROR MODEL NOT FOUND")

    # 2. **Logging using BackgroundTasks**
    current_time = datetime.now().isoformat()
    
    background_tasks.add_task(
        write_prediction_log,
        timestamp=current_time,
        request_text=data.text,
        predicted_sentiment=predicted_sentiment,
        true_sentiment=data.true_label # Uses the input true_label
    )

    # 3. **Return Response**
    return PredictionResponse(predicted_sentiment=predicted_sentiment)
