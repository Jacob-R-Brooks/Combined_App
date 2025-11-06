import requests
import json
import os
from sklearn.metrics import accuracy_score, precision_score

# --- Configuration ---
# The endpoint where your FastAPI application is running
PREDICT_URL = "http://localhost:8000/predict"
# Name of the test data file
TEST_DATA_FILE = "test.json" 

# --- Functions ---

def load_test_data(file_path):
    """Loads the test data from the specified JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Test data file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None

def run_tests_and_collect_results(test_data):
    """Sends requests to the API and collects true and predicted labels."""
    y_true = []
    y_pred = []
    
    print(f"Sending {len(test_data)} requests to {PREDICT_URL}...")
    
    for i, item in enumerate(test_data):
        
        # Prepare Request Payload
        payload = {
            "text": item.get("text"),
            "true_label": item.get("true_label") # This goes to FastAPI and is logged
        }
        
        # Send POST Request
        try:
            response = requests.post(PREDICT_URL, json=payload)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            # 3. Extract Prediction
            prediction_result = response.json()
            predicted_sentiment = prediction_result.get("predicted_sentiment")
            
            # 4. Collect Results
            true_label = item.get("true_label")
            
            if true_label and predicted_sentiment:
                # Ensure labels are standardized (Title Case) for comparison
                y_true.append(true_label.title()) 
                y_pred.append(predicted_sentiment.title())
            
        except requests.exceptions.RequestException as e:
            print(f"\nRequest failed for item {i+1}: {e}")
            break
        
    return y_true, y_pred

def calculate_and_print_metrics(y_true, y_pred):
    """Calculates and prints the final accuracy and precision scores."""
    if not y_true:
        print("\nNo valid predictions were collected to calculate metrics.")
        return

    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    # Using 'weighted' precision for a balanced view across classes
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n" + "="*40)
    print(" FINAL MODEL EVALUATION ")
    print(f"Total labeled samples tested: {len(y_true)}")
    print("="*40)
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print("="*40)


if __name__ == "__main__":
    #Load Data
    test_data = load_test_data(TEST_DATA_FILE)
    
    if test_data:
        # run test and print
        y_true, y_pred = run_tests_and_collect_results(test_data)
        calculate_and_print_metrics(y_true, y_pred)