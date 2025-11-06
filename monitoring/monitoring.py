import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
APP_ROOT_DIR = os.path.dirname(SCRIPT_DIR)
LOGS_DIR = os.path.join(APP_ROOT_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "prediction_logs.json")
IMDB_FILE_PATH = os.path.join(SCRIPT_DIR, "IMDB Dataset.csv") 

# --- Data Loading Functions ---
@st.cache_data(show_spinner="Reading and parsing logs...")
def load_full_logs(file_path):
    """
    Reads the full NDJSON log file and returns a DataFrame.
    This function is used to feed the metrics and distribution plots.
    """
    if not os.path.exists(file_path):
        return pd.DataFrame()
    
    log_data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        log_entry = json.loads(line)
                        log_data.append(log_entry)
                    except json.JSONDecodeError:
                        continue
    except Exception:
        return pd.DataFrame()
        
    return pd.DataFrame(log_data)

@st.cache_data(show_spinner="Reading and processing IMDB dataset...")
def load_and_process_imdb(file_path):
    """
    Reads the IMDB CSV, calculates review length, and prepares sentiment data.
    """
    if not os.path.exists(file_path):
        st.error(f"IMDB Dataset file not found at: **{file_path}**")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"An error occurred while reading the IMDB file: {e}")
        return pd.DataFrame()

    if 'review' not in df.columns or 'sentiment' not in df.columns:
        st.error("IMDB Dataset must contain columns named 'review' and 'sentiment'.")
        return pd.DataFrame()

    # Calculate sentence length for the density plot
    df['length'] = df['review'].apply(len)
    
    # Standardize sentiment label and rename for consistent plotting
    df['label'] = df['sentiment'].str.title() # e.g., 'positive' -> 'Positive'
    df['source'] = 'IMDB Dataset'
    
    return df[['length', 'label', 'source']]

# --- Data Processing for Plots and Metrics ---
def get_sentiment_counts(df_logs, df_imdb):
    """Combines predicted and training sentiment counts for plotting."""
    
    # Prepare Logged Predictions DF
    df_log_plot = df_logs.copy()
    df_log_plot = df_log_plot.rename(columns={'predicted_sentiment': 'label'})
    df_log_plot['type'] = 'Logged Predictions'

    # Prepare IMDB Sentiments DF
    df_imdb_plot = df_imdb.copy()
    df_imdb_plot['type'] = 'Training Data'

    # Combine the two DataFrames for counting
    combined_sentiment_df = pd.concat([
        df_log_plot[['label', 'type']], 
        df_imdb_plot[['label', 'type']]
    ]).dropna(subset=['label']) # Drop any rows where sentiment/prediction is missing
    
    # Calculate final counts for plotting
    counts = combined_sentiment_df.groupby(['type', 'label']).size().reset_index(name='count')
    
    return counts

# --- Load Data at App Start ---
full_logs_df = load_full_logs(LOG_FILE_PATH)
df_imdb_full = load_and_process_imdb("IMDB Dataset.csv")
#df_imdb_full = load_and_process_imdb(IMDB_FILE_PATH)

# Prepare DataFrames for plotting components
# 1. Length DF (for Density Plot)
df_logs_length = full_logs_df.copy()
if not df_logs_length.empty:
    df_logs_length['length'] = df_logs_length['request_text'].apply(len)
    df_logs_length['source'] = 'Logged Inference Request'
    df_logs_length = df_logs_length[['length', 'source']]
    
# 2. Sentiment Counts DF (for Bar Chart)
sentiment_counts_df = pd.DataFrame()
if not full_logs_df.empty and not df_imdb_full.empty:
    sentiment_counts_df = get_sentiment_counts(full_logs_df, df_imdb_full)

# Combine for Density Plot
combined_df_length = pd.concat([df_logs_length, df_imdb_full[['length', 'source']]]).reset_index(drop=True)

# --- Streamlit Application Layout ---

st.set_page_config(
    page_title="Prediction Log Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Model & Data Analysis Dashboard")
st.markdown("---")

## 🚀 Live Performance Metrics

# Filter the logs to include only records where true_sentiment is provided
labeled_df = full_logs_df.dropna(subset=['true_sentiment']).copy()

if not labeled_df.empty:
    # 1. Prepare Ground Truth (y_true) and Predictions (y_pred)
    y_true = labeled_df['true_sentiment'].str.title()
    y_pred = labeled_df['predicted_sentiment'].str.title()
    
    # 2. Calculate Metrics
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        
        st.subheader(f"Performance Metrics (N={len(labeled_df)} Labeled Samples)")
        
        col1, col2, col3 = st.columns(3)
        
        # Display Metrics
        col1.metric("**Live Accuracy**", f"{accuracy:.3f}")
        col2.metric("**Weighted Precision**", f"{precision:.3f}")
        col3.metric("Labeled Count", f"{len(labeled_df)}")
        
        st.markdown("---")

        # 3. Performance Alert Logic
        ACCURACY_THRESHOLD = 0.8
        
        if accuracy < ACCURACY_THRESHOLD:
            st.error(
                f"**PERFORMANCE ALERT:** Model Accuracy ({accuracy:.3f}) "
                f"has dropped below the threshold of {ACCURACY_THRESHOLD:.1f}!"
            )
        else:
            st.success("Model performance is currently stable.")
            
    except Exception as e:
        st.error(f"Error calculating metrics. Ensure predicted and true sentiments have matching labels. Details: {e}")

else:
    st.info("No performance metrics available yet. Need more logs with a `true_sentiment` (true label) to calculate.")

st.markdown("---")

## 📉 Text Length Density Plot

st.subheader("Text Length Density Plot: Logged Requests vs. Training Data")

if combined_df_length.empty or len(combined_df_length['source'].unique()) < 2:
    st.warning("Could not load sufficient data from both sources to generate the density plot.")
else:
    fig_density = px.histogram(
        combined_df_length,
        x="length",
        color="source",
        histnorm='probability density',
        nbins=100,
        marginal="box",
        opacity=0.6,
        barmode='overlay',
        title="Distribution of Text Length (Character Count)"
    )
    
    st.plotly_chart(fig_density, use_container_width=True)

    st.caption(
        "This plot compares the character length distribution of the inference requests "
        "against the original training/development data. Significant differences "
        "could indicate **data drift**."
    )

st.markdown("---")

## 📊 Sentiment Distribution Comparison

st.subheader("Predicted vs. Training Sentiment Distribution")

if not sentiment_counts_df.empty:
    fig_sentiment = px.bar(
        sentiment_counts_df,
        x='label',
        y='count',
        color='type',
        barmode='group',
        labels={'label': 'Sentiment Label', 'count': 'Count', 'type': 'Data Source'},
        title='Sentiment Distribution: Live Predictions vs. Training Data'
    )

    fig_sentiment.update_layout(
        xaxis_title="Sentiment Label",
        yaxis_title="Frequency (Count)",
        legend_title="Data Source"
    )
    
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    st.caption(
        "This chart compares how frequently each sentiment label appears in the training data "
        "to how frequently it is predicted in live inference requests."
    )
else:
    st.info("Insufficient data to generate the sentiment distribution plot.")

st.markdown("---")

## 📝 Raw Processed Log Data

st.subheader("Raw Processed Log Data (First 10 Rows)")
if not full_logs_df.empty:
    st.dataframe(full_logs_df.head(10), use_container_width=True)
else:
    st.info("No prediction log data found.")