import streamlit as st
import pandas as pd
from textblob import TextBlob
import speech_recognition as sr
import matplotlib.pyplot as plt
import io

# Global variables for storing data
results = []

# Speech-to-Text Conversion
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Calibrating for background noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        st.info("Listening... Speak now!")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Sentiment Analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, polarity

# File Upload and Speech-to-Text
def process_audio_file(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Data Visualization
def plot_sentiment_distribution(data):
    # Create a DataFrame with all three columns
    df = pd.DataFrame(data, columns=["Text", "Sentiment", "Polarity"])
    
    # Count occurrences of each sentiment
    sentiment_counts = df["Sentiment"].value_counts()

    # Define colors for sentiments dynamically (can be adjusted based on sentiment types)
    sentiment_colors = {
        'Positive': 'green',
        'Negative': 'red',
        'Neutral': 'gray'
    }

    # Assign colors based on the sentiment type
    colors = [sentiment_colors.get(sentiment, 'gray') for sentiment in sentiment_counts.index]

    # Plot
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", color=colors, ax=ax, width=0.1)  # Adjust width here
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Sentiment")
    st.pyplot(fig)



# Download Results as CSV
def download_results(data):
    df = pd.DataFrame(data, columns=["Text", "Sentiment", "Polarity"])
    csv = df.to_csv(index=False)
    return csv

# Streamlit UI
st.set_page_config(page_title="Speech Sentiment Analysis", layout="wide")
st.title("🎙️ Speech Sentiment Analysis")
st.write("Analyze the sentiment of spoken words in real-time!")

# Sidebar
st.sidebar.header("Options")
use_microphone = st.sidebar.checkbox("Use Microphone", value=True)
upload_file = st.sidebar.file_uploader("Or Upload an Audio File", type=["wav", "flac"])

# Main App
if use_microphone and st.button("Start Recording"):
    st.info("Recording in progress...")
    text = speech_to_text()
    if text:
        st.success(f"Transcribed Text: {text}")
        sentiment, polarity = analyze_sentiment(text)
        results.append([text, sentiment, polarity])
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Polarity Score:** {polarity:.2f}")
        st.success("Analysis Complete!")

if upload_file:
    st.info("Processing uploaded file...")
    text = process_audio_file(upload_file)
    if text:
        st.success(f"Transcribed Text: {text}")
        sentiment, polarity = analyze_sentiment(text)
        results.append([text, sentiment, polarity])
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Polarity Score:** {polarity:.2f}")
        st.success("Analysis Complete!")

# Display Data
if results:
    st.header("📊 Analysis Results")
    df = pd.DataFrame(results, columns=["Text", "Sentiment", "Polarity"])
    st.table(df)

    # Plot Sentiment Distribution
    st.header("📈 Sentiment Distribution")
    plot_sentiment_distribution(results)

    # Download CSV
    st.header("📥 Download Results")
    csv_data = download_results(results)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="sentiment_results.csv",
        mime="text/csv",
    )
