import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib

from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time  # added for timestamp-based cache busting

# Initialize Flask app with template folder path
app = Flask(__name__, template_folder=os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'frontend', 'templates'))

# Load the Sentiment140 dataset (you can download this from Kaggle or use a sample dataset)
df = pd.read_csv('sentiment140.csv', encoding='latin-1', header=None)
df.columns = ['target', 'id', 'date', 'query', 'user', 'text']

# Text Preprocessing function
def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    return text.lower()

# Sentiment Analysis function using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Route for rendering the front-end page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in frontend/templates

# API route to process sentiment of text
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data['text']
    sentiment = analyze_sentiment(preprocess_text(text))
    print(f"sentiment: {sentiment}")

    # Ensure the static directory exists
    static_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Create visualization
    fig, ax = plt.subplots()
    sentiments = ['Positive', 'Negative', 'Neutral']
    values = [1 if sentiment == s else 0 for s in sentiments]

    # Assign colors
    colors = ['green' if s == 'Positive' else 'red' if s == 'Negative' else 'blue' for s in sentiments]

    ax.bar(sentiments, values, color=colors)
    ax.set_title("Sentiment Distribution")
    print(f"values: {values}")

    # Save plot
    plot_path = os.path.join(static_dir, 'sentiment_plot.png')
    plt.savefig(plot_path)
    plt.close(fig)

    # Append timestamp to prevent browser caching
    timestamp = int(time.time())
    plot_url = f'static/sentiment_plot.png?{timestamp}'

    return jsonify({'sentiment': sentiment, 'plot_url': plot_url})


if __name__ == '__main__':
    app.run(debug=True)
