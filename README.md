# imdb_reviews_sentiment_analysis
📌 Project Overview:
Movie reviews often contain strong opinions — positive or negative — and analyzing them helps in understanding public sentiment. This project builds a deep learning model that analyzes the sentiment (positive or negative) of IMDB movie reviews.

The model is deployed in an interactive Streamlit web app, where users can input or paste a review and get instant feedback on the sentiment.

🎯 Objective:
Train a deep learning model (LSTM/CNN) on the IMDB reviews dataset

Predict whether a given review is Positive or Negative

Provide a real-time and user-friendly web interface using Streamlit

🧠 Model Details:
Component	Description
Dataset	IMDB movie reviews (50,000 labeled reviews)
Model Type	LSTM / Bidirectional LSTM (Keras/TensorFlow)
Input Features	Tokenized text, padded sequences
Output	Binary: Positive (1) / Negative (0)
Accuracy	~85–90% depending on model configuration
