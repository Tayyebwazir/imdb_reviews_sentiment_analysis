import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load Keras model
model = load_model('sentiment_model.keras')

# Preprocessing function
def preprocess_review(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=200)
    return padded

# Streamlit UI
st.set_page_config(page_title="🎥 IMDB Sentiment Analysis", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.write("🔍 Paste a movie review below to analyze its sentiment!")

# Input text area
review = st.text_area("📝 Enter Review Here", height=150)

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        padded_review = preprocess_review(review)
        prediction = model.predict(padded_review)[0][0]
        sentiment = "😊 Positive" if prediction > 0.5 else "😞 Negative"
        
        st.subheader("Prediction Result:")
        st.write(f"**Sentiment:** {sentiment}")
        st.progress(int(prediction * 100) if prediction > 0.5 else int((1 - prediction) * 100))
