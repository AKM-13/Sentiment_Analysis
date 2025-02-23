import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the saved model
model = tf.keras.models.load_model('sentiment_lstm_model (1).h5')

# Load the saved tokenizer
with open('tokenizer (1).pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
    

# Preprocessing function
def preprocess_text(text, maxlen=200):
    text = text.lower()  # Convert to lowercase
    sequences = tokenizer.texts_to_sequences([text])  # Tokenize the text
    st.write(f"Tokenized sequence: {sequences}")  # Debug: Show tokenized sequence
    padded_sequence = pad_sequences(sequences, maxlen=maxlen)  # Pad sequence
    return padded_sequence

# Function to predict sentiment
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)[0][0]  # Probability between 0 and 1
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    emoji = "🙂" if sentiment == "Positive" else "😞"
    return sentiment, prediction * 100, emoji

# Streamlit UI
st.title("Sentiment Analysis App")

# Input text from the user
input_text = st.text_area("Enter a movie review:")

# When the user clicks the button, make the prediction
if st.button("Predict Sentiment"):
    if input_text.strip():
        sentiment, probability, emoji = predict_sentiment(input_text)
        st.markdown(f"### The sentiment of the review is: **{sentiment}** {emoji}")
        st.markdown(f"### Prediction Probability: **{probability:.2f}%**")
    else:
        st.write("Please enter a review to predict the sentiment.")
