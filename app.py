import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📩 SMS Spam Detector")
st.write("Enter a message below to check if it's spam or ham:")

# Input box
user_input = st.text_area("Message")

if st.button("Predict"):
    if user_input.strip() != "":
        # Transform and predict
        X_new = vectorizer.transform([user_input])
        prediction = model.predict(X_new)[0]
        label = "🚨 Spam" if prediction == 1 else "✅ Ham"
        st.subheader(f"Prediction: {label}")
    else:
        st.warning("Please enter a message first.")
