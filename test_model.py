import joblib

# Load saved model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Test messages
messages = [
    "Congratulations! You have won a free lottery. Claim now!!!",
    "Hey, are we still meeting for lunch today?"
]

# Transform and predict
X_new = vectorizer.transform(messages)
predictions = model.predict(X_new)

for msg, pred in zip(messages, predictions):
    label = "Spam" if pred == 1 else "Ham"
    print(f"{msg} --> {label}")
