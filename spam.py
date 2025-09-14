import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the prepared dataset
df = pd.read_csv("data/spam.csv")

# Convert labels to numbers
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Check result
print(df.head())
print(df[['label', 'label_num']].head(10))

# Features (X) and Labels (y)
X_text = df['text']
y = df['label_num']

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # use top 5000 words
X = vectorizer.fit_transform(X_text)

print("Shape of X:", X.shape)  # (rows, features)

# Split dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Initialize the model
model = MultinomialNB()

# Train (fit) the model on training data
model.fit(X_train, y_train)

print("Model training complete")

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Precision, Recall, F1
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved!")
