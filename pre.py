import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Function to preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# Function to train model
def train_model():
    # Predefined past data (DO NOT CHANGE FORMAT)
    data = {
        "email": [
            "I loved the demo. How do I purchase it?",
            "Can you offer a discount?",
            "Thanks for the follow-up. I’ll buy it now.",
            "Not interested at the moment, maybe later."
        ],
        "bought": [0, 0, 1, 0]  # 1 = Bought, 0 = Not Bought
    }

    df = pd.DataFrame(data)
    df["cleaned_email"] = df["email"].apply(clean_text)

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(df["cleaned_email"], df["bought"])
    return model

# Function to predict purchase based on email
def predict_purchase(email_text):
    model = train_model()  # Train the model on predefined data
    cleaned = clean_text(email_text)
    prediction = model.predict([cleaned])[0]
    return "Likely to Buy" if prediction == 1 else "Not Likely to Buy"

# 🚀 Example Usage (Only provide email, get prediction)
email_input = "I am interested in your product. Can you share pricing?"
print(f"Prediction: {predict_purchase(email_input)}")

email_input2 = "Not sure if I want to buy right now."
print(f"Prediction: {predict_purchase(email_input2)}")
