import joblib
import sys
import os

# allow access to utils folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess import clean_text

# load trained model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def predict_news(text):

    text = clean_text(text)

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector).max()

    label = "REAL" if prediction == 1 else "FAKE"

    return label, round(probability * 100, 2)