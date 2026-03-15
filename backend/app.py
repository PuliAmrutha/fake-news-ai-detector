from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_news
import os

app = Flask(__name__)
CORS(app)

# Home route
@app.route("/")
def home():
    return "Fake News Detection API Running"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.get_json()

    if "news" not in data:
        return jsonify({"error": "No news text provided"}), 400

    text = data["news"]

    label, confidence = predict_news(text)

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })


# Run the app (important for Railway)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
