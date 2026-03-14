from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_news

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Fake News Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data["news"]

    label, confidence = predict_news(text)

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)