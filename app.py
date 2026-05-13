from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# 1. Yahan double underscore aayega
app = Flask(__name__) 

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return "Housing Price Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]

        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)

        return jsonify({
            "prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# 2. Yahan bhi double underscores aayenge
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)