from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Загрузка предварительно обученной модели (предположим, что она есть в папке models)
MODEL_PATH = "models/patient_monitoring_model.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None  # Можно заменить заглушкой или создать заглушечную модель

@app.route('/')
def home():
    return "AI-powered Patient Monitoring System is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)  # Преобразование входных данных
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
