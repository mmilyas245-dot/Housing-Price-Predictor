import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# 1. Load your trained model at startup
model = joblib.load('housing_model.pkl')


@app.route('/')
def home():
  return 'Housing Price Prediction API is live and ready!'


@app.route('/predict', methods=['POST'])
def predict():
  try:
    # 2. Get JSON input sent by the user/frontend
    data = request.get_json()

    # 3. Extract feature values from request body
    features = data.get('features')

    # Convert features to 2D numpy array: [[feature1, feature2, ...]]
    input_array = np.array(features).reshape(1, -1)

    # 4. Make prediction using the loaded model
    prediction = model.predict(input_array)

    # 5. Return prediction result as JSON
    return jsonify({
        'status': 'success',
        'predicted_price': round(float(prediction[0]), 2),
    })

  except Exception as e:
    return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '_main_':
  app.run(host='0.0.0.0', port=5000)