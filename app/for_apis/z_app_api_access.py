from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('models/bike_demand_gradient_boosting_model.pkl')


# Define the home route
@app.route('/')
def home():
    return "Welcome to the Bike Demand Prediction API"


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the request
    data = request.get_json(force=True)

    # Prepare the feature vector for prediction
    features = np.array([
        data['season'],
        data['yr'],
        data['mnth'],
        data['hr'],
        data['holiday'],
        data['weekday'],
        data['workingday'],
        data['weathersit'],
        data['temp'],
        data['atemp'],
        data['hum'],
        data['windspeed']
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=False)
