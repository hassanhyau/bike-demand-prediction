from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('models/gradient_boosting_model_bike_demand.pkl')
#model = joblib.load('models/gradient_boosting_model_bike_demanda.pkl')
#model = joblib.load('models/random_forest_model_bike_demand.pkl')
#model = joblib.load('models/linear_regression_model_bike_demand.pkl')
#model = joblib.load('models/svm_model_bike_demand.pkl')

# Define the home route with an HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        features = [
            request.form.get('temp', type=float),
            request.form.get('hum', type=float),
            request.form.get('windspeed', type=float),
            request.form.get('season', type=int),
            request.form.get('yr', type=int),
            request.form.get('mnth', type=int),
            # request.form.get('hr', type=int),
            request.form.get('holiday', type=int),
            request.form.get('weekday', type=int),
            request.form.get('workingday', type=int),
            request.form.get('weathersit', type=int)
        ]

        # Check for any None values that could indicate missing form inputs
        if None in features:
            return jsonify({'prediction_text': 'Error: Missing input values. Please fill out all fields.'})

        # Prepare the feature vector for prediction
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Return the prediction as JSON
        return jsonify({'prediction_text': f'The Predicted Bike Demand for the Day is: {int(prediction[0])}'})

    except Exception as e:
        # Log the exception (if running in a local environment)
        print(f"Error: {e}")
        return jsonify({'prediction_text': f"An error occurred: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=False)
