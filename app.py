from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained KNN model
with open('diabetes_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.json

    # Extract features from the request
    features = [data['Pregnancies'], data['Glucose'], data['BloodPressure'], 
                data['SkinThickness'], data['Insulin'], data['BMI'], 
                data['DiabetesPedigreeFunction'], data['Age']]

    # Convert to numpy array and reshape for model input
    features = np.array(features).reshape(1, -1)

    # Predict using the KNN model
    prediction = knn_model.predict(features)[0]  # Get the prediction (0 or 1)

    # Return the prediction as a JSON response
    return jsonify({'diabetes': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
