import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import os

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Create index.html file if it doesn't exist
with open('templates/index.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Forest Fire Prediction</title>
</head>
<body>
    <h1>Welcome to Forest Fire Prediction Application</h1>
    <p>Click <a href="/predictdata">here</a> to start making predictions.</p>
</body>
</html>""")

# Create home.html file if it doesn't exist
with open('templates/home.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Fire Weather Index Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
        }
        input {
            width: 100%;
            padding: 8px;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #f2f2f2;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Fire Weather Index Prediction</h1>
        
        <form method="POST" action="/predictdata">
            <div class="form-group">
                <label for="Temperature">Temperature (°C):</label>
                <input type="number" step="0.01" id="Temperature" name="Temperature" required>
            </div>
            
            <div class="form-group">
                <label for="RH">Relative Humidity (%):</label>
                <input type="number" step="0.01" id="RH" name="RH" required>
            </div>
            
            <div class="form-group">
                <label for="Ws">Wind Speed (km/h):</label>
                <input type="number" step="0.01" id="Ws" name="Ws" required>
            </div>
            
            <div class="form-group">
                <label for="Rain">Rain (mm):</label>
                <input type="number" step="0.01" id="Rain" name="Rain" required>
            </div>
            
            <div class="form-group">
                <label for="FFMC">Fine Fuel Moisture Code:</label>
                <input type="number" step="0.01" id="FFMC" name="FFMC" required>
            </div>
            
            <div class="form-group">
                <label for="DMC">Duff Moisture Code:</label>
                <input type="number" step="0.01" id="DMC" name="DMC" required>
            </div>
            
            <div class="form-group">
                <label for="ISI">Initial Spread Index:</label>
                <input type="number" step="0.01" id="ISI" name="ISI" required>
            </div>
            
            <div class="form-group">
                <label for="Classes">Classes:</label>
                <input type="text" id="Classes" name="Classes">
            </div>
            
            <div class="form-group">
                <label for="Region">Region:</label>
                <input type="text" id="Region" name="Region">
            </div>
            
            <button type="submit">Predict</button>
        </form>
        
        {% if result %}
        <div class="result">
            <h3>Prediction Result:</h3>
            <p>{{ result }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>""")

# Initialize Flask application
application = Flask(__name__, template_folder='templates')
app = application  # This ensures compatibility with both local development and deployment

# Add a test route that doesn't depend on templates or models
@app.route('/test')
def test():
    return "Hello, this is a test route! The application is running."

# Load the model and scaler with proper error handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'model.pkl')
scaler_path = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

try:
    ridge_model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    # Define fallback behavior
    ridge_model = None
    scaler = None

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error in index route: {e}")
        return f"An error occurred: {e}", 500

@app.route('/predictdata', methods=["GET", "POST"])
def predict_data():
    if request.method == "POST":
        try:
            # Check if models were loaded successfully
            if ridge_model is None or scaler is None:
                return render_template('home.html', result="Error: Models could not be loaded.")
            
            # Get all the form data and convert to appropriate types
            Temperature = float(request.form.get("Temperature"))
            RH = float(request.form.get("RH"))
            Ws = float(request.form.get("Ws"))
            Rain = float(request.form.get("Rain"))
            FFMC = float(request.form.get("FFMC"))
            DMC = float(request.form.get("DMC"))
            ISI = float(request.form.get("ISI"))
            
            # Handle optional fields
            Classes = request.form.get("Classes", "0")
            Region = request.form.get("Region", "0")
            
            # Convert Classes and Region to numeric values if needed
            # This assumes you've handled these categorical variables in your model training
            classes_value = 0  # Default value, modify based on your model
            region_value = 0   # Default value, modify based on your model
            
            # Include all features needed by your model
            new_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, classes_value, region_value]]
            print(f"Input data before scaling: {new_data}")
            
            # Transform the data
            new_data_scaled = scaler.transform(new_data)
            print(f"Input data after scaling: {new_data_scaled}")
            
            # Make prediction
            prediction = ridge_model.predict(new_data_scaled)
            
            # Get the prediction value and format it for display
            result = f"The predicted Fire Weather Index (FWI) is: {prediction[0]:.2f}"
            
            # Print to console for debugging
            print("Prediction completed:", result)
            
            # Return the template with the result
            return render_template('home.html', result=result)
        
        except Exception as e:
            # Log the error and return it to the template
            error_message = f"Error during prediction: {str(e)}"
            print(error_message)
            return render_template('home.html', result=error_message)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)