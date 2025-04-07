import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load the model and scaler with absolute paths
ridge_model = pickle.load(open(r'C:\Users\HP\Desktop\aws deployment project\model\model.pkl', 'rb'))
standard_scaler = pickle.load(open(r'C:\Users\HP\Desktop\aws deployment project\model\scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=["GET", "POST"])
def predict_data():
    if request.method=="POST":
        try:
            # Get all the form data and convert to appropriate types
            Temperature = float(request.form.get("Temperature"))
            RH = float(request.form.get("RH"))
            Ws = float(request.form.get("Ws"))
            Rain = float(request.form.get("Rain"))
            FFMC = float(request.form.get("FFMC"))
            DMC = float(request.form.get("DMC"))
            ISI = float(request.form.get("ISI"))
            Classes = request.form.get("Classes")
            Region = request.form.get("Region")
            
            # Include all 9 features or adjust according to your model requirements
            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, 0, 0]])  # Adjust as needed
            
            # Make prediction - using ridge_model instead of model
            prediction = ridge_model.predict(new_data_scaled)
            
            # Get the prediction value and format it for display
            result = f"The predicted FWI is: {prediction[0]:.2f}"
            
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

if __name__ =="__main__":
    app.run(host="0.0.0.0")