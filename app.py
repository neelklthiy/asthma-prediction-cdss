from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('asthma_model.pkl')
except Exception as e:
    print("Warning: Model not found. Run model_setup.py first.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', error="Model not loaded. Please train the model first.")
        
    try:
        # Get data from form
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        smoking = int(request.form['smoking'])
        family_history = int(request.form['family_history'])
        wheezing = int(request.form['wheezing'])
        shortness_of_breath = int(request.form['shortness_of_breath'])
        chest_tightness = int(request.form['chest_tightness'])
        cough = int(request.form['cough'])
        air_pollution = int(request.form['air_pollution'])
        physical_activity = int(request.form['physical_activity'])

        # Create numpy array for prediction
        input_data = np.array([[
            age, gender, smoking, family_history, wheezing, 
            shortness_of_breath, chest_tightness, cough, 
            air_pollution, physical_activity
        ]])

        # Predict probability and class
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Calculate risk score (%)
        risk_score = probabilities[1] * 100
        
        # Determine Severity based on probability
        if risk_score < 30:
            severity = "Low Risk"
            color = "bg-green-500"
        elif risk_score < 70:
            severity = "Moderate"
            color = "bg-yellow-500"
        else:
            severity = "High/Severe"
            color = "bg-red-500"
            
        result = "Asthma Detected" if prediction == 1 else "No Asthma"
        
        return render_template('result.html', 
                               result=result, 
                               risk_score=f"{risk_score:.2f}%", 
                               severity=severity,
                               color_class=color,
                               stats=f"Based on {len(input_data[0])} health factors analyzed.")
                               
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
