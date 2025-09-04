from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = 'linear_regression_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    model = None
    print("Error: Model file not found. Please train and save the model first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="Model not loaded. Please check the model file.")

    try:
        data = request.form.to_dict()
        input_data = np.array([
            float(data['Adult Mortality']),
            float(data['Alcohol']),
            float(data['Percentage Expenditure']),
            float(data['Hepatitis B']),
            float(data['Measles']),
            float(data['BMI']),
            float(data['Polio']),
            float(data['Total expenditure']),
            float(data['Diphtheria']),
            float(data['HIV/AIDS']),
            float(data['GDP']),
            float(data['Population']),
            float(data['Income composition of resources']),
            float(data['Schooling'])
        ]).reshape(1, -1)

        prediction = model.predict(input_data)[0]
        return render_template('index.html', prediction=f"Predicted Life Expectancy: {prediction:.2f}")
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
