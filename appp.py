# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import os


app = Flask(__name__)

# Load the saved KNN model
model_path = 'C:/Users/Industry4.0/train.pkl'
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Get input values from form
            feature1 = float(request.form['feature1'])
            feature2 = float(request.form['feature2'])
            feature3 = float(request.form['feature3'])
            feature4 = float(request.form['feature4'])

            # Prepare the input array
            input_data = np.array([[feature1, feature2, feature3, feature4]])

            # Make prediction
            prediction = model.predict(input_data)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

