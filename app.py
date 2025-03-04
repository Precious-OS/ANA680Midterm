# app.py
import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('models', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Mapping for prediction output (assuming race/ethnicity groups from the dataset)
race_mapping = {0: 'Group A', 1: 'Group B', 2: 'Group C', 3: 'Group D', 4: 'Group E'}


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form data
        math_score = float(request.form['math_score'])
        reading_score = float(request.form['reading_score'])
        writing_score = float(request.form['writing_score'])

        # Calculate total score
        total_score = math_score + reading_score + writing_score

        # Prepare input for model
        input_data = np.array([[math_score, reading_score, writing_score, total_score]])

        # Make prediction
        pred = model.predict(input_data)[0]
        prediction = race_mapping.get(pred, "Unknown")

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))