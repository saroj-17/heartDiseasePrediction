from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__, template_folder='/home/saroj/Desktop/flask_app/Templates')

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    data = {
        'BMI': float(request.form['BMI']),
        'PhysicalHealth': float(request.form['PhysicalHealth']),
        'MentalHealth': float(request.form['MentalHealth']),
        'SleepTime': float(request.form['SleepTime']),
        'Smoking': request.form['Smoking'],
        'AlcoholDrinking': request.form['AlcoholDrinking'],
        'Stroke': request.form['Stroke'],
        'DiffWalking': request.form['DiffWalking'],
        'Sex': request.form['Sex'],
        'AgeCategory': request.form['AgeCategory'],
        'Race': request.form['Race'],
        'Diabetic': request.form['Diabetic'],
        'PhysicalActivity': request.form['PhysicalActivity'],
        'GenHealth': request.form['GenHealth'],
        'Asthma': request.form['Asthma'],
        'KidneyDisease': request.form['KidneyDisease'],
        'SkinCancer': request.form['SkinCancer']
    }

    # Create a DataFrame from the form data
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict_proba(df)[:, 1][0]

    # Return the result to the user
    return render_template('index.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
