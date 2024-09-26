import pickle
from flask import Flask, request, render_template
import numpy as np

application = Flask(__name__)
app = application

# Load the models
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('temperature'))
        RH = float(request.form.get('rh'))
        Ws = float(request.form.get('ws'))
        Rain = float(request.form.get('rain'))
        FFMC = float(request.form.get('ffmc'))
        DMC = float(request.form.get('dmc'))
        ISI = float(request.form.get('isi'))
        Classes = float(request.form.get('classes'))
        Region = float(request.form.get('region'))

        # Fix the typo here
        new_data_scaled = scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', results=result[0])
    else:
        return render_template("home.html")

if __name__ == "__main__":
    application.run(host="0.0.0.0")