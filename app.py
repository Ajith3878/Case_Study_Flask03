from flask import Flask,request, url_for, redirect, render_template
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    model = pickle.load(open('model (2).pkl','rb'))
    prediction = model.predict(features)
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
