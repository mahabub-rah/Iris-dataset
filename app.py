from flask import Flask, request, render_template
import numpy as np
import pickle

# load model
model = pickle.load(open('model/model.pkl', 'rb'))

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sel_len = float(request.form.get('sepal_length'))
    sel_wid = float(request.form.get('sepal_width'))
    pel_len = float(request.form.get('petal_length'))
    pel_wid = float(request.form.get('petal_width'))

    feature= np.array([[sel_len,sel_wid,pel_len,pel_wid]])
    pred = model.predict(feature)
    prediction = f'Prediction Class is: {pred[0]}'

    return render_template('index.html', prediction= prediction)


if __name__=="__main__":
    app.run(debug=True)