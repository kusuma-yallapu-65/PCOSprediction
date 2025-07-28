import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
rfcmodel=pickle.load(open('rfcmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    output=rfcmodel.predict((np.array(list(data.values())).reshape(1,-1)))
    print(output[0])
    return jsonify(int((output[0])))

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=rfcmodel.predict(final_input)[0]
    messages = {
        0: "No PCOS detected. Maintain a healthy lifestyle, including a balanced diet and regular exercise, to reduce the risk of developing PCOS.",
        1: "You are likely to have PCOS. Please consult a doctor for further evaluation and guidance. In the meantime, consider making lifestyle changes such as eating a balanced diet, exercising regularly, and managing stress to help manage symptoms."
    }

    return render_template("home.html",prediction_text=messages[output])



if __name__=="__main__":
    app.run(debug=True)
   
     