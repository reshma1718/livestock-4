import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app4 = Flask(__name__)
model4 = pickle.load(open('model4.pkl', 'rb'))

@app4.route('/')
def home():
    return render_template('index4.html')

@app4.route('/predict',methods=['POST'])
def predict():
   
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model4.predict(final_features)
    
    output = round(prediction[0], 2)

    return render_template('index4.html', prediction_text='predicted goat census to be million {}'.format(output))

if __name__ == "__main__":
    app4.run(debug=True) 