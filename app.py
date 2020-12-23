#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np


lr = joblib.load("model.pkl") # Load "model.pkl"
model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"    
    
# Your API definition
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify("TITANIC PREDICTOR")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        json_ = request.get_json()
        query = pd.get_dummies(pd.DataFrame(json_))
        query = query.reindex(columns=model_columns, fill_value=0)
        prediction = list(lr.predict(query))

        return jsonify({'prediction': str(prediction)})

    except:
        
        #return jsonify("error happening")
        return jsonify({'trace': traceback.format_exc()})
        
if __name__ == '__main__':
    
    app.run(debug=True)


# In[ ]:




