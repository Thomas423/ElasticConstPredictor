from flask import Flask, request, render_template

import pandas as pd
import numpy as np
import pickle
import matminer
import sys
import re

from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import Meredig

app = Flask(__name__)

@app.route('/')
def home():
     return render_template('index1.html')
    
@app.route('/predict')
def predict():
    compound_name = request.args.get('compound formula', '')
    if compound_name == '':
        formula = 'You did not enter a compound.'
    else:
        formula = compound_name
        formulas = []
        formulas.append(formula)
        
        df = pd.DataFrame((formulas), columns=['chemical_formula'])
        df = StrToComposition().featurize_dataframe(df, col_id='chemical_formula')
        df = Meredig().featurize_dataframe(df, col_id='composition')
        
        kvrh_model = pickle.load(open('kvrh_model.pkl', 'rb'))
        gvrh_model = pickle.load(open('gvrh_model.pkl', 'rb'))
        
        kvrh_pred = kvrh_model.predict(df.values[:,2:])
        gvrh_pred = gvrh_model.predict(df.values[:,2:])
        hard_pred = 0.92 * ((gvrh_pred/kvrh_pred)**1.137) * ((gvrh_pred)**0.708)
        

    return render_template('index1.html').format(kvrh_pred, gvrh_pred, hard_pred)    
    
