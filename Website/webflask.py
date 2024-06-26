from flask import Flask, request, render_template

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import Meredig






app = Flask(__name__)

@app.route('/')
def home():
     return render_template('index3.html')


@app.route('/randforest')
def randforest():
     return render_template('index4.html')




@app.route('/mlp')
def mlp():
     return render_template('index5.html')




@app.route('/predict')
def predict():
    compound_name = request.args.get('compound formula', '')
    if compound_name.isspace == True:
        formula = 'You did not enter a compound.'
        compound_name = " "
        kvrh_pred = " "
        gvrh_pred = " "
        hard_pred = " "
    else:
        formula = compound_name
        formulas = []
        formulas.append(formula)

        df = pd.DataFrame((formulas), columns=['chemical_formula'])
        df = StrToComposition().featurize_dataframe(df, col_id='chemical_formula')
        df = Meredig().featurize_dataframe(df, col_id='composition')

        kvrh_model = pickle.load(open('/home/th2302/mysite/kvrh_model.pkl', 'rb'))
        gvrh_model = pickle.load(open('/home/th2302/mysite/gvrh_model.pkl', 'rb'))

        kvrh_pred = kvrh_model.predict(df.values[:,2:])
        gvrh_pred = gvrh_model.predict(df.values[:,2:])
        hard_pred = 0.92 * ((gvrh_pred/kvrh_pred)**1.137) * ((gvrh_pred)**0.708)


    return render_template('index4.html').format(compound_name, kvrh_pred, gvrh_pred, hard_pred)



@app.route('/predict2')
def predict2():
    bulk_model = tf.keras.models.load_model('/home/th2302/Website/Bulk_model.keras')
    shear_model = tf.keras.models.load_model('/home/th2302/Website/Shear_model.keras')

    compound_name = request.args.get('compound formula', '')
    if compound_name.isspace == True:
        formula = 'You did not enter a compound.'
        compound_name = " "
        kvrh_pred = " "
        gvrh_pred = " "
        hard_pred = " "
    else:
        formula = compound_name
        formulas = []
        formulas.append(formula)

        df = pd.DataFrame((formulas), columns=['chemical_formula'])
        df = StrToComposition().featurize_dataframe(df, col_id='chemical_formula')
        df = Meredig().featurize_dataframe(df, col_id='composition')



        x = df.values[:,2:]
        x_arr = np.array(x)
        x_tensor = tf.convert_to_tensor(x_arr, dtype=tf.float64)


        kvrh_pred = bulk_model.predict(x_tensor)
        gvrh_pred = shear_model.predict(x_tensor)
        hard_pred = 0.92 * ((gvrh_pred/kvrh_pred)**1.137) * ((gvrh_pred)**0.708)


    return render_template('index5.html').format(compound_name, kvrh_pred, gvrh_pred, hard_pred)
