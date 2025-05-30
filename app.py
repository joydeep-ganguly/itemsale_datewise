import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from flask import Flask, render_template, request


area_encoder = pickle.load(open('./item_encoder_area.pkl','rb'))
item_encoder = pickle.load(open('./item_encoder_item.pkl','rb'))
festival = pd.read_pickle('./festivals.pkl')
scaler = pickle.load(open('./item_scaler.pkl','rb'))
model = pickle.load(open('./item_model.pkl','rb'))

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        form_itemsale = pd.DataFrame(request.form.to_dict(),index=[0])

        #form_itemsale['area'] = form_itemsale['area'].apply(lambda x : '24 PARGANAS(S)' if x == '24PGS(S)' else x)
        #form_itemsale['area'] = form_itemsale['area'].apply(lambda x : '24 PARGANAS(N)' if x == 'NORTH 24 PGS' else x)
        #form_itemsale = form_itemsale[(form_itemsale['area'] != '-') & (form_itemsale['area'] != '')]

        form_itemsale['area'] = form_itemsale['area'].apply(lambda x : area_encoder[x])
        form_itemsale['itemcode'] = form_itemsale['itemcode'].apply(lambda x : item_encoder[x])

        form_itemsale['ddate'] = pd.to_datetime(form_itemsale.ddate, format='%d-%m-%Y')
        form_itemsale['day'] = form_itemsale.ddate.dt.day
        form_itemsale['month'] = form_itemsale.ddate.dt.month
        form_itemsale['year'] = form_itemsale.ddate.dt.year
        form_itemsale['weekday'] = form_itemsale.ddate.dt.weekday
        form_itemsale['weekend'] = form_itemsale.weekday.apply(lambda x: 1 if x > 4 else 0)

        festival['ddate'] =  pd.to_datetime(festival.ddate, format='%d-%m-%Y')
        festival['festival'] = 1
        form_itemsale = pd.merge(left=form_itemsale,right=festival,left_on='ddate',right_on='ddate',how='left')
        form_itemsale.festival = form_itemsale.festival.fillna(0)

        form_itemsale['c1'] = np.sin(form_itemsale.month * (2 * np.pi / 12))
        form_itemsale['c2'] = np.cos(form_itemsale.month * (2 * np.pi / 12))
        form_itemsale.drop('ddate', axis=1, inplace=True)
        features = form_itemsale.drop(['year'],axis=1)
        return features.to_html()

        X = scaler.transform(features)
        X = pd.DataFrame(X)
        X.columns = features.columns
        #return X.to_html()

        prediction = model.predict(X)
        #return str(np.round(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)

