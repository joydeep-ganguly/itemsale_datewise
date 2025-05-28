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

def isweekend(year,month,day):
    d = datetime(year,month,day)
    if d.weekday() > 4:
        return 1
    else:
        return 0
def day_of_week(year,month,day):
    d = datetime(year,month,day)
    return d.weekday()

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        form_inputs = pd.DataFrame(request.form.to_dict(),index=[0])
        #form_inputs['area'] = form_inputs.area.str.strip()
        form_inputs['area'] = form_inputs.area.apply(lambda x : '24 PARGANAS(S)' if x == '24PGS(S)' else x)
        form_inputs['area'] = form_inputs.area.apply(lambda x : '24 PARGANAS(N)' if x == 'NORTH 24 PGS' else x)
        form_inputs = form_inputs[(form_inputs.area!='-') & (form_inputs.area!='')]

        form_inputs.area = form_inputs.area.apply(lambda x : area_encoder[x])
        form_inputs.itemcode = form_inputs.itemcode.apply(lambda x : item_encoder[x])

        form_inputs['ddate'] = pd.to_datetime(form_inputs.ddate, format='%d-%m-%Y')
        form_inputs['day'] = form_inputs.ddate.dt.day
        form_inputs['month'] = form_inputs.ddate.dt.month
        form_inputs['year'] = form_inputs.ddate.dt.year
        form_inputs['weekday'] = form_inputs.ddate.dt.weekday
        form_inputs['weekend'] = form_inputs.weekday.apply(lambda x: 1 if x > 4 else 0)

        festival['ddate'] =  pd.to_datetime(festival.ddate, format='%d-%m-%Y')
        festival['festival'] = 1
        form_inputs = pd.merge(left=form_inputs,right=festival,left_on='ddate',right_on='ddate',how='left')
        form_inputs.festival = form_inputs.festival.fillna(0)

        form_inputs['c1'] = np.sin(form_inputs.month * (2 * np.pi / 12))
        form_inputs['c2'] = np.cos(form_inputs.month * (2 * np.pi / 12))
        form_inputs.drop('ddate', axis=1, inplace=True)
        features = form_inputs.drop(['year'],axis=1)

        X = scaler.transform(features)
        X = pd.DataFrame(X)
        X.columns = features.columns

        prediction = model.predict(X)
        return str(np.round(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)

