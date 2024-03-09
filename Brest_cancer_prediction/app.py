from flask import Flask,request,render_template
import pandas
import numpy as np
import pickle

model=pickle.load(open("/config/workspace/Brest_cancer_prediction/model_cancer.pkl",'rb'))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    features=request.form['feature']
    features_lst=features.split(',')
    np_features=np.asarray(features_lst,dtype=np.float32)
    pred=model.predict(np_features.reshape(1,-1))

    output=["Cancrouse" if pred[0]==1 else "Not Cancrouse"]
    return render_template('index.html',message=output)
if __name__=="__main__":
    app.run(host="0.0.0.0")
