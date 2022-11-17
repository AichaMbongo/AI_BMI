from flask import render_template,request
from app import app
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb')) 
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home',data='hey')

@app.route("/prediction",methods=["POST"])
def prediction():
    height=float(request.form['height'])
    weight=float(request.form['weight'])
    arr=np.array([[height,weight]])
    pred=model.predict(arr)
    return render_template('prediction.html',data=pred)

if __name__ == "__main__":
    app.run(debug=True)