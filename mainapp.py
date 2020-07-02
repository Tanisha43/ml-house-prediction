from flask import Flask, render_template, request
import pickle
import numpy as np
import math

app = Flask(__name__)
model = pickle.load(open('home.pkl','rb'))

@app.route("/")
def index():

    return render_template('index.html')


@app.route("/predict",methods=['POST'])
def predict():

	int_features =[int(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)
	#output = round(prediction[0],2)
	return render_template('index.html',prediction_text="Price of house Should be {}".format(math.floor(prediction)))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)









