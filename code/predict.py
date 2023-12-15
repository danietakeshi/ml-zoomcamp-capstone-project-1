import pickle
from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb

input_file = 'model_eta=0.1_max_depth=6_v0.947.bin'

with open(input_file, 'rb') as f_in: 
    dv, model, features = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
	info = request.get_json()

	X = dv.transform([info])
	dX = xgb.DMatrix(X, feature_names=features)
	y_pred = model.predict(dX)
	probability = y_pred.round(1)
	churn_flag = probability > 0.5

	result = {
		'churn_probability': float(probability),
		'churn_flag': bool(churn_flag)
	}

	return jsonify(result)

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=9696)
