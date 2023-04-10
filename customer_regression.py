import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import sklearn.ensemble

def float_if_possible(strg):
    try:
        return float(strg)
    except ValueError:
        return strg


def cut_columns(columns, examp):
	arr = []
	for i in columns:
		arr.append(float_if_possible(examp[i]))
	return arr


def trainModel(data, columns, y):
	ys = []
	processed_data = []
	for i in range(len(data)):
		if data[i][5] != "0":
				processed_data.append(cut_columns(columns, data[i]))
				ys.append(float_if_possible(data[i][y]))

	X = np.array(processed_data)
	reggie = SVR(kernel='linear', C=1e0)		
	print 'Training non-customer model ', len(X), 'examples'
	reggie = reggie.fit(X, ys)
	return reggie

def get_models():
	with open('RossTrain4.csv', 'rb') as train:
		train_data = csv.reader(train)
		train_data = list(train_data)
		train_data = train_data[1:]

		# remove date, open, sales, and customer columns for no customer model
		columns = [0,1,6,7,8]
		cust_predictor = trainModel(train_data, columns, 4)
		
		
		# remove data, open, sales for the w/ customer model
		columns = [0,1,4,6,7,8]
		sales_predictor = trainModel(train_data, columns, 3)
	
	return cust_predictor, sales_predictor

def test(cust_predictor, sales_predictor, filename):
	with open(filename, 'rb') as dev:
		dev_data = csv.reader(dev)
		dev_data = list(dev_data)
		dev_data = dev_data[1:]

	processed_dev = []
	realCustomers = []
	columns = [0,1,6,7,8]
	zeroGold = 0
	for i in range(len(dev_data)):
		if dev_data[i][5] != "0":
			processed_dev.append(cut_columns(columns, dev_data[i]))
			realCustomers.append(float_if_possible(dev_data[i][4]))
		else:
			zeroGold += 1

	X = np.array(processed_dev)
	print 'Predicting ', len(X), ' customers'
	customers = cust_predictor.predict(X)

	abs_error = 0
	#print predictions
	# calculate square error
	for i,p in enumerate(customers):
		abs_error += (abs(p - realCustomers[i]))/(len(realCustomers) + zeroGold)
	print 'Abs error on customers', abs_error

	gold = []
	cust_dev = []
	counter = 0
	for i in range(len(dev_data)):
		if dev_data[i][5] != "0":
			c = customers[counter]
			cust_dev.append(processed_dev[counter][:2] + [c] + processed_dev[counter][2:])
			counter += 1
			gold.append(float_if_possible(dev_data[i][3]))
		

	X_cust = np.array(cust_dev)
	print 'Prediction ', len(X_cust), ' sales'
	predictions = cust_model.predict(X_cust)

	abs_error = 0
	#print predictions
	# calculate square error
	for i,p in enumerate(predictions):
		abs_error += (abs(p - gold[i]))/(len(gold) + zeroGold)
	print 'Abs error', abs_error

model, cust_model = get_models()
test(model, cust_model, 'RossDev4.csv')
test(model, cust_model, 'RossTest4.csv')
