import csv
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def float_if_possible(strg):
    try:
        return float(strg)
    except ValueError:
        return strg

def train():
	with open('RossTrain2.csv', 'rb') as train:
		train_data = csv.reader(train)
		train_data = list(train_data)
		train_data = train_data[1:]
		
		y = []

		# remove date, open, and sales columns
		for i in range(len(train_data)):
			y.append(float_if_possible(train_data[i][3]))
			train_data[i] = [float_if_possible(train_data[i][0]), float_if_possible(train_data[i][1]),float_if_possible(train_data[i][4]), float_if_possible(train_data[i][6]), float_if_possible(train_data[i][7]), float_if_possible(train_data[i][8])]

	X = np.array(train_data)
	svr_lin = SVR(kernel='linear', C=1e3)#LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
	print 'Training on ', len(X), 'examples'
	model = svr_lin.fit(X, y)
	return model

def test(model):
	with open('RossDev2.csv', 'rb') as dev:
		dev_data = csv.reader(dev)
		dev_data = list(dev_data)
		dev_data = dev_data[1:]

	gold = []
	for i in range(len(dev_data)):
		gold.append(float_if_possible(dev_data[i][3]))
		dev_data[i] = [float_if_possible(dev_data[i][0]), float_if_possible(dev_data[i][1]),float_if_possible(dev_data[i][4]), float_if_possible(dev_data[i][6]), float_if_possible(dev_data[i][7]), float_if_possible(dev_data[i][8])]

	X = np.array(dev_data)
	print 'Predicting'
	predictions = model.predict(X)

	square_error = 0
	print predictions
	for i,p in enumerate(predictions):
		square_error += ((p - gold[i])**2)/len(gold)
	print 'Square error', square_error

model = train()
test(model)