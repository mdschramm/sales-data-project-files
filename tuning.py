import csv
import numpy as np
from sklearn.svm import SVR
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
import sys
import sklearn.ensemble
import math
from sklearn import grid_search

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

def makeStoreMap():
	with open('C:\Users\David\Desktop\proj\proj\progress\\storeD.csv', 'rb') as store:
		store_data = csv.reader(store)
		store_data = list(store_data)
		store_data = store_data[1:]

	arr = []
	for row in store_data:
		arr.append(row[1:4])
	return arr

def train(store_data):
	with open('C:\Users\David\Desktop\proj\proj\progress\\RossTrain4D.csv', 'rb') as train:
		train_data = csv.reader(train)
		train_data = list(train_data)
		train_data = train_data[1:]
		
		y = []

		# remove date, open, and sales columns
		processed_data = []
		# remove date, open, and sales columns
		columns = [0,1,2,6,7,8]
		for i in range(len(train_data)):
			if train_data[i][5] != "0":
				processed_data.append(cut_columns(columns, train_data[i]) + store_data[int(train_data[i][0]) - 1])
				y.append(float_if_possible(train_data[i][3]))
	X = np.array(processed_data)
	#print X
	#X_scaled = sklearn.preprocessing.scale(X)
	reggie = sklearn.ensemble.GradientBoostingRegressor('ls', learning_rate = .9, n_estimators = 350)
	#print reggie.get_params()
	parameters = {'learning_rate':[.5, .7, .9], 'n_estimators': [250, 300, 350]}#, 'alpha': [.7, .9, .5], 'min_samples_leaf': [1, 2, 4], 'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 4]}
        #[.1, .2, .4, .6, .9, .01, .05, .07]}
	clf = grid_search.GridSearchCV(reggie, parameters)
	clf.fit(X, y)
	print clf.best_params_
	sys.exit(0)
	#svr_lin = SVR(kernel='linear', C=1e0)
	print 'Training on ', len(X), 'examples'
	#print X_scaled
	#print X
	#sys.exit(0)
	#model = svr_lin.fit(X, y)
	reggie = reggie.fit(X, y)
	return reggie

def test(model, store_data, filename):
	with open(filename, 'rb') as dev:
		dev_data = csv.reader(dev)
		dev_data = list(dev_data)
		dev_data = dev_data[1:]
        y = []
	gold = []
	columns = [0,1,2,6,7,8]
	zeroGold = 0
	processed_dev = []
	for i in range(len(dev_data)):
		if dev_data[i][5] != "0":
			processed_dev.append(cut_columns(columns, dev_data[i]) + store_data[int(dev_data[i][0]) - 1])
			y.append(float_if_possible(dev_data[i][3]))
			gold.append(float_if_possible(dev_data[i][3]))
			if dev_data[i][3] == 0:
			    print i
	        else:
	               zeroGold +=1
	X = np.array(processed_dev)
	print 'Predicting on ' , len(X), 'examples'
	predictions = model.predict(X)
	square_error = 0
	abs_error = 0
	percent_error = 0
	#print predictions
	#print gold
	for i,p in enumerate(predictions):
		square_error += ((p - gold[i])**2)/(len(gold) + zeroGold)
		abs_error += (abs(p-gold[i]))/(len(gold) + zeroGold)
		if gold[i] != 0:
		  percent_error += (abs(p-gold[i])/gold[i])/(len(gold) + zeroGold)
	print 'Square error', square_error
	print 'abs error', abs_error
	print 'percent error', percent_error
	print 'rms', math.sqrt(square_error)
	
startTime = time.time()
store_data = makeStoreMap()
model = train(store_data)
test(model,store_data, 'C:\Users\David\Desktop\proj\proj\progress\\RossDev4D.csv') 
test(model,store_data, 'C:\Users\David\Desktop\proj\proj\progress\\RossTest4D.csv')
endTime = time.time()
time1 = endTime - startTime
print time1