import csv
import numpy as np
from sklearn.svm import SVR
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
import sys
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
def train():
	with open('C:\Users\David\Desktop\proj\proj\progress\\RossTrain4D.csv', 'rb') as train:
		train_data = csv.reader(train)
		train_data = list(train_data)
		train_data = train_data[1:]
		
		y = []

		# remove date, open, and sales columns
		processed_data = []
		# remove date, open, and sales columns
		columns = [0,1,2,4,6,7,8]
		for i in range(len(train_data)):
			if train_data[i][5] != "0":
				processed_data.append(cut_columns(columns, train_data[i]))
				y.append(float_if_possible(train_data[i][3]))
	X = np.array(processed_data)
	#print X
	#X_scaled = sklearn.preprocessing.scale(X)
	reggie = sklearn.ensemble.GradientBoostingRegressor('ls')
	print 'Training on ', len(X), 'examples'
	#print X_scaled
	#print X
	#sys.exit(0)
	reggie = reggie.fit(X, y)
	return reggie

def test(model, filename):
	with open(filename, 'rb') as dev:
		dev_data = csv.reader(dev)
		dev_data = list(dev_data)
		dev_data = dev_data[1:]
        y = []
	gold = []
	columns = [0,1,2,4,6,7,8]
	zeroGold = 0
	processed_dev = []
	for i in range(len(dev_data)):
		if dev_data[i][5] != "0":
			processed_dev.append(cut_columns(columns, dev_data[i]))
			y.append(float_if_possible(dev_data[i][3]))
			gold.append(float_if_possible(dev_data[i][3]))
	        else:
	               zeroGold +=1
	X = np.array(processed_dev)
	print 'Predicting on ' , len(X), 'examples'
	predictions = model.predict(X)
	square_error = 0
	abs_error = 0
	#print predictions
	#print gold
	for i,p in enumerate(predictions):
		square_error += ((p - gold[i])**2)/(len(gold) + zeroGold)
		abs_error += (abs(p-gold[i]))/(len(gold) + zeroGold)
	print 'Square error', square_error
	print 'abs error', abs_error
	
startTime = time.time()
model = train()
test(model, 'C:\Users\David\Desktop\proj\proj\progress\\RossDev4D.csv') 
test(model, 'C:\Users\David\Desktop\proj\proj\progress\\RossTest4D.csv')
endTime = time.time()
time1 = endTime - startTime
print time1