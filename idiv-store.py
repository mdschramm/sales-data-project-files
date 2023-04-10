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
	with open('storeD.csv', 'rb') as store:
		store_data = csv.reader(store)
		store_data = list(store_data)
		store_data = store_data[1:]

	arr = []
	for row in store_data:
		arr.append(row[1:4])
	return arr

def train(store_data):
	with open('RossTrain4D.csv', 'rb') as train:
		train_data = csv.reader(train)
		train_data = list(train_data)
		train_data = train_data[1:]
		
		y = []
		dev_y = []
		# remove date, open, and sales columns
		processed_data = []
		processed_dev = []
		# remove date, open, and sales columns
		columns = [0, 1, 2, 6, 7, 8]
		zeroGold = 0
  		for i in range(len(train_data)):
  			r = random.random()
 			if train_data[i][5] != "0":
				if r < .8:
					processed_data.append(cut_columns(columns, train_data[i]) + store_data[int(train_data[i][0]) - 1])
    				y.append(float_if_possible(train_data[i][3]))
    			else:
    				processed_dev.append(cut_columns(columns, train_data[i]) + store_data[int(train_data[i][0]) - 1])
    				dev_y.append(float_if_possible(train_data[i][3]))
    		elif r > .8:
	    		zeroGold += 1
	    	else:
	    		pass
    	for s in range(1114):
		    X = []
		    Y = []
		    for _ in xrange(len(processed_data)):
		        row = processed_data[_]
		        if row[0] == s+1:
		                X.append(row)
		                Y.append(y[_])
		    if len(X) == 0:
		           continue
		    X = np.array(X)
	       	    #print X
	       	    #X_scaled = sklearn.preprocessing.scale(X)
	       	    reggie = sklearn.ensemble.GradientBoostingRegressor('ls')
	       	    #svr_lin = SVR(kernel='linear', C=1e0)
	       	    print 'Training on ', len(X), 'examples s=', s+1
	       	    #print X_scaled
	       	    #print X
	       	    #sys.exit(0)
	       	    #model = svr_lin.fit(X, y)
	       	    reggie = reggie.fit(X, Y)
	       	    modelMap[s+1] = reggie
	       	    #return reggie
	    return processed_dev, dev_y, zeroGold

def test(processed_dev, dev_y, zeroGold):
 #        square_error = 0
 #        abs_error = 0
 #        percent_error = 0
	# with open(filename, 'rb') as dev:
	# 	dev_data = csv.reader(dev)
	# 	dev_data = list(dev_data)
	# 	dev_data = dev_data[1:]
 #        y = []
	# gold = []
	# columns = [0,1,2,6,7,8]
	# zeroGold = 0
	# processed_dev = []
	# for i in range(len(dev_data)):
	# 	if dev_data[i][5] != "0":
	# 		processed_dev.append(cut_columns(columns, dev_data[i]) + store_data[int(dev_data[i][0]) - 1])
	# 		y.append(float_if_possible(dev_data[i][3]))
	# 		gold.append(float_if_possible(dev_data[i][3]))
	# 		if dev_data[i][3] == 0:
	# 		    print i
	#         else:
	#                zeroGold +=1
	X = np.array(processed_dev)
	for s in range(1114):
	    X = []
	    Y = []
	    for _ in xrange(len(processed_dev)):
	        row = processed_dev[_]
	        if row[0] == s+1:
	                X.append(row)
	                Y.append(dev_y[_])
	    if len(X) == 0:
	           continue
	    X = np.array(X)
	    print 'Predicting on ' , len(X), 'examples'
	    model = modelMap.get(s+1)
	    predictions = model.predict(X)
	    for i,p in enumerate(predictions):
	  	square_error += ((p - gold[i])**2)/(len(gold) + zeroGold)
		abs_error += (abs(p-gold[i]))/(len(gold) + zeroGold)
		if gold[i] != 0:
		  percent_error += (abs(p-gold[i])/gold[i])/(len(gold) + zeroGold)
	print 'Square error', square_error
	print 'abs error', abs_error
	print 'percent error', percent_error
	print 'rms', math.sqrt(square_error)
modelMap = {}	
startTime = time.time()
store_data = makeStoreMap()
processed_dev, dev_y, zeroGold = train(store_data)
test(processed_dev, dev_y, zeroGold) 
#test(store_data, 'C:\Users\David\Desktop\proj\proj\progress\\RossTest4D.csv')
endTime = time.time()
time1 = endTime - startTime
print time1