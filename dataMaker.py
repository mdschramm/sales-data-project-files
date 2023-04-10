import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import xlwt
import random
training_size = 60000#65536
test_size = 1000
dev_size = 1000
def get_data():
	with open('C:\Users\David\Desktop\proj\proj\progress\\train.csv', 'rb') as train:
			data = csv.reader(train)
			data = list(data)
			header_row = data[0]


	indices = {}
	indices['customers'] = header_row.index('Customers')
	indices['sales'] = header_row.index('Sales')
	indices['dow'] = header_row.index('DayOfWeek')
	indices['promo'] = header_row.index('Promo')
	indices['schoolHoli'] = header_row.index('SchoolHoliday')
	indices['store'] = header_row.index('Store')

	return indices, data
headers = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'State Holiday', 'School Holiday']
indices, data = get_data()

sampled = [data[i] for i in (random.sample(xrange(len(data)), training_size + test_size + dev_size))]
train_data = sampled[:training_size]
dev_data = sampled[training_size:training_size + dev_size]
test_data = sampled[training_size+dev_size:training_size+dev_size+dev_size]

wb = xlwt.Workbook()
ts = wb.add_sheet('train')
ds = wb.add_sheet('dev')
tts = wb.add_sheet('test')

def float_if_possible(strg):
    try:
        return float(strg)
    except ValueError:
        return strg

for x in xrange(len(headers)):
    ts.write(0, x, headers[x])
    ds.write(0, x, headers[x])
    tts.write(0, x, headers[x])
for y in xrange(len(train_data)):
    arr = train_data[y]
    for x in xrange(len(arr)):
        ts.write(y + 1, x, float_if_possible(arr[x]))
for y in xrange(len(dev_data)):
    arr = train_data[y]
    for x in xrange(len(arr)):
        ds.write(y + 1, x, float_if_possible(arr[x]))
for y in xrange(len(test_data)):
    arr = train_data[y]
    for x in xrange(len(arr)):
        tts.write(y + 1, x, float_if_possible(arr[x]))        
wb.save('RossData.xls')