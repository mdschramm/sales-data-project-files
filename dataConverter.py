import csv
import xlwt

mappy = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
def train():
    wb = xlwt.Workbook()
    ts = wb.add_sheet('train')
    with open('C:\Users\David\Desktop\proj\proj\progress\\RossTest4.csv', 'rb') as train:
		train_data = csv.reader(train)
		train_data = list(train_data)
		train_data = train_data[1:]
		for i in xrange(len(train_data)):
		      string = train_data[i][2].split("-")
		      val = mappy[int(string[1]) - 1] + int(string[2])
		      ts.write(i, 0, int(val))
	        wb.save('Dates3.xls')
train()		    