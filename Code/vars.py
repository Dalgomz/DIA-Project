from numpy import random

customers = [45,30,20,5]
maxCustomers = 0
for c in customers:
	maxCustomers += c

experiments = 1
# cvRate[ccn][i] for ccv (Customer Class Number) conv rate at price[i]
cvRate1 = [[0.15,0.11,0.09,0.06],[0.34,0.32,0.30,0.28],[0.2,0.18,0.13,0.1],[0.72,0.71,0.70,0.69]]
cvRate2 = [[0.15,0.15,0.15,0.15],[0.34,0.34,0.34,0.34],[0.2,0.2,0.2,0.2],[0.72,0.72,0.72,0.72]]
# Add seasons CV maker
days = 365
seasons = [days//4,(days//4)*2,(days//4)*3,days]
price1 = [40,45,50,55]
basePrice1 = 30
price2 = [20,24,28,32]
basePrice2 = 10
promos = [0,-3,-6,-9]
promDist = [40,30,20,10]

def generateCvRate2():
	# Generate random conv rate for customers near a function shape
	array = []
	cv2 = []
	val = random.randint(10, 20)
	for i in range(4):
		array.append((val-(i*2))/100)
	cv2.append(array)
	array = []
	val = random.randint(29, 39)
	for i in range(4):
		array.append((val-i*5)/100)
	cv2.append(array)
	array = []
	val = random.randint(15, 25)
	for i in range(4):
		array.append((val-(i*2))/100)
	cv2.append(array)
	array = []
	val = random.randint(67, 77)
	for i in range(4):
		array.append((val-(i*1))/100)
	cv2.append(array)
	return cv2 # shape [[0.15,0.15,0.15,0.15],[0.34,0.34,0.34,0.34],[0.2,0.2,0.2,0.2],[0.72,0.72,0.72,0.72]]

def cvRate1Seasons():
	seasonMarks = [[10, 29, 15, 67, 10], [7, 18, 10, 67, 4], [10, 29, 18, 67, 10], [16, 70, 40, 75, 8]]
	cvRate1 = []
	for i in range(len(seasonMarks)):
		cvRate1.append([])
		for j in range(4):
			val = random.randint(seasonMarks[i][j],seasonMarks[i][j]+seasonMarks[i][4])
			cvRate1[i].append([])
			for k in range(4):
				cvRate1[i][j].append((val-k)/100)

	return cvRate1

def cvRate2Seasons():
	seasonMarks = [[10, 29, 15, 67, 10], [7, 18, 10, 67, 4], [10, 29, 18, 67, 10], [16, 70, 40, 75, 8]]
	cvRate2 = []
	for i in range(len(seasonMarks)):
		cvRate2.append([])
		for j in range(4):
			val = random.randint(seasonMarks[i][j],seasonMarks[i][j]+seasonMarks[i][4])
			cvRate2[i].append([])
			for k in range(4):
				cvRate2[i][j].append((val-k)/100)
	
	return cvRate2