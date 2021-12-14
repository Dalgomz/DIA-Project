from numpy import random

customers = [45,30,20,5]
maxCustomers = 0
for c in customers:
	maxCustomers += c

experiments = 5
# cvRate[ccn][i] for ccv (Customer Class Number) conv rate at price[i]
cvRate1 = [[0.15,0.11,0.09,0.06],[0.34,0.32,0.30,0.28],[0.2,0.18,0.13,0.1],[0.72,0.71,0.70,0.69]]
cvRate2 = [[0.15,0.15,0.15,0.15],[0.34,0.34,0.34,0.34],[0.2,0.2,0.2,0.2],[0.72,0.72,0.72,0.72]]
# Add seasons CV maker
days = 365
seasons = [days//4,(days//4)*2,(days//4)*3]
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
		array.append((val-(i*2))*0.01)
	cv2.append(array)
	array = []
	val = random.randint(29, 39)
	for i in range(4):
		array.append((val-i*5)*0.01)
	cv2.append(array)
	array = []
	val = random.randint(15, 25)
	for i in range(4):
		array.append((val-(i*2))*0.01)
	cv2.append(array)
	array = []
	val = random.randint(67, 77)
	for i in range(4):
		array.append((val-(i*1))*0.01)
	cv2.append(array)
	return cv2 #[[0.15,0.15,0.15,0.15],[0.34,0.34,0.34,0.34],[0.2,0.2,0.2,0.2],[0.72,0.72,0.72,0.72]]

def cvRate1Seasons(seasons):
	print(len(seasons))
	cvRate1 = [[0.15,0.11,0.09,0.06],[0.34,0.32,0.30,0.28],[0.2,0.18,0.13,0.1],[0.72,0.71,0.70,0.69]]
	return cvRate1

def cvRate2Seasons(seasons):
	print(len(seasons))
	cvRate1 = [[0.15,0.11,0.09,0.06],[0.34,0.32,0.30,0.28],[0.2,0.18,0.13,0.1],[0.72,0.71,0.70,0.69]]
	return generateCvRate2