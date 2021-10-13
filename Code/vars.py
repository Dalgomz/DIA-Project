
customers = [45,30,20,5]
maxCustomers = 0
for c in customers:
	maxCustomers += c

# cvRate[ccn][i] for ccv (Customer Class Number) conv rate at price[i]
cvRate1 = [[0.15,0.14,0.13,0.12],[0.34,0.33,0.32,0.31],[0.2,0.19,0.18,0.17],[0.72,0.71,0.70,0.69]]
cvRate2 = [[0.15,0.15,0.15,0.15],[0.34,0.34,0.34,0.34],[0.2,0.2,0.2,0.2],[0.72,0.72,0.72,0.72]]
price1 = [40,45,50,55]
basePrice1 = 30
price2 = [20,24,28,32]
basePrice2 = 10
promos = [0,-3,-6,-9]
promDist = []

def generateCvRate2():
	return [[0.15,0.15,0.15,0.15],[0.34,0.34,0.34,0.34],[0.2,0.2,0.2,0.2],[0.72,0.72,0.72,0.72]]
