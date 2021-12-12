# Fixed Price of Item 2 (all prices are the same)
# Unknown customers per class (Bernoulli Distrbution)
# Unknown conversion rate of item 2 
# Optimize the Asignement of promos (Hungarian)
# UCB and Thompson sapmpling to compare porformance
# Maximize the revenue by optimizing price of item 1

import hungarianAlgorithm
import numpy as np
import matplotlib.pyplot as plt 
import vars as v
from Environment import Environment
from learners.ThompsonLearner import *
from learners.UCB1Learner import *

np.random.seed(4)

# Definition of variables

totalCustomers = v.maxCustomers
customers = v.customers
convRates1 = v.cvRate1
convRates2 = v.generateCvRate2()
item1Prices = v.price1
item1Cost = v.basePrice1
item2Prices = v.price2
item2Cost = v.basePrice2
discounts = v.promos
promoDistribution = hungarianAlgorithm.productDist(customers, v.promDist, item2Cost, discounts.copy(), [convRates2[0][0]*100,convRates2[1][0]*100,convRates2[2][0]*100,convRates2[3][0]*100])
nArms = len(customers)
maxRevenue = max(item1Prices)+max(item2Prices)-(item1Cost+item2Cost)

# Time in days
T = 365

nExperiments = 1
tsRewardsPerExperiment = []
ucbRewardsPerExperiment = []



# Clarivoyant 
dailyOptimalRewards = [0,0,0,0]
for i in range(len(item1Prices)):
	for j in range(len(customers)):
		# Customer j with Price i for its conversion rate
		dailyOptimalRewards[i] += ((item1Prices[i]-item1Cost)*customers[j]*convRates1[j][i])
		for k in range(len(promoDistribution)):
			# Discount asignation
			if promoDistribution[k][0] == j:
				dailyOptimalRewards[i] += ((item2Prices[0]-item2Cost-discounts[ promoDistribution[k][1] ])*promoDistribution[k][2]*convRates2[j][0])

print(dailyOptimalRewards)
optimalPrice = dailyOptimalRewards.index(max(dailyOptimalRewards))
optimalRewards = [(dailyOptimalRewards[optimalPrice]/(maxRevenue * totalCustomers)) for x in range(T)]
optimalPrice = dailyOptimalRewards[optimalPrice]

for e in range(0,nExperiments):
	#print('\r', "Progress: {}/{} days".format(e, nExperiments), end=" ")
	env = Environment(nArms, customers, convRates1, convRates2)
	tsLearner = ThompsonLearner(nArms)
	ucbLearner = UCBLearner(nArms)
	for t in range(0,T):
		# Available promos number
		dailyAvailablePromos = []
		for k in range(len(promoDistribution)):
			dailyAvailablePromos.append(promoDistribution[k][2])
		# Thompson Learner
		tsPulledArm = tsLearner.pullArm()
		tsReward = 0
		tsSale = 0
		
		# UCB1 Learner
		ucbPulledArm = ucbLearner.pullArm()
		ucbReward = 0
		ucbSale = 0


		# Sale to all daily clients
		for i in range(len(customers)):
			it1, it2 = env.round(i, tsPulledArm)
			buyers = it2
			tsReward += (item1Prices[tsPulledArm] - item1Cost) * it1
			for k in range(len(promoDistribution)):
				# if there are available promos for this type of customer
				if i == promoDistribution[k][0] and dailyAvailablePromos[k] > 0:
					# If more customers than promos
					if  buyers >= dailyAvailablePromos[k]:
						tsReward += (item2Prices[0] - item2Cost) * dailyAvailablePromos[k]
						buyers -= dailyAvailablePromos[k]
						dailyAvailablePromos = 0
					# If less customers than promos, but remaining
					elif buyers > 0:
						tsReward += (item2Prices[0] - item2Cost) * buyers
						dailyAvailablePromos[k] -= buyers
						buyers = 0

			it1, it2 = env.round(i, ucbPulledArm)
			buyers = it2
			ucbReward += (item1Prices[ucbPulledArm] - item1Cost) * it1
			ucbReward += (item2Prices[0] - item2Cost) * it2
			for k in range(len(promoDistribution)):
				# if there are available promos for this type of customer
				if i == promoDistribution[k][0] and dailyAvailablePromos[k] > 0:
					# If more customers than promos
					if  buyers >= dailyAvailablePromos[k]:
						ucbReward += (item2Prices[0] - item2Cost) * dailyAvailablePromos[k]
						buyers -= dailyAvailablePromos[k]
						dailyAvailablePromos = 0
					# If less customers than promos, but remaining
					elif buyers > 0:
						ucbReward += (item2Prices[0] - item2Cost) * buyers
						dailyAvailablePromos[k] -= buyers
						buyers = 0

		rewardTS = tsReward / (maxRevenue * totalCustomers)
		rewardUCB = ucbReward / (maxRevenue * totalCustomers)

		tsLearner.update(tsPulledArm, rewardTS)
		ucbLearner.update(ucbPulledArm, rewardUCB)

	tsRewardsPerExperiment.append(tsLearner.collectedRewards)
	ucbRewardsPerExperiment.append(ucbLearner.collectedRewards)

tsData = np.cumsum(np.mean(tsRewardsPerExperiment,axis=0))
ucbData = np.cumsum(np.mean(ucbRewardsPerExperiment,axis=0))
print("Best Price learnt by Thompson Sampling:",item1Prices[np.argmax([len(a) for a in tsLearner.rewardsPerArm])],"$")
print("Best price learnt bu UCB1:",item1Prices[np.argmax([len(a) for a in ucbLearner.rewardsPerArm])],"$")

plt.figure(0)
plt.plot(tsData, label='Thomson Sampling', color='tab:blue')
plt.plot(ucbData, label='UCB1', color='tab:green')
plt.plot(np.cumsum(optimalRewards), label='Carivoyant', color='tab:red')
plt.legend(loc='lower right')
plt.grid(linestyle='--')
plt.xlabel('Days')
plt.ylabel('Rewards')
plt.title('Cumulative Reward collected by both learners')
plt.show()

# Daily reward

tsData = np.mean(np.multiply(tsRewardsPerExperiment,(maxRevenue * totalCustomers)),axis=0)
ucbData = np.mean(np.multiply(ucbRewardsPerExperiment,(maxRevenue * totalCustomers)),axis=0)
optData = np.multiply(optimalRewards,(maxRevenue * totalCustomers))
def moving_average(x, w):
	return np.convolve(x, np.ones(w), 'valid') / w

plt.figure(0)
plt.plot(moving_average(tsData, 5), label='Thomson Sampling', color='tab:blue')
plt.plot(moving_average(ucbData, 5), label='UCB1', color='tab:green')
plt.plot(optData, label='Carivoyant', color='tab:red')
plt.legend(loc='lower right')
plt.grid(linestyle='--')
plt.xlabel('Days')
plt.ylabel('Revenue')
plt.title('Daily Reward learnt by both learners')
plt.show()

# Regret
plt.figure(0)
plt.plot(np.cumsum(np.array(optData) - np.array(tsData)), label='Thomson Sampling', color='tab:blue')
plt.plot(np.cumsum(np.array(optData) - np.array(ucbData)), label='UCB1', color='tab:green')
plt.legend(loc='lower right')
plt.grid(linestyle='--')
plt.xlabel('Days')
plt.ylabel('Regret')
plt.title('Regret of both learners')
plt.show()