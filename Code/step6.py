# Unknown customers per class (Bernoulli Distrbution)
# Unknown conversion rates of items 

#*// Use function to get these*
# Optimize Asignement of promos
# Optimize Asignement of prices
# UCB and Thompson sapmpling to compare porformance
# Maximize the revenue by optimizing price of item 1

import hungarianAlgorithm
import numpy as np
import matplotlib.pyplot as plt 
import vars as v
from Environment import Environment
from learners.ThompsonLearner import *
from learners.UCB1Learner import *

np.random.seed(6)

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
cvr2Sort = [[convRates2[0][0]*100,convRates2[1][0]*100,convRates2[2][0]*100,convRates2[3][0]*100],
			[convRates2[0][1]*100,convRates2[1][1]*100,convRates2[2][1]*100,convRates2[3][1]*100],
			[convRates2[0][2]*100,convRates2[1][2]*100,convRates2[2][2]*100,convRates2[3][2]*100],
			[convRates2[0][3]*100,convRates2[1][3]*100,convRates2[2][3]*100,convRates2[3][3]*100]]
promoDistribution = hungarianAlgorithm.productPriceDist(customers, v.promDist, v.price2, discounts.copy(), cvr2Sort)
nArms = len(customers)
maxRevenue = max(item1Prices)+max(item2Prices)-(item1Cost+item2Cost)

# Time in days
T = 365

nExperiments = v.experiments
tsRewardsPerExperiment = []
ucbRewardsPerExperiment = []

# Clarivoyant 
dailyOptimalRewards = [0,0,0,0]
for i in range(len(item1Prices)):
	for j in range(len(customers)):
		# Customer j with Price i for its conversion rate
		dailyOptimalRewards[i] += ((item1Prices[i]-item1Cost)*customers[j]*convRates1[j][i])
		prof2 = [item2Prices[0] *convRates1[j][i] * convRates2[j][0], item2Prices[1] *convRates1[j][i] * convRates2[j][1], item2Prices[2] *convRates1[j][i] * convRates2[j][2], item2Prices[3] *convRates1[j][i] * convRates2[j][3]]
		bp2 = prof2.index(max(prof2))
		for k in range(len(promoDistribution)):
			# Discount asignation
			if promoDistribution[k][0] == j:
				dailyOptimalRewards[i] += ((item2Prices[bp2]-item2Cost-discounts[ promoDistribution[k][1] ])*promoDistribution[k][2]*convRates1[j][i]*convRates2[j][bp2])

optimalPrice = dailyOptimalRewards.index(max(dailyOptimalRewards))
optimalRewards = [(dailyOptimalRewards[optimalPrice]/(maxRevenue * totalCustomers)) for x in range(T)]
optimalPrice = dailyOptimalRewards[optimalPrice]

for e in range(0,nExperiments):
	#print('\r', "Progress: {}/{} days".format(e, nExperiments), end=" ")
	env = Environment(nArms, customers, convRates1, convRates2)
	tsLearner = ThompsonLearner(nArms)
	tsLearner2 = ThompsonLearner(nArms)
	ucbLearner = UCBLearner(nArms)
	ucbLearner2 = UCBLearner(nArms)
	for t in range(0,T):
		# Available promos number
		dailyAvailablePromos = []
		for k in range(len(promoDistribution)):
			dailyAvailablePromos.append(promoDistribution[k][2])

		# Thompson Learner
		tsPulledArm = tsLearner.pullArm()
		tsPulledArm2 = tsLearner2.pullArm()
		tsReward = 0
		tsReward2 = 0
		
		
		# UCB1 Learner
		ucbPulledArm = ucbLearner.pullArm()
		ucbPulledArm2 = ucbLearner2.pullArm()
		ucbReward = 0
		ucbReward2 = 0


		# Sale to all daily clients
		for i in range(len(customers)):
			it1, it2 = env.round2(i, tsPulledArm, tsPulledArm2)
			buyers = it2
			tsReward += (item1Prices[tsPulledArm] - item1Cost) * it1
			for k in range(len(promoDistribution)):
				# if there are available promos for this type of customer
				if i == promoDistribution[k][0] and dailyAvailablePromos[k] > 0:
					# If more customers than promos
					if  buyers >= dailyAvailablePromos[k]:
						tsReward2 += (item2Prices[tsPulledArm2] - item2Cost) * dailyAvailablePromos[k]
						buyers -= dailyAvailablePromos[k]
						dailyAvailablePromos[k] = 0
					# If less customers than promos, but remaining
					elif buyers > 0:
						tsReward2 += (item2Prices[tsPulledArm2] - item2Cost) * buyers
						dailyAvailablePromos[k] -= buyers
						buyers = 0

			it1, it2 = env.round2(i, ucbPulledArm, ucbPulledArm2)
			buyers = it2
			ucbReward += (item1Prices[ucbPulledArm] - item1Cost) * it1
			for k in range(len(promoDistribution)):
				# if there are available promos for this type of customer
				if i == promoDistribution[k][0] and dailyAvailablePromos[k] > 0:
					# If more customers than promos
					if  buyers >= dailyAvailablePromos[k]:
						ucbReward2 += (item2Prices[ucbPulledArm2] - item2Cost) * dailyAvailablePromos[k]
						buyers -= dailyAvailablePromos[k]
						dailyAvailablePromos[k] = 0
					# If less customers than promos, but remaining
					elif buyers > 0:
						ucbReward2 += (item2Prices[ucbPulledArm2] - item2Cost) * buyers
						dailyAvailablePromos[k] -= buyers
						buyers = 0

		rewardTS1 = (tsReward) / (maxRevenue * totalCustomers)
		rewardUCB1 = (ucbReward) / (maxRevenue * totalCustomers)
		rewardTS2 = (tsReward2) / (maxRevenue * totalCustomers)
		rewardUCB2 = (ucbReward2) / (maxRevenue * totalCustomers)

		tsLearner.update(tsPulledArm, rewardTS1)
		ucbLearner.update(ucbPulledArm, rewardUCB1)
		tsLearner2.update(tsPulledArm2, rewardTS2)
		ucbLearner2.update(ucbPulledArm2, rewardUCB2)

	tsCollectedRewards = tsLearner.collectedRewards + tsLearner2.collectedRewards
	ucbCollectedRewards = ucbLearner.collectedRewards + ucbLearner2.collectedRewards
	tsRewardsPerExperiment.append(tsCollectedRewards)
	ucbRewardsPerExperiment.append(ucbCollectedRewards)

tsData = np.cumsum(np.mean(tsRewardsPerExperiment,axis=0))
ucbData = np.cumsum(np.mean(ucbRewardsPerExperiment,axis=0))
print("Best prices learnt by Thompson Sampling:\nItem 1: ",item1Prices[np.argmax([len(a) for a in tsLearner.rewardsPerArm])],"$")
print("Item 2: ",item2Prices[np.argmax([len(a) for a in tsLearner2.rewardsPerArm])],"$")
print("Best prices learnt by UCB1: \nItem 1:",item1Prices[np.argmax([len(a) for a in ucbLearner.rewardsPerArm])],"$")
print("Item 2:",item2Prices[np.argmax([len(a) for a in ucbLearner2.rewardsPerArm])],"$")

print("Profit:")
print("Thompson Sampling:",tsData[-1],"$")
print("UCB:",ucbData[-1],"$")

plt.figure(figsize=(14, 5))
plt.plot(tsData, label='Thomson Sampling', color='tab:blue')
plt.plot(ucbData, label='UCB1', color='tab:green')
plt.plot(np.cumsum(optimalRewards), label='Carivoyant', color='tab:red')
plt.legend(loc='lower right')
plt.grid(linestyle='--')
plt.xlabel('Days')
plt.ylabel('Rewards')
plt.title('Cumulative Reward collected by both learners')
plt.savefig('img/s6-1.png')
# plot.show()

# Daily reward

tsData = np.mean(np.multiply(tsRewardsPerExperiment,(maxRevenue * totalCustomers)),axis=0)
ucbData = np.mean(np.multiply(ucbRewardsPerExperiment,(maxRevenue * totalCustomers)),axis=0)
optData = np.multiply(optimalRewards,(maxRevenue * totalCustomers))
def moving_average(x, w):
	return np.convolve(x, np.ones(w), 'valid') / w

plt.figure(figsize=(14, 5))
plt.plot(moving_average(tsData, 5), label='Thomson Sampling', color='tab:blue')
plt.plot(moving_average(ucbData, 5), label='UCB1', color='tab:green')
plt.plot(optData, label='Carivoyant', color='tab:red')
plt.legend(loc='lower right')
plt.grid(linestyle='--')
plt.xlabel('Days')
plt.ylabel('Revenue')
plt.title('Daily Reward learnt by both learners')
plt.savefig('img/s6-2.png')
# plot.show()

print("Regret:")
print("Thompson Sampling:",(np.array(optData) - np.array(tsData))[-1],"$")
print("UCB:",(np.array(optData) - np.array(ucbData))[-1],"$")
# Regret
plt.figure(figsize=(14, 5))
plt.plot(np.cumsum(np.array(optData) - np.array(tsData)), label='Thomson Sampling', color='tab:blue')
plt.plot(np.cumsum(np.array(optData) - np.array(ucbData)), label='UCB1', color='tab:green')
plt.legend(loc='lower right')
plt.grid(linestyle='--')
plt.xlabel('Days')
plt.ylabel('Regret')
plt.title('Regret of both learners')
plt.savefig('img/s6-3.png')
# plot.show()