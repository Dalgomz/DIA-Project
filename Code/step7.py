# Unknown customers per class (Bernoulli Distrbution)
# Unknown conversion rates of items
# Seasons included
# Sliding Window and Thompson sampling to compare performance on 
# Same environment as step 6

import hungarianAlgorithm
import numpy as np
import matplotlib.pyplot as plt 
import vars as v
from Environment import Environment
from NonStaticEnvironment import NonStaticEnvironment
from learners.ThompsonLearner import *
from learners.UCB1Learner import *
from learners.TSlidingWindow import *

np.random.seed(7)

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
nArms = len(customers)
maxRevenue = max(item1Prices)+max(item2Prices)-(item1Cost+item2Cost)
seasonsBreak = v.seasons
seasonDuration = v.seasons[0]
seasonRates1 = v.cvRate1Seasons()
seasonRates2 = v.cvRate2Seasons()
promoDistributionSeason = [
	hungarianAlgorithm.productPriceDist(customers, v.promDist, v.price2, discounts.copy(), v.sortingCv(seasonRates1, 0)),
	hungarianAlgorithm.productPriceDist(customers, v.promDist, v.price2, discounts.copy(), v.sortingCv(seasonRates1, 1)),
	hungarianAlgorithm.productPriceDist(customers, v.promDist, v.price2, discounts.copy(), v.sortingCv(seasonRates1, 2)),
	hungarianAlgorithm.productPriceDist(customers, v.promDist, v.price2, discounts.copy(), v.sortingCv(seasonRates1, 3))] # One per season
promoDistribution = promoDistributionSeason[0]
# Time in days
T = 365

nExperiments = v.experiments
tsRewardsPerExperiment = []
ucbRewardsPerExperiment = []
swRewardsPerExperiment = []
bp2 = [0,0,0,0]

# Clarivoyant 
# Do one predict per season
dailyOptimalRewards = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
for season in range(len(seasonsBreak)):
	convRates1=seasonRates1[season]
	convRates2=seasonRates2[season]
	for i in range(len(item1Prices)):
		for j in range(len(customers)):
			# Customer j with Price i for its conversion rate
			dailyOptimalRewards[season][i] += ((item1Prices[i]-item1Cost)*customers[j]*convRates1[j][i])
			prof2 = [item2Prices[0] *convRates1[j][i] * convRates2[j][0], item2Prices[1] *convRates1[j][i] * convRates2[j][1], item2Prices[2] *convRates1[j][i] * convRates2[j][2], item2Prices[3] *convRates1[j][i] * convRates2[j][3]]
			bp2[season] = prof2.index(max(prof2))
			for k in range(len(promoDistribution)):
				# Discount asignation
				if promoDistribution[k][0] == j:
					dailyOptimalRewards[season][i] += ((item2Prices[bp2[season]]-item2Cost-discounts[ promoDistribution[k][1] ])*promoDistribution[k][2]*convRates1[j][i]*convRates2[j][bp2[season]])

optimalRewards = []
lastSeason = 0
for season in range(len(seasonsBreak)):
	optimalPrice = dailyOptimalRewards[season].index(max(dailyOptimalRewards[season]))
	optimalSeasonRewards = [(dailyOptimalRewards[season][optimalPrice]/(maxRevenue * totalCustomers)) for x in range(seasonsBreak[season]-lastSeason)]
	optimalRewards = optimalRewards + optimalSeasonRewards
	optimalPrice = dailyOptimalRewards[season][optimalPrice]
	lastSeason = seasonsBreak[season]

convRates1 = seasonRates1[0]
convRates2 = seasonRates2[0]
season = 0
for e in range(0,nExperiments):
	#print('\r', "Progress: {}/{} days".format(e, nExperiments), end=" ")
	env = NonStaticEnvironment(nArms, customers, seasonRates1, seasonRates2, T)
	tsLearner = ThompsonLearner(nArms)
	tsLearner2 = ThompsonLearner(nArms)
	swLearner = TSlidingWindow(nArms, seasonDuration)
	swLearner2 = TSlidingWindow(nArms, seasonDuration)
	ucbLearner = UCBLearner(nArms)
	ucbLearner2 = UCBLearner(nArms)
	for t in range(0,T):
		if t in seasonsBreak:
			# Reset learners
			season += 1
			promoDistribution = promoDistributionSeason[season]
			# Change conversion rates

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

		# Sliding Window
		swPulledArm = swLearner.pullArm()
		swPulledArm2 = swLearner2.pullArm()
		swReward = 0
		swReward2 = 0

		# Sale to all daily clients classes
		for i in range(len(customers)):
			it1, it2 = env.round(i, tsPulledArm, tsPulledArm2)
			buyers = it2
			tsReward += (item1Prices[tsPulledArm] - item1Cost) * it1
			# Applying promos 
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

			it1, it2 = env.round(i, ucbPulledArm, ucbPulledArm2)
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
						
			it1, it2 = env.round(i, swPulledArm, swPulledArm2)
			buyers = it2
			swReward += (item1Prices[swPulledArm] - item1Cost) * it1
			# Applying promos 
			for k in range(len(promoDistribution)):
				# if there are available promos for this type of customer
				if i == promoDistribution[k][0] and dailyAvailablePromos[k] > 0:
					# If more customers than promos
					if  buyers >= dailyAvailablePromos[k]:
						swReward2 += (item2Prices[swPulledArm2] - item2Cost) * dailyAvailablePromos[k]
						buyers -= dailyAvailablePromos[k]
						dailyAvailablePromos[k] = 0
					# If less customers than promos, but remaining
					elif buyers > 0:
						swReward2 += (item2Prices[swPulledArm2] - item2Cost) * buyers
						dailyAvailablePromos[k] -= buyers
						buyers = 0
		env.nextDay()

		rewardTS = tsReward / (maxRevenue * totalCustomers)
		rewardTS2 = tsReward2 / (maxRevenue * totalCustomers)
		rewardUCB = ucbReward / (maxRevenue * totalCustomers)
		rewardUCB2 = ucbReward2 / (maxRevenue * totalCustomers)
		rewardSW = swReward / (maxRevenue * totalCustomers)
		rewardSW2 = swReward2 / (maxRevenue * totalCustomers)

		tsLearner.update(tsPulledArm, rewardTS)
		tsLearner2.update(tsPulledArm2, rewardTS2)
		ucbLearner.update(ucbPulledArm, rewardUCB)
		ucbLearner2.update(ucbPulledArm2, rewardUCB2)
		swLearner.update(swPulledArm, rewardSW)
		swLearner2.update(swPulledArm2, rewardSW2)

	tsCollectedRewards = tsLearner.collectedRewards + tsLearner2.collectedRewards
	ucbCollectedRewards = ucbLearner.collectedRewards + ucbLearner2.collectedRewards
	swCollectedRewards = swLearner.collectedRewards + swLearner2.collectedRewards

	tsRewardsPerExperiment.append(tsCollectedRewards)
	ucbRewardsPerExperiment.append(ucbCollectedRewards)
	swRewardsPerExperiment.append(swCollectedRewards)

tsData = np.cumsum(np.mean(tsRewardsPerExperiment,axis=0))
ucbData = np.cumsum(np.mean(ucbRewardsPerExperiment,axis=0))
swData = np.cumsum(np.mean(swRewardsPerExperiment,axis=0))

print("Best Price learnt by Thompson Sampling: \nItem 1:",item1Prices[np.argmax([len(a) for a in tsLearner.rewardsPerArm])],"$")
print("Item 2:",item2Prices[np.argmax([len(a) for a in tsLearner2.rewardsPerArm])],"$")
print("Best price learnt by UCB1: \nItem 1:",item1Prices[np.argmax([len(a) for a in ucbLearner.rewardsPerArm])],"$")
print("Item 2:",item2Prices[np.argmax([len(a) for a in ucbLearner2.rewardsPerArm])],"$")
print("Best price learnt by Sliding Window: \nItem 1:",item1Prices[np.argmax([len(a) for a in swLearner.rewardsPerArm])],"$")
print("Item 2:",item2Prices[np.argmax([len(a) for a in swLearner2.rewardsPerArm])],"$")

print("Profit:")
print("Thompson Sampling:",swData[-1],"$")
print("UCB:",ucbData[-1],"$")
print("Sliding Window:",tsData[-1],"$")

plt.figure(figsize=(14, 5))
plt.plot(swData, label='Thomson Sampling', color='tab:blue')
plt.plot(ucbData, label='UCB1', color='tab:green')
plt.plot(tsData, label='Sliding Window', color='tab:orange')
plt.plot(np.cumsum(optimalRewards), label='Carivoyant', color='tab:red')
plt.legend(loc='lower right')
plt.grid(linestyle='--')
plt.xlabel('Days')
plt.ylabel('Rewards')
plt.title('Cumulative Reward collected by both learners')
plt.savefig('img/s7-1.png')
# plot.show()

# Daily reward

tsData = np.mean(np.multiply(tsRewardsPerExperiment,(maxRevenue * totalCustomers)),axis=0)
ucbData = np.mean(np.multiply(ucbRewardsPerExperiment,(maxRevenue * totalCustomers)),axis=0)
swData = np.mean(np.multiply(swRewardsPerExperiment,(maxRevenue * totalCustomers)),axis=0)
optData = np.multiply(optimalRewards,(maxRevenue * totalCustomers))
def moving_average(x, w):
	return np.convolve(x, np.ones(w), 'valid') / w

plt.figure(figsize=(14, 5))
plt.plot(moving_average(swData, 10), label='Thompson Sampling', color='tab:blue')
plt.plot(moving_average(ucbData, 10), label='UCB1', color='tab:green')
plt.plot(moving_average(tsData, 10), label='Sliding Window', color='tab:orange')
plt.plot(optData, label='Carivoyant', color='tab:red')
plt.legend(loc='lower right')
plt.grid(linestyle='--')
plt.xlabel('Days')
plt.ylabel('Revenue')
plt.title('Daily Reward learnt by both learners')
plt.savefig('img/s7-2.png')
# plot.show()


print("Regret:")
print("Thompson Sampling:",(np.array(optData) - np.array(swData))[-1],"$")
print("UCB:",(np.array(optData) - np.array(ucbData))[-1],"$")
print("Sliding Window:",(np.array(optData) - np.array(tsData))[-1],"$")
# Regret
plt.figure(figsize=(14, 5))
plt.plot(np.cumsum(np.array(optData) - np.array(swData)), label='Thomson Sampling', color='tab:blue')
plt.plot(np.cumsum(np.array(optData) - np.array(ucbData)), label='UCB1', color='tab:green')
plt.plot(np.cumsum(np.array(optData) - np.array(tsData)), label='Sliding Window', color='tab:orange')
plt.legend(loc='lower right')
plt.grid(linestyle='--')
plt.xlabel('Days')
plt.ylabel('Regret')
plt.title('Regret of both learners')
plt.savefig('img/s7-3.png')
# plot.show()