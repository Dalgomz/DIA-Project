import numpy as np

# Fixed Asignement of promos
# Fixed Price of Item 2
# Knows customers per class
# Known conversion rate of item 2
# UCB and Thompson sapmpling to compare porformance
# Maximize the revenue by optimizing price of item 1

# Modify for receive dual probabilities (buy 2 items) and number of costumers per day 

class Environment():
	# Number of Customer Types, Number Of Customers, Conversion Rate 1, Conversion rate 2,
	# TO DO. where do the promo Assignment go?
	def __init__(self, nArms, customers, probabilities1, probabilities2):
		self.nArms = nArms
		self.customers = customers
		self.probabilities1 = probabilities1
		self.probabilities2 = probabilities2

	def round(self, ccn, pulledArm):
		# binomial, (How many, probability)
		# pulled arm -> indice del mejor brazo (Each arm is a price)
		# reward = How many clients made a purchase
		reward = np.random.binomial(self.customers[ccn], self.probabilities1[ccn][pulledArm])
		if (reward > 0):
			reward2 = np.random.binomial(reward, self.probabilities2[ccn][pulledArm])
		else:
			reward2 = 0
		return reward, reward2
