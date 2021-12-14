from Environment import Environment
import numpy as np

# Fixed Asignement of promos
# Fixed Price of Item 2
# Knows customers per class
# Known conversion rate of item 2
# UCB and Thompson sapmpling to compare porformance
# Maximize the revenue by optimizing price of item 1

# Modify for receive dual probabilities (buy 2 items) and number of costumers per day 

class NonStaticEnvironment(Environment):
	# Number of Customer Types, Number Of Customers, Conversion Rate 1, Conversion rate 2,
	# TO DO. where do the promo Assignment go?
	def __init__(self, nArms, customers, probabilities1, probabilities2, horizon):
		super().__init__(nArms, customers, probabilities1, probabilities2)
		self.t = 0
		nPhases = len(self.probabilities1)
		self.phaseSize = horizon/nPhases

	def round(self, ccn, pulledArm):
		# ccn - customer class number
		# binomial, (How many, probability)
		# pulled arm -> indice del mejor brazo (Each arm is a price)
		# reward = How many clients made a purchase
		
		currentPhase = int(self.t / self.phaseSize)
		if currentPhase > 3: currentPhase = 3
		p1 = self.probabilities1[currentPhase][ccn][pulledArm]
		reward = np.random.binomial(self.customers[ccn], p1)
		if (reward > 0):
			p2 = self.probabilities2[currentPhase][ccn][pulledArm]
			reward2 = np.random.binomial(reward, p2)
		else:
			reward2 = 0
		return reward, reward2
	
	def nextDay(self):
		self.t +=1