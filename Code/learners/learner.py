import numpy as np

class Learner:
	def __init__(self, nArms):
		self.t = 0
		self.nArms = nArms 
		self.rewardsPerArm = x = [[] for i in range(nArms)]
		self.collectedRewards = np.array([])

	def updateObservations(self, pulledArm, reward):
		self.rewardsPerArm[pulledArm].append(reward)
		self.collectedRewards = np.append(self.collectedRewards,reward)