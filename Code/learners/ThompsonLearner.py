from .Learner import *
import numpy as numpy

class ThompsonLearner(Learner):

	def __init__(self,nArms):
		super().__init__(nArms)
		# Matrix of X, 2 filled with 1
		self.betaParameters = np.ones((nArms,2))

	def pullArm(self):
		idx = np.argmax(np.random.beta(self.betaParameters[:,0],self.betaParameters[:,1]))
		return idx 

	def update(self, pulledArm, reward):
		self.t += 1
		self.updateObservations(pulledArm, reward)
		self.betaParameters[pulledArm, 0] = self.betaParameters[pulledArm, 0] + reward
		self.betaParameters[pulledArm, 1] = self.betaParameters[pulledArm, 1] + 1.0 - reward
