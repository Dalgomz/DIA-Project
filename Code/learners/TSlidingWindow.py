from .Learner import *
from .ThompsonLearner import ThompsonLearner
import numpy as numpy

class TSlidingWindow(ThompsonLearner):
	# 7:41
	def __init__(self, nArms, windowSize):
		super().__init__(nArms)
		self.windowSize = windowSize
		self.pulledArms = np.array([])

	def update(self, pulledArm, reward):
		self.t += 1
		self.updateObservations(pulledArm, reward)
		self.pulledArms = np.append(self.pulledArms, pulledArm)
		for arm in range(self.nArms):
			nSamples = np.sum(self.pulledArms[-self.windowSize:] == arm)
			cumRew = np.sum(self.rewardsPerArm[arm][-nSamples:]) if nSamples > 0 else 0
			self.betaParameters[arm, 0] = cumRew + 1.0
			self.betaParameters[arm, 1] = nSamples - cumRew + 1.0
