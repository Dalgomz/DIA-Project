from .Learner import *
import numpy as numpy

class UCBLearner(Learner):

	def __init__(self,nArms):
		super().__init__(nArms)
		self.empiricalMeans = np.zeros(nArms)
		self.confidence = np.zeros(nArms)

	def pullArm(self):
		if self.t < self.nArms:
			arm = self.t
		else:
			upperBound = self.empiricalMeans + self.confidence
			arm = np.random.choice(np.where(upperBound == upperBound.max())[0])
		return arm
	
		
	def update(self, pulledArm, reward):
		self.t += 1
		self.updateObservations(pulledArm, reward)
		self.empiricalMeans[pulledArm] = (self.empiricalMeans[pulledArm]*(self.t-1)+reward)/self.t
		for a in range(self.nArms):
			numberPulled = max(1,len(self.rewardsPerArm[a]))
			self.confidence[a] = (2*np.log(self.t) / numberPulled)**0.5    
		#two commented line has do the same thing of update_observations(), but professor split is. why?
