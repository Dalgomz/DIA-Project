import numpy as np
import matplotlib.pyplot as plt 
from Environment import *
from ThompsonLearner import *

nArms = 4
p = np.array([0.15, 0.1, 0.1, 0.35])
opt = p[3]

T = 300

nExperiments = 5
tsRewardsPerExperiment = []

for e in range(0,nExperiments):
	print('\r', "Progress: {}/{} days".format(e, nExperiments), end=" \n")# if e % 5 == 0 else False
	env = Environment(nArms = nArms, probabilities = p)
	tsLearner = ThompsonLearner(nArms)
	for t in range(0,T):
		pulledArm = tsLearner.pullArm()
		reward = env.round(pulledArm)
		tsLearner.update(pulledArm, reward)
	tsRewardsPerExperiment.append(tsLearner.collectedRewards)
	print(opt)
	
	

plt.figure(0)
plt.plot(np.cumsum(np.mean(opt-tsRewardsPerExperiment,axis=0)),'x')
plt.show()