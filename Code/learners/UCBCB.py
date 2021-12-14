from .Learner import *
from .UCB1Learner import UCBLearner
from .CUMSUM import CUMSUM
import numpy as numpy

class UCBCB(UCBLearner):
    def __init__(self, nArms, M=100, eps=0.05, h=20, alpha=0.01):
        super().__init__(nArms)
        self.changeDetection = [CUMSUM(M, eps, h) for _ in range(nArms)]
        self.validRewardsPerArms = [[] for _ in range(nArms)]
        self.detections = [[] for _ in range(nArms)]
        self.alpha = alpha

    def pullArm(self):
        if np.random.binomial(1, 1 - self.alpha):
            return np.argmax(self.empiricalMeans + self.confidence)
        else:
            return np.random.randint(self.nArms)

    def update(self, pulledArm, reward):
        self.t += 1
        if self.changeDetection[pulledArm].update(reward):
            self.detections[pulledArm].append(self.t)
            self.validRewardsPerArms[pulledArm] = []
            self.changeDetection[pulledArm].reset()
        self.updateObservations(pulledArm, reward)
        self.empiricalMeans[pulledArm] = np.mean(self.validRewardsPerArms[pulledArm])
        totalValidSamples = sum([len(x) for x in self.validRewardsPerArms])

        for a in range(self.nArms):
            nSamples = len(self.validRewardsPerArms[a])
            self.confidence[a] = (2 * np.log(totalValidSamples) / nSamples) ** 0.5 if nSamples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        self.rewardsPerArm[pulled_arm].append(reward)
        self.validRewardsPerArms[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)