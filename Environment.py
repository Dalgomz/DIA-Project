# Environmental random variables

import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from scipy.stats import truncnorm

# Daily number of customers
mu_customers = np.array([100, 150, 200, 250])
sigma_customers = np.array([10, 20, 30, 40])

lower = 0
upper = np.inf

N = np.array([])

for i in range(0, 3):
    np.append(N, (truncnorm(
        (lower - mu_customers[i]) / sigma_customers[i], (upper - mu_customers[i]) / sigma_customers[i],
        loc=mu_customers[i], scale=sigma_customers[i])))

# Conversion rates

# Represents the different willingness to buy by customers of different classes
class_multiplier = np.array([1, 0.95, 0.9, 0.85])


# Model seasonality as a sin that reaches its max points in summer and winter
# t represent the day, will go from 0 to 364
def seasonality(t):
    return (math.cos(t / (182 / math.pi))) / 4 + 0.75


# Conversion rate for clients of class 1 buying object 1:
def p11(t, p1):
    return seasonality(t) * p1 / 100 * class_multiplier[1]
