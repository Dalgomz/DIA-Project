import numpy as np
import math
price_1 = np.array([40,45,50,55])
def Conv_rate1(price):
    conv_rate1=np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            conv_rate1[i][j]=0.05*i+0.5*math.cos((price[j]-30)/18)

    return conv_rate1
def Conv_rate2(price):
    conv_rate2=np.zeros((4,4,4))
    for i in range(4):
        for j in range(4):
            for k in range(4):

                conv_rate2[i][j][k]=0.045*i+0.35*math.cos((price[j]-50)/60)+0.05*k

    return conv_rate2