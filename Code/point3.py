import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.financial import rate
from enviroment.Environment import Enviroment
from learners.thompson import Thompson
from learners.ucb import UCB
from convertion_rate import Conv_rate1
from convertion_rate import Conv_rate2



np.random.seed(0)


# Task 3
def main():
   
    T = 365
    # arms apply to price 1
    N_PRICES = 4
    price_1 = np.array([40,45,50,55])

    # number of arms equal to number of price candidates for product 1
    n_arms = N_PRICES

    # since promos only apply to price 2, we consider that we only have one promo (0% discount)
    n_promos = 1

    # maximum revenue margin can't be larger than 1. Divide scenario margins by 700
    margin1 = np.array([ 10., 14., 18., 22.]) / 30.
    # price 2 is fixed so we have the maximum margin possible for each sale
    margin2 = np.array([5.]) / 70.

    # Number of customers of class i
    n_customers = np.array([400, 300, 200, 100])
 

    # Conversion rate of the first item, from class i at price j.
    conv_rate1 = Conv_rate1(price_1)

  
    # Conversion rate of the second item, from class i at original price j, discount from promo k
    # conversion rates are the same for every arm
    conv_rate2 = np.array([[[.25]] * 4, [[.05]] * 4, [[.15]] * 4, [[.45]] * 4])

  
    # in this case 100% of every class has the only promo available
    promo_assig = np.array([[1.], [1.], [1.], [1.]])

  
    st_env1 = Enviroment(n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig)
    st_env2 = Enviroment(n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig)

    learner1 = Thompson(n_arms)
    learner2 = UCB(n_arms)

    rewards1 = []
    rewards2 = []
    arms1 = []
    arms2 = []
    for i in range(T):
        arm1 = learner1.pull_arm()
        arm2 = learner2.pull_arm()

        reward1 = st_env1.round(arm1)
        reward2 = st_env2.round(arm2)

        learner1.update(arm1, reward1)
        learner2.update(arm2, reward2)

        rewards1.append(reward1)
        rewards2.append(reward2)

        arms1.append(arm1)
        arms2.append(arm2)

    print()
    print('LEARNING RESULTS')
    print()
    print("Thompson learner converges to price {}$ for product 1".format(price_1[np.argmax([len(a) for a in learner1.rewards_per_arm])]))
    print("UCB learner converges to price {}$ for product 1".format(price_1[np.argmax([len(a) for a in learner2.rewards_per_arm])]))
    print()
    print(f'Total margin collected by UCB: {np.sum(rewards2)}')
    print(f'Total margin collected by Thompson Sampling: {np.sum(rewards1)}')

    plt.plot(np.cumsum(rewards1), label='Thomson Sampling', color='tab:blue')
    plt.plot(np.cumsum(rewards2), label='UCB1', color='tab:green')
    plt.legend(loc='lower right')
    plt.grid(linestyle='--')
    plt.xlabel('Days')
    plt.ylabel('Reward')
    plt.title('Cumulative Rewards collected by both learners')
    plt.show()

    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    # regrets are calculated in terms of the expected value of the reward for each pulled arm
    def expected_value_of_reward(pulled_arm):
        reward = 0
        for cust_class in range(len(n_customers)):
            reward += margin1[pulled_arm] * conv_rate1[cust_class, pulled_arm]
            for promo in range(n_promos):
                reward += margin2[promo] * conv_rate2[cust_class, pulled_arm, promo] * promo_assig[cust_class, promo] * \
                          conv_rate1[cust_class, pulled_arm]
        return reward

    # since environment is stationary we can just get the argmax of the expected reward for a single round of each arm
    # to get the arm chosen by the clairvoyant algorithm
    clairvoyant_arm = np.argmax([expected_value_of_reward(i) for i in range(4)])

    # expected rewards for each algorithm in each round
    rewards_expected1 = []
    rewards_expected2 = []
    rewards_clairvoyant_expected = []
    for i in range(T):
        arm1 = arms1[i]
        arm2 = arms2[i]
        rewards_expected1.append(expected_value_of_reward(arm1))
        rewards_expected2.append(expected_value_of_reward(arm2))
        rewards_clairvoyant_expected.append(expected_value_of_reward(clairvoyant_arm))

    print()
    print(f'Total expected regret of UCB: {np.sum(np.subtract(rewards_clairvoyant_expected, rewards_expected2))}')
    print(f'Total expected regret of Thompson Sampling: '
          f'{np.sum(np.subtract(rewards_clairvoyant_expected, rewards_expected1))}')

    plt.plot(moving_average(rewards_expected1, 5), label='Thomson Sampling', color='tab:blue')
    plt.plot(moving_average(rewards_expected2, 5), label='UCB1',  color='tab:green')
    plt.plot(rewards_clairvoyant_expected, label='Clairvoyant', color='tab:red')
    plt.legend(loc='lower right')
    plt.grid(linestyle='--')
    plt.xlabel('Days')
    plt.ylabel('Expected Reward')
    plt.title('Expected rewards for each algorithm (Average 5 Days)')
    plt.show()



if __name__ == '__main__':
    main()
