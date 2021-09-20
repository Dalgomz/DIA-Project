import random
import numpy as np
import matplotlib.pyplot as plt

from learners.ucb import UCB
from enviroment.Sequential_Arrival_Environment import SequentialArrivalEnvironment

from convertion_rate import Conv_rate1
from convertion_rate import Conv_rate2

np.random.seed(4)
random.seed(4)

# Number of Days
T = 365

# max number of customers of each class
N_CLASSES = 4
n_class = np.array([400, 300, 200, 100])

# respective base prices of the two products
Prod1 = 30
Prod2 = 10

# Prices of products
N_PRICES = 4
price1 = np.array([40, 45, 50, 55])

price2 = np.array([20, 24, 28, 32])

# Profit margin for each price candidate for both products
margin1 = np.array([10., 15., 20., 25.]) / Prod1
margin2 = np.array([[(10-x*3) for x in range(0, 4)],
                    [(14-x*3) for x in range(0, 4)],
                    [(18-x*3) for x in range(0, 4)],
                    [(22-x*3) for x in range(0, 4)]]) / Prod2


# conversion of each class for each price candidate of product 1
conv_1 = Conv_rate1(price1)

# conversion of each class for each price candidate (axis 0) and promo (axis 1) of product 2
conv_2 = Conv_rate2(price2)

# two alternative settings for number of promos of each class
N_PROMOS = 4


promo_assignment = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

n_arms1 = int(conv_1.size / N_CLASSES)
n_arms2 = int(conv_2.size / N_CLASSES / N_PROMOS)


def random_positive_choice(iterable):
    """random choice of index from non-zero elements of iterable"""
    iterable = np.array(iterable)
    indices = np.argwhere(iterable > 0).reshape(-1)
    index = np.random.choice(indices)

    return index


# Step 4
def main():
    # Initiallizing Environment and learnenrs
    environment = SequentialArrivalEnvironment(margin1=margin1, margin2=margin2, conv_rate1=conv_1, conv_rate2=conv_2)
    learner1 = UCB(n_arms1)
    learner2 = UCB(n_arms2)

    def expected_value_of_reward(pulled_arm1, pulled_arm2, current_customer_class, current_promo_assignment):
        """calculate and return expected value of reward for arm choices based on conversion rates"""
        reward = margin1[pulled_arm1] * conv_1[current_customer_class, pulled_arm1] * Prod1
        if pulled_arm2 != -1:
            promo = np.argwhere(current_promo_assignment[current_customer_class] == 1)[0]
            reward += margin2[pulled_arm2, promo] * conv_1[current_customer_class, pulled_arm1]\
                * conv_2[current_customer_class, pulled_arm2, promo] * Prod2
        return reward

    # Start of Learning 
    print(">> Starting Learning <<")

    rewards1 = []
    rewards2 = []
    expected_rewards = []
    clairvoyant_expected_rewards = []
    arms1 = []
    arms2 = []
    for i in range(T):
        print('\r', "Progress: {}/{} days".format(i, T), end=" ") if i % 5 == 0 else False
        # sample number of customer for each class and truncate at 0 to avoid negative
        round_class_num = np.random.normal(n_class, 10)
        round_class_num = [int(n) if n >= 0 else 0 for n in round_class_num]
        # initialize variables for accumulating round rewards
        round_reward1 = 0
        round_reward2 = 0
        round_expected_reward = 0
        round_clairvoyant_expected = 0
        for c in range(sum(round_class_num)):
            # simulate customer arrival by random choice of class which has customers remaining for the day
            customer_class = random_positive_choice(iterable=round_class_num)
            round_class_num[customer_class] -= 1

            # pull price 1 arm, observe rewards
            arm1 = learner1.pull_arm()
            reward1 = environment.sub_round_1(customer_class, arm1)

            # pull price 2 arm if positive reward and update else reward2 = 0
            if reward1 > 0:
                arm2 = learner2.pull_arm()
                reward2 = environment.sub_round_2(customer_class, arm2, 0)
                learner2.update(arm2, reward2)
            else:
                reward2 = 0
                arm2 = -1

            # update learner 1
            learner1.update(arm1, reward1+reward2)

            arms1.append(arm1)
            arms2.append(arm2)

            # add rewards to cumulative sums of round rewards and calculate expected rewards
            round_reward1 += reward1 * Prod1
            round_reward2 += reward2 * Prod2
            round_expected_reward += expected_value_of_reward(arm1, arm2, customer_class, promo_assignment)
            round_clairvoyant_expected += np.max([[expected_value_of_reward(i, j, customer_class, promo_assignment)
                                                   for i in range(N_PRICES)]
                                                  for j in range(N_PRICES)])

        # append round rewards to lists of rewards
        rewards1.append(round_reward1)
        rewards2.append(round_reward2)
        expected_rewards.append(round_expected_reward)
        clairvoyant_expected_rewards.append(round_clairvoyant_expected)

    rewards1 = np.array(rewards1)
    rewards2 = np.array(rewards2)
    expected_rewards = np.array(expected_rewards)[:, 0]
    clairvoyant_expected_rewards = np.array(clairvoyant_expected_rewards)
    rewards = rewards1 + rewards2

    print("\r", "Progress: {}/{} days".format(T, T))

    # Results
    print()
    print('Learning Results: \n')
    print("learner 1 converges to price {}$ for product 1".format(price1[np.argmax([len(a) for a in learner1.rewards_per_arm])]))
    print("learner 2 converges to price {}$ for product 2".format(price2[np.argmax([len(a) for a in learner2.rewards_per_arm])]))
    print()
    print(f'Total profit collected from product 1: {np.sum(rewards1)}')
    print(f'Total profit collected from product 2: {np.sum(rewards2)}')

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(np.cumsum(rewards1), label='Product 1 (Game)', color='tab:blue')
    ax1.legend(loc='lower right')
    ax1.grid(linestyle='--')
    ax1.set_ylabel('Cumulative Reward')

    ax2.plot(np.cumsum(rewards2), label='Product 2 (DLC Bundle)', color='tab:green')
    ax2.legend(loc='lower right')
    ax2.grid(linestyle='--')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Cumulative Reward')

    fig.suptitle('Cumulative Rewards from each product')
    plt.show()

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
  
    # regrets are calculated in terms of the expected value of the reward for each pulled arm

    print()
    print(f'Total expected regret: {np.sum(np.subtract(clairvoyant_expected_rewards, expected_rewards))}')

    plt.plot(moving_average(expected_rewards, 10), label='UCB Expected rewards', color="tab:green")
    plt.plot(np.ones(len(expected_rewards))*np.mean(expected_rewards), color='steelblue', linestyle=(0,(6,10)))
    plt.plot(moving_average(clairvoyant_expected_rewards, 10), label='Clairvoyant Expected rewards',
             color='tab:red')
    plt.plot(np.ones(len(clairvoyant_expected_rewards))*np.mean(clairvoyant_expected_rewards), color='tomato', linestyle=(0,(6,10)))
    plt.legend(loc='center right')
    plt.title('Expected rewards of each algorithm (Average 10 Days)')
    plt.xlabel('Days')
    plt.ylabel('Expected Reward')
    plt.grid(linestyle='--')
    plt.show()

if __name__ == '__main__':
    main()
