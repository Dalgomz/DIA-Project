import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from learners.ucb import UCB, Matching_UCB
from enviroment.Sequential_Arrival_Environment import SequentialArrivalEnvironment
from convertion_rate import Conv_rate1
from convertion_rate import Conv_rate2

np.random.seed(5)
random.seed(5)

# Number of Days
T = 365

# number of customers of each class
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


# conversion rates of each product
conv_1 = Conv_rate1(price1)
conv_2 = Conv_rate2(price2)

# two alternative settings for number of promos of each class
N_PROMOS = 4

promo_setting = np.array([0.3, 0.15, 0.25])

col_promo = [0, 0, 0, 0, 1, 2, 3]


def random_positive_choice(iterable):
    """random choice of index from non-zero elements of iterable"""
    iterable = np.array(iterable)
    indices = np.argwhere(iterable > 0).reshape(-1)
    index = np.random.choice(indices)

    return index


# Step 5
def main():
    # Initiallizing Environment and learnenrs
    environment = SequentialArrivalEnvironment(margin1=margin1, margin2=margin2, conv_rate1=conv_1, conv_rate2=conv_2)
    empirical_customer_amount = 300
    learner1 = UCB(N_CLASSES)
    extra_promos = N_CLASSES - 1  # we create additional copies of p0 as a hack for the linear sum assignment
    all_promos = N_PROMOS + extra_promos
    learner2 = Matching_UCB(all_promos * N_CLASSES, N_CLASSES, all_promos, col_promo)
    arm_pull_count= np.zeros((N_CLASSES, N_PROMOS))

    def expected_value_of_reward(pulled_arm1, pulled_arm2, current_customer_class, curr_promo):
        """calculate and return expected value of reward for arm choices based on conversion rates"""
        reward = margin1[pulled_arm1] * conv_1[current_customer_class, pulled_arm1] * Prod1
        if pulled_arm2 != -1:
            reward += margin2[pulled_arm2, curr_promo] * conv_1[current_customer_class, pulled_arm1] \
                      * conv_2[current_customer_class, pulled_arm2, curr_promo] * Prod2
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
        daily_customer_amount = sum(round_class_num)

        daily_promos = [n * empirical_customer_amount for n in promo_setting]

        # initialize variables for accumulating round rewards
        round_reward1 = 0
        round_reward2 = 0
        round_expected_reward = 0
        round_clairvoyant_expected = 0
        for c in range(daily_customer_amount):
            # simulate customer arrival by random choice of class which has customers remaining for the day
            customer_class = random_positive_choice(iterable=round_class_num)
            round_class_num[customer_class] -= 1
            chosen_promo = 0

            # pull price 1 arm, observe rewards
            arm1 = learner1.pull_arm()
            reward1 = environment.sub_round_1(customer_class, 0)  # The second parameter is 0 due to fixed prices

            # pull price 2 arm if positive reward and update else reward2 = 0
            if reward1 > 0:
                row_ind, col_ind = learner2.pull_arms(daily_promos)
                chosen_promo = col_ind[customer_class] - extra_promos

                if chosen_promo < 0:
                    chosen_promo = 0
                if chosen_promo > 0:
                    daily_promos[chosen_promo - 1] -= 1  # daily_promos [#p1 #p2 #p3], chosen_promo [p0 p1 p2 p3]

                reward2 = environment.sub_round_2(customer_class, 0,
                                                  chosen_promo)  # The second parameter is 0 due to fixed prices.  chosen_promo+1 since [p0 p1 p2 p3]
                arm2 = customer_class * all_promos + chosen_promo
                arm_pull_count[customer_class, chosen_promo]+=1
                if chosen_promo == 0:
                    #  Update all arms that correspond to P0 for a given customer_class
                    for promo in range(extra_promos + 1):
                        learner2.update_one(customer_class * all_promos + promo, reward2)
                else:
                    learner2.update_one(arm2, reward2)
            else:
                reward2 = 0
                arm2 = -1

            # update learner 1
            learner1.update(arm1, reward1 + reward2)

            arms1.append(arm1)
            arms2.append(arm2)

            # add rewards to cumulative sums of round rewards and calculate expected rewards
            round_reward1 += reward1 * Prod1
            round_reward2 += reward2 * Prod2
            round_expected_reward += expected_value_of_reward(0, 0, customer_class, chosen_promo)  # first and
            # second parameters are 0 due to fixed prices
            round_clairvoyant_expected += np.max([expected_value_of_reward(0, 0, customer_class, p)
                                                   for p in range(N_PROMOS)])

        # append round rewards to lists of rewards
        rewards1.append(round_reward1)
        rewards2.append(round_reward2)
        expected_rewards.append(round_expected_reward)
        clairvoyant_expected_rewards.append(round_clairvoyant_expected)

        # update empirical number of customers per day
        empirical_customer_amount = (empirical_customer_amount * i + daily_customer_amount)/(i + 1)

    rewards1 = np.array(rewards1)
    rewards2 = np.array(rewards2)
    expected_rewards = np.array(expected_rewards)
    clairvoyant_expected_rewards = np.array(clairvoyant_expected_rewards)
    rewards = rewards1 + rewards2

    print("\r", "Progress: {}/{} days".format(T, T))

    
    # Results
    print()
    print('Learning Results: \n')
    print("The number of times a customer of a class was offered each promo level is shown below:")
    print(arm_pull_count)
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
