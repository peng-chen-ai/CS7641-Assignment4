
import gym
import gym.spaces
import numpy as np
from collections import defaultdict
from collections import deque
import sys
import math
import time
start_time = time.time()

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        #list1 = list(range(0, env.observation_space.n))
        # for i in range(len(list1)):
        #     list1[i] = str(list1[i])
        # self.Q = dict(zip(list1, [np.zeros(self.nA)] * 500))

        #print('self.Q',self.Q )
        self.epsilon = 0.0001
        #self.epsilon = 1.0
        self.alpha = 0.2
        self.gamma = 0.1

        print('Epsilon: {}, Alpha = {}'.format(self.epsilon ,self.alpha) )

    def epsilon_greedy_probs(self, Q_s, epsilon):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """

        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)

        # print('np.argmax(Q_s)',np.argmax(Q_s))
        # print(1 - epsilon + (epsilon / self.nA))
        return policy_s

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        #Use epsilon-greedy(Q) policy to choose the action.
        #state_policy = np.ones(self.nA) * self.epsilon / self.nA
        #state_policy[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon / self.nA)

        state_policy = self.epsilon_greedy_probs(self.Q[state], self.epsilon)
        action = np.random.choice(np.arange(self.nA), p=state_policy)

        '''
        if  np.random.random() < epsilon:
            action = np.random.randint(self.nA)
        else:
            action = np.argmax(self.Q[state])
        '''
        return action

    def step(self, state, action, reward, next_state, done,reward_boundary,total_episode):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        #Q-learning (sarsamax)
        old_Q = self.Q[state][action]

        self.Q[state][action] = old_Q + (self.alpha * (reward + (self.gamma * np.max(self.Q[next_state]) - old_Q)))

        if len(self.Q) > 400 or total_episode > 20000-1:
            print('Q', len(self.Q))

        if reward_boundary>9.2:
            Q_List = [[k,str(v)] for v,k in sorted([[np.argmax(v),k] for k, v in self.Q.items()],reverse=True)]
            Q_sorted = sorted(Q_List, key=lambda x: x[0])
            print('Q_sorted',Q_sorted)
            #print('Q',self.Q)
            #print('Q_List',len(Q_List))
            #print([el[1] for el in Q_sorted])

def interact(env, agent, num_episodes=20000, window=100):
    """ Monitor agent's performance.

    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    #epsilon = 0.9997
    state_list = []
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        state = env.reset()
        state_list.append(state)
        state_list = list(set(state_list))
        # initialize the sampled reward
        samp_reward = 0
        while True:
            # agent selects an action
            action = agent.select_action(state)
            # action = agent.select_action(state,epsilon)
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)
            # agent performs internal updates based on sampled experience
            if i_episode >500 and avg_reward>9.2:
                agent.step(state, action, reward, next_state, done,avg_reward,i_episode)
            else:
                agent.step(state, action, reward, next_state, done,reward_boundary=1.0,total_episode=500)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            if done:
                # save final sampled reward
                samp_rewards.append(samp_reward)
                break
        #epsilon *= 0.9997
        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == num_episodes: print('\n')

    # print('state_list',state_list)
    # print(len(state_list))
    return avg_rewards, best_avg_reward

env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)
print(avg_rewards)

end_time = time.time() - start_time
print('Wall clock time', end_time)

import matplotlib.pyplot as plt
positive_only = 1
if positive_only == 1:
    negative = [item for item in avg_rewards if item < 8]
    positive = [item for item in avg_rewards if item >= 8]
    negative_to_zero = [8.0] * len(negative)
    negative_to_zero.extend(positive)
    #print('negative',negative)
    line11 = plt.plot(range(0, len(negative_to_zero), 1), negative_to_zero)
else:
    line11 = plt.plot(range(0, len(avg_rewards), 1), avg_rewards)

plt.legend(loc='lower right')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()
