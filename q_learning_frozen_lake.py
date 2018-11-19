import gym
import os
import numpy as np

import matplotlib.pyplot as plt
from gym import wrappers
import time
from datetime import datetime

# Environment initialization
folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'q_learning')
env = gym.wrappers.Monitor(gym.make('FrozenLake-v0'), folder, force=True)

# Q and rewards
Q = np.zeros((env.observation_space.n, env.action_space.n))

rewards = []
iterations = []

# Parameters
alpha = 0.9
discount = 0.95
episodes = 20000

start_time = time.time()

# Episodes
for episode in range(episodes):
    # Refresh state
    state = env.reset()
    done = False
    t_reward = 0
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    # Run episode
    for i in range(max_steps):
        if done:
            break

        current = state
        action = np.argmax(Q[current, :] + np.random.randn(1, env.action_space.n) * (1 / float(episode + 1)))
        state, reward, done, info = env.step(action)
        t_reward += reward
        Q[current, action] += alpha * (reward + discount * np.max(Q[state, :]) - Q[current, action])
    print('Q[current, :]',Q[current, :])
    # alpha = 0.5-0.00004*episode
    rewards.append(t_reward)
    iterations.append(i)

# Close environment
env.close()

# Plot results
def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

size = int(episodes / 50)
print('size',size)
chunks = list(chunk_list(rewards, size))

averages = [sum(chunk) / len(chunk) for chunk in chunks]
print('averages',averages)

end_time = time.time() - start_time
print('Wall clock time', end_time)
plt.plot(range(0, len(rewards), size), averages)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()

# Push solution
api_key = os.environ.get('GYM_API_KEY', False)
if api_key:
    print ('Push solution? (y/n)')
    if raw_input().lower() == 'y':
        gym.upload(folder, api_key=api_key)

# learning rate = 0.2 and episodes= 20k
#averages_lr2= [0.085, 0.1925, 0.2, 0.225, 0.18, 0.22, 0.195, 0.185, 0.265, 0.2275, 0.2275, 0.195, 0.2075, 0.185, 0.2275, 0.39, 0.4525, 0.4225, 0.455, 0.3975, 0.44, 0.4475, 0.465, 0.435, 0.495, 0.45, 0.44, 0.46, 0.45, 0.48, 0.4125, 0.4525, 0.4075, 0.455, 0.4725, 0.44, 0.4675, 0.47, 0.455, 0.44, 0.4875, 0.4125, 0.4475, 0.44, 0.425, 0.435, 0.4275, 0.415, 0.425, 0.445]
# learning rate = 0.3 and episodes= 20k
#averages_lr3=[0.1725, 0.3475, 0.2825, 0.3125, 0.32, 0.2775, 0.2975, 0.325, 0.3025, 0.335, 0.34, 0.6525, 0.66, 0.625, 0.6475, 0.64, 0.6275, 0.585, 0.645, 0.655, 0.625, 0.6125, 0.65, 0.65, 0.5825, 0.6125, 0.6675, 0.655, 0.6525, 0.595, 0.63, 0.6125, 0.6575, 0.6475, 0.6575, 0.6325, 0.6475, 0.625, 0.6425, 0.645, 0.6375, 0.6325, 0.6225, 0.6175, 0.6725, 0.62, 0.6575, 0.6225, 0.675, 0.66]
# learning rate = 0.4 and episodes= 20k
#averages_lr4=[0.205, 0.375, 0.4025, 0.485, 0.47, 0.485, 0.4925, 0.5, 0.5125, 0.52, 0.375, 0.4575, 0.4475, 0.445, 0.475, 0.4875, 0.4725, 0.3575, 0.38, 0.39, 0.41, 0.2925, 0.375, 0.3725, 0.4225, 0.3725, 0.3775, 0.3925, 0.42, 0.3525, 0.4075, 0.365, 0.375, 0.4425, 0.5175, 0.4875, 0.475, 0.4875, 0.51, 0.46, 0.4475, 0.47, 0.505, 0.49, 0.515, 0.485, 0.52, 0.35, 0.3175, 0.4025]
# learning rate = 0.5 and episodes= 20k
#averages_lr5=[0.3125, 0.53, 0.565, 0.505, 0.4775, 0.545, 0.51, 0.715, 0.6975, 0.7125, 0.7025, 0.745, 0.7725, 0.7375, 0.7975, 0.745, 0.7225, 0.705, 0.6825, 0.7225, 0.69, 0.76, 0.7275, 0.7075, 0.7475, 0.725, 0.7025, 0.725, 0.7725, 0.7175, 0.69, 0.7125, 0.745, 0.7425, 0.755, 0.765, 0.7675, 0.7325, 0.74, 0.7375, 0.7975, 0.765, 0.7125, 0.7225, 0.725, 0.7325, 0.7475, 0.7225, 0.735, 0.7325]

# learning rate = 0.6 and episodes= 20k
#averages_lr6=[0.1125, 0.445, 0.4375, 0.4075, 0.44, 0.44, 0.4475, 0.52, 0.58, 0.6025, 0.5575, 0.5875, 0.6125, 0.605, 0.5725, 0.5175, 0.595, 0.61, 0.58, 0.575, 0.61, 0.585, 0.58, 0.5775, 0.525, 0.635, 0.5975, 0.595, 0.5875, 0.535, 0.625, 0.575, 0.6625, 0.62, 0.6, 0.6175, 0.595, 0.57, 0.6125, 0.58, 0.585, 0.5875, 0.595, 0.58, 0.5875, 0.58, 0.5975, 0.575, 0.545, 0.58]

# learning rate = 0.7 and episodes= 20k
#averages_lr7=[0.1475, 0.4875, 0.54, 0.63, 0.5975, 0.5725, 0.54, 0.5625, 0.59, 0.5775, 0.4725, 0.53, 0.5975, 0.515, 0.5525, 0.6075, 0.5875, 0.58, 0.585, 0.5575, 0.5675, 0.5625, 0.625, 0.6075, 0.6, 0.625, 0.6, 0.5975, 0.5925, 0.5775, 0.585, 0.56, 0.57, 0.565, 0.58, 0.5875, 0.6125, 0.5725, 0.5775, 0.5975, 0.54, 0.585, 0.56, 0.5675, 0.565, 0.5875, 0.5775, 0.525, 0.59, 0.59]

# learning rate = 0.8 and episodes= 20k
#averages_lr8=[0.06, 0.4525, 0.5675, 0.535, 0.56, 0.5225, 0.58, 0.5475, 0.5775, 0.5875, 0.5575, 0.56, 0.5475, 0.55, 0.5, 0.5425, 0.56, 0.5875, 0.59, 0.565, 0.63, 0.5775, 0.525, 0.5825, 0.605, 0.5975, 0.565, 0.57, 0.5875, 0.53, 0.6125, 0.635, 0.5275, 0.5875, 0.5725, 0.615, 0.5875, 0.6225, 0.5925, 0.59, 0.6325, 0.575, 0.595, 0.5775, 0.585, 0.5975, 0.615, 0.58, 0.6175, 0.5675]

# learning rate = 0.9 and episodes= 20k
#averages_lr9=[0.1625, 0.5725, 0.7175, 0.755, 0.745, 0.745, 0.6925, 0.73, 0.7275, 0.7525, 0.7775, 0.72, 0.7375, 0.7075, 0.7225, 0.645, 0.7475, 0.67, 0.6925, 0.69, 0.7175, 0.7075, 0.76, 0.725, 0.71, 0.6975, 0.7275, 0.7, 0.7325, 0.71, 0.72, 0.7275, 0.6975, 0.7325, 0.765, 0.7525, 0.7, 0.7375, 0.725, 0.6475, 0.7125, 0.745, 0.675, 0.715, 0.715, 0.75, 0.74, 0.71, 0.705, 0.75]

#decay lr=0.9
#averages_lr9_decay =[0.0075, 0.5525, 0.545, 0.57, 0.5275, 0.5225, 0.52, 0.5625, 0.555, 0.515, 0.5025, 0.535, 0.52, 0.545, 0.585, 0.55, 0.5475, 0.54, 0.535, 0.5575, 0.5475, 0.5475, 0.5475, 0.52, 0.5275, 0.6, 0.5275, 0.5225, 0.595, 0.5125, 0.5575, 0.5525, 0.535, 0.5025, 0.5525, 0.5425, 0.5425, 0.5575, 0.5325, 0.5975, 0.61, 0.5875, 0.58, 0.5225, 0.56, 0.575, 0.5225, 0.5475, 0.54, 0.5675]
#decay lr= 0.5
#averages_lr5_decay =[0.24, 0.3075, 0.2875, 0.305, 0.3075, 0.3175, 0.325, 0.2575, 0.315, 0.2475, 0.3125, 0.3275, 0.335, 0.28, 0.2775, 0.33, 0.29, 0.29, 0.285, 0.2825, 0.2825, 0.305, 0.295, 0.285, 0.27, 0.3175, 0.285, 0.305, 0.3025, 0.2975, 0.285, 0.2975, 0.3, 0.31, 0.3525, 0.2875, 0.315, 0.3275, 0.32, 0.31, 0.2775, 0.325, 0.305, 0.275, 0.325, 0.315, 0.2875, 0.335, 0.33, 0.2875]
