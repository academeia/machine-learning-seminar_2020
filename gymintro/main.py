#!/usr/bin/env python3
import gym
import numpy as np

A = 3 # the number of action
N = 50 # the number of one of status
lr = 0.2 # the learning rate
gamma = 0.99 # the time discount rate
epsilon = 0.002 # used in the epsilon-greedy method
episode_num = 10000 # the number of episodes
step_num = 200 # the number of steps in one episode

image_dir = 'images' # directory to save images
save_image = False # if True, then save result simulation as mp4 file
show_rewards = False # if True, then show rewards graph

# transform observation to status in {0, ..., N-1}
def get_status(env, observation):
    env_min = env.observation_space.low
    env_max = env.observation_space.high
    env_dx = (env_max - env_min) / N
    p = int((observation[0] - env_min[0]) / env_dx[0]) # position
    v = int((observation[1] - env_min[1]) / env_dx[1]) # velocity
    return p, v


def update_q_table(q_table, action, observation, next_observation, reward):
    # Q(s, a)
    p, v = get_status(env, observation)
    q_value = q_table[p][v][action]

    # max Q(s', a')
    next_p, next_v = get_status(env, next_observation)
    next_max_q_value = max(q_table[next_p][next_v])

    # update q-table
    q_table[p][v][action] = q_value + lr * (reward + gamma * next_max_q_value - q_value)

    return q_table


def get_action(env, q_table, observation, epsilon=epsilon):
    if np.random.uniform(0, 1) > epsilon:
        p, v = get_status(env, observation)
        action = np.argmax(q_table[p][v])
    else:
        action = np.random.choice(range(A))
    return action


def one_episode(env, q_table, init_observation, rewards, episode):
    # initialization
    total_reward = 0
    observation = init_observation

    for _ in range(step_num):
        # choose an action by epsilon-greedy method
        action = get_action(env, q_table, observation)

        # move the car, get the next observation and reward
        next_observation, reward, done, _info = env.step(action)

        # update q-table
        q_table = update_q_table(q_table, action, observation, next_observation, reward)

        # update observation and `total_reward`
        observation = next_observation
        total_reward += reward

        if done: # if True, then finish one episode
            rewards.append(total_reward)
            if episode % 100 == 0:
                print("episode: {}, total_reward: {}".format(episode, total_reward))
            break

    return q_table, rewards


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    rewards = []

    # initialize q-table
    q_table = np.zeros((N, N, A))

    # learning
    for episode in range(episode_num):
        init_observation = env.reset()
        q_table, rewards = one_episode(env, q_table, init_observation, rewards, episode)


    # show rewards
    if show_rewards:
        import matplotlib.pyplot as plt
        plt.plot(range(episode_num), rewards)
        plt.show()

    # to save mp4 file
    if save_image:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder_dir = os.path.join(current_dir, image_dir)
        os.makedirs(folder_dir, exist_ok=True)
        from gym import wrappers
        env = wrappers.Monitor(env, 'images', force=True)

    # initialization
    observation = env.reset()
    
    # show result of learning
    for _ in range(step_num):

        # choose an action by taking argmax of q-table
        action = get_action(env, q_table, observation, epsilon=-1)

        # move the car, get the next observation and reward
        observation, _reward, done, _info = env.step(action)

        # show environment
        env.render()

        if done:
            env.close()
            break

