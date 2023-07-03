import gymnasium as gym
import numpy as np
import sys
from collections import defaultdict, deque
import check_test
from plot_utils import plot_values
import matplotlib.pyplot as plt
import random
import math


def sarsa(env, num_episodes, alpha, gamma = 1.0, plot_every = 100):
    Q = defaultdict(lambda: np.zeros(env.nA))
    # monitor performance
    tmp_scores = deque(maxlen=plot_every)  # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)  # average scores over every plot_every episodes
    epsilon = 0.5
    eps_decay_rate = 0.9999
    eps_min = 0.05


    for i_episode in range(1, num_episodes+1):
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        score = 0
        # epsilon = np.max([epsilon*eps_decay_rate, eps_min])
        epsilon = 1 / i_episode

        state = env.reset()[0]
        action = epsilon_greedy_policy(env, Q, state, epsilon)

        while True:

            next_state, reward, terminated, truncated, info = env.step(action)
            score += reward
            if terminated or truncated:
                Q_old = Q[state][action]
                Q[state][action] = Q_old + alpha*(reward + 0 - Q_old)
                tmp_scores.append(score)
                break
            else:
                # Update Q
                next_action = epsilon_greedy_policy(env, Q, next_state, epsilon)
                Q_old = Q[state][action]
                Q[state][action] = Q_old + alpha*(reward + Q[next_state][next_action] - Q_old)
                state = next_state
                action = next_action
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print("\n")
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))
    return Q


def q_learning(env, num_episodes, alpha, gamma=1.0, plot_every = 100):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    avg_scores = deque(maxlen = num_episodes)
    tmp_scores = deque(maxlen = plot_every)

    epsilon = 0.5
    eps_decay_rate = 0.8
    eps_min = 0.0

    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # epsilon = np.max([epsilon*eps_decay_rate, eps_min])
        epsilon = epsilon*eps_decay_rate
        # epsilon = 1/i_episode

        state = env.reset()[0]
        score = 0
        while True:
            action = epsilon_greedy_policy(env, Q, state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            score += reward
            if terminated or truncated:
                # Update Q
                Q_old = Q[state][action]
                Q[state][action] = Q_old + alpha*(reward + 0 - Q_old)
                tmp_scores.append(score)
                break
            else:
                # Update Q
                Q_old = Q[state][action]
                next_action = argmax_policy(env, Q, next_state)
                Q[state][action] = Q_old + alpha*(reward + Q[next_state][next_action] - Q_old)
                state = next_state
        if i_episode % 100 == 0:
            avg_scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print("\n")
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))

    return Q

def expected_sarsa(env, num_episodes, alpha, gamma=1.0, plot_every = 100):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    avg_scores = deque(maxlen=num_episodes)
    tmp_scores = deque(maxlen=plot_every)

    epsilon = 0.005
    # eps_decay_rate = 0.9999
    # eps_min = 0.05

    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()[0]
        # epsilon = np.max([epsilon*eps_decay_rate, eps_min])
        # epsilon = 1/i_episode
        score = 0

        while True:
            action = epsilon_greedy_policy(env, Q, state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            score += reward

            # Update Q
            Q_old = Q[state][action]
            probs = np.ones(env.nA) * epsilon / env.nA
            if state in Q:
                max_action = np.argmax(Q[next_state])
            else:
                max_action = np.random.choice(np.arange(env.nA))
            probs[max_action] = 1 - epsilon + epsilon / env.nA
            Q[state][action] = Q_old + alpha*(reward + np.dot(Q[next_state], probs) - Q_old)

            if terminated or truncated:
                tmp_scores.append(score)
                break
            state = next_state
        avg_scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print("\n")
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))

    return Q

def epsilon_greedy_policy(env, Q, state, epsilon):
    probs = np.ones(env.nA)*epsilon/env.nA
    if state in Q:
        max_action = np.argmax(Q[state])
    else:
        max_action = np.random.choice(np.arange(env.nA))
    probs[max_action] = 1 - epsilon + epsilon/env.nA
    action = np.random.choice(np.arange(env.nA), p = probs)
    return action

def argmax_policy(env, Q, state):
    if state in Q:
        max_action = np.argmax(Q[state])
    else:
        max_action = np.random.choice(np.arange(env.nA))
    return max_action

env = gym.make('CliffWalking-v0')
print(env.action_space)
print(env.observation_space)

# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(env, 5000, .01)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)

# obtain the estimated optimal policy and corresponding action-value function
Q_sarsamax = q_learning(env, 5000, .01)

# print the estimated optimal policy
policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
check_test.run_check('td_control_check', policy_sarsamax)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsamax)

# plot the estimated optimal state-value function
plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])

# obtain the estimated optimal policy and corresponding action-value function
Q_expsarsa = expected_sarsa(env, 5000, 1)

# print the estimated optimal policy
policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_expsarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_expsarsa)

# plot the estimated optimal state-value function
plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])