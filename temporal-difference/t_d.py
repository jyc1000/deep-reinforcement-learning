import gymnasium as gym
import numpy as np
import sys
from collections import defaultdict
import check_test

def sarsa(env, num_episodes, alpha):
    Q = defaultdict(lambda: np.zeros(env.nA))

    for i_episode in range(1, num_episodes+1):
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        epsilon = 0.5
        eps_decay_rate = 0.9999
        eps_min = 0.05

        state = env.reset()[0]

        while True:
            epsilon = np.max([epsilon*eps_decay_rate, eps_min])
            action = get_next_action(env, Q, state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            # Update Q
            next_action = get_next_action(env, Q, next_state, epsilon)
            Q_old = Q[state][action]
            Q[state][action] = Q_old + alpha*(reward + Q[next_state][next_action] - Q_old)
            state = next_state
    return Q

def get_next_action(env, Q, state, epsilon):
    probs = np.ones(env.nA)*epsilon/env.nA
    if state in Q:
        max_action = np.argmax(Q[state])
    else:
        max_action = np.random.choice(np.arange(env.nA))
    probs[max_action] = 1 - epsilon + epsilon/env.nA
    action = np.random.choice(np.arange(env.nA), p = probs)
    return action

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

