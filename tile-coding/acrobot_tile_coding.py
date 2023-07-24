# Import common libraries
import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import QTable as qt

# Set plotting options
# %matplotlib inline
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)


class Agent:
    def __init__(self, tq, alpha, gamma):
        self.tq_table = tq
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(range(action_space_size - 1))
        else:
            return np.argmax([self.tq_table.get(state, action) for action in range(action_space_size - 1)])

    def update_q_tables(self, state, next_state, action, reward):
        # Sarsamax
        value = reward + self.gamma * np.max([self.tq_table.get(next_state, next_action) for next_action in range(action_space_size - 1)])
        self.tq_table.update(state, action, value, self.alpha)


# Create an environment
env = gym.make('Acrobot-v1')

# Explore state (observation) space
low = env.observation_space.low
high = env.observation_space.high
action_space_size = env.action_space.n

print("State space:", env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

# Explore action space
print("Action space:", env.action_space)
print("Action space size:", action_space_size)

tiling_specs = [((10, 10), (-0.066, -0.33)),
                ((10, 10), (0.0, 0.0)),
                ((10, 10), (0.066, 0.33))]

n_bins = 5
bins = tuple([n_bins]*env.observation_space.shape[0])
offset_pos = (env.observation_space.high - env.observation_space.low)/(3*n_bins)

tiling_specs = [(bins, -offset_pos),
                (bins, tuple([0.0]*env.observation_space.shape[0])),
                (bins, offset_pos)]

tq = qt.TiledQTable(low, high, tiling_specs, action_space_size)

agent = Agent(tq, alpha=0.02, gamma=0.99)

num_episodes = 1000
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01
average_every = 100
scores = []
max_avg_score = float('-inf')

for i_episode in range(1, num_episodes + 1):
    epsilon = max(epsilon*epsilon_decay, epsilon_min)
    state = env.reset(seed=505)[0]
    action = agent.get_action(state, epsilon)
    score = 0

    while True:
        # Interact with environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        score += reward

        # Update Q table (Sarsamax)
        agent.update_q_tables(state, next_state, action, reward)

        if terminated or truncated:
            scores.append(score)
            if len(scores) > average_every:
                avg_score = np.mean(scores[-average_every:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
            if i_episode % average_every == 0:
                print("Episode: {}/{}. Current max avg score is: {}".format(i_episode, num_episodes, max_avg_score))
            break

        # Get action (Choose max action from Q tables)
        action = agent.get_action(next_state, epsilon)
        state = next_state
