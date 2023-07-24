from unityagents import UnityEnvironment
import numpy as np
import ddpg_agent
from collections import deque
import torch

# env = UnityEnvironment(file_name='Reacher.app')
env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ddpg_agent = ddpg_agent.Agent(state_size, action_size, 123)
num_episodes = 200
scores_window = deque(maxlen=100)
# scores = np.array()
import time

for i_episode in range(1, num_episodes+1):
    start = time.time()
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        # actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = ddpg_agent.act(states)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        ddpg_agent.step(states, actions, rewards, next_states, dones)
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break

    scores_window.append(scores)
    # scores.append(scores)
    if i_episode % 10 == 0:
        print("Episode {} avg score: {}".format(i_episode, np.mean(scores_window)))

    if np.mean(scores_window) >= 30.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                     np.mean(scores_window)))
        # torch.save(dqn_agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        break

    print(f'Episode: {i_episode}. Total score: {np.mean(scores) :.3f}. Device = {device}. Time per episode: {time.time()-start :.2f} seconds')
    # print(f"Device = {device}; Time per episode: {time.time()-start :.3f} seconds")
    # print(f"Device = {device}; Time per batch: {(time.time() - start) / 3:.3f} seconds")
env.close()

