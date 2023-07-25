from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent
from collections import deque
import torch
import time
import matplotlib.pyplot as plt

# Parameters
# env = UnityEnvironment(file_name='Reacher.app')
env = UnityEnvironment(file_name='Reacher_Windows_x86_64_multiple/Reacher.exe')
num_episodes = 1000
target_score = 30.0
random_seed = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
continue_learning = False

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

ddpg_agent = Agent(state_size, action_size, random_seed, num_agents, device, continue_learning)
scores_window = deque(maxlen=100)
scores = []

for i_episode in range(1, num_episodes+1):
    start = time.time()
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    ddpg_agent.reset()
    score = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = ddpg_agent.act(states)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        ddpg_agent.step(states, actions, rewards, next_states, dones)
        score += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break

    scores_window.append(np.mean(score))
    scores.append(np.mean(score))

    print(f'Episode {i_episode} score: {np.mean(score) :.3f}. Average score: {np.mean(scores_window):.3f}. '
          f'Device: {device}. Time per episode: {time.time()-start :.1f} seconds.')

    if np.mean(scores_window) >= target_score:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                     np.mean(scores_window)))
        torch.save(ddpg_agent.actor_local.state_dict(), 'actor_local.pth')
        torch.save(ddpg_agent.critic_local.state_dict(), 'critic_local.pth')
        torch.save(ddpg_agent.actor_target.state_dict(), 'actor_target.pth')
        torch.save(ddpg_agent.critic_target.state_dict(), 'critic_target.pth')
        break

env.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

