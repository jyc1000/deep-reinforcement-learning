import numpy as np
import torch
from collections import deque
from unityagents import UnityEnvironment
from DQNAgent import Agent
import matplotlib.pyplot as plt

# The task is episodic, and in order to solve the environment,
# your agent must get an average score of +13 over 100 consecutive episodes.

env = UnityEnvironment(file_name="Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

dqn_agent = Agent(state_size=state_size, action_size=action_size, seed=123)

visualize = False
if visualize:
    dqn_agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    epsilon = 0
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score

    while True:
        action = dqn_agent.act(state, epsilon)
        env_info = env.step(action)[brain_name]  # send the action to the environment
        reward = env_info.rewards[0]  # get the reward
        next_state = env_info.vector_observations[0]  # get the next state
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        # collect experiences(experience replay) and update Q value neural network
        # dqn_agent.step(state, action, reward, next_state, done)
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break
    print(f"Score is {score}")
    env.close()

else:
    num_episodes = 1000
    epsilon = 0.5
    eps_decay = 0.995
    eps_end = 0.005
    scores_window = deque(maxlen=100)
    scores = []
    for i_episode in range(1, num_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score

        while True:
            # action = np.random.randint(action_size)  # select an action
            action = dqn_agent.act(state, epsilon)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            reward = env_info.rewards[0]  # get the reward
            next_state = env_info.vector_observations[0]  # get the next state
            done = env_info.local_done[0]  # see if episode has finished
            score += reward  # update the score
            # collect experiences(experience replay) and update Q value neural network
            dqn_agent.step(state, action, reward, next_state, done)
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break
        epsilon = max(epsilon*eps_decay, eps_end)
        scores_window.append(score)
        scores.append(score)
        if i_episode % 100 == 0:
            print("Episode {} avg score: {}".format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(dqn_agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break


    env.close()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()




