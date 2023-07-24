from mlagents_envs.environment import UnityEnvironment
import numpy as np

from DQNAgent import Agent

dqn_agent = Agent()

print("before env")
env = UnityEnvironment(file_name="Banana_Windows_x86_64")
print("after env")
# get the default brain
brain_name = env.brain_names[0]
print("brain_name")
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

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]  # get the current state
score = 0  # initialize the score
while True:
    print('in loop')
    # action = np.random.randint(action_size)  # select an action
    action = dqn_agent.get_action(state)
    env_info = env.step(action)[brain_name]  # send the action to the environment
    next_state = env_info.vector_observations[0]  # get the next state
    reward = env_info.rewards[0]  # get the reward
    done = env_info.local_done[0]  # see if episode has finished
    # Update Q-value
    dqn_agent.update_Q(state, action, next_state, reward, done)
    score += reward  # update the score
    state = next_state  # roll over the state to next time step
    if done:  # exit loop if episode finished
        break

print("Score: {}".format(score))

env.close()
