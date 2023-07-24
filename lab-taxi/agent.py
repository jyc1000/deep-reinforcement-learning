import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.005
        self.similar_states = dict()

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # epsilon-greedy algorithm
        probs = np.ones(self.nA) * self.epsilon/self.nA
        if state in self.Q:
            max_action = np.argmax(self.Q[state])
        else:
            max_action = np.random.choice(np.arange(self.nA))
        probs[max_action] = 1 - self.epsilon + self.epsilon/self.nA
        action = np.random.choice(np.arange(self.nA), p=probs)

        return action

    def step(self, state, action, reward, next_state, terminated, truncated):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        # Expected SARSA
        alpha = 1
        curr_Q = self.Q[state][action]
        if terminated or truncated:
            self.Q[state][action] = curr_Q + alpha * (reward + 0 - curr_Q)
        else:
            probs = np.ones(self.nA) * self.epsilon/self.nA
            if next_state in self.Q:
                max_action = np.argmax(self.Q[next_state])
            else:
                max_action = np.random.choice(np.arange(self.nA))
            probs[max_action] = 1 - self.epsilon + self.epsilon / self.nA

            self.Q[state][action] = curr_Q + alpha*(reward + np.dot(self.Q[next_state], probs) - curr_Q)

        # Using knowledge of world, update similar state
