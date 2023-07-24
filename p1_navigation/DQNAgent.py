from Network import Policy
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque, namedtuple

# LR = 0.003
# UPDATE_EVERY = 4
# BATCH_SIZE = 25
# BUFFER_SIZE = 10000
# GAMMA = 0.99
# TAU = 0.2

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_size, action_size, seed):
        self.action_size = action_size
        self.state_size = state_size
        self.qnetwork_local = Policy(state_size, action_size, seed).to(device)
        self.qnetwork_target = Policy(state_size, action_size, seed).to(device)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.i_step = 0
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.seed = random.seed(seed)

    def act(self, state, epsilon):
        if random.random() < epsilon:
            action = random.randint(0, self.action_size-1)
        else:
            self.qnetwork_local.eval()
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = self.qnetwork_local.forward(state)
            action = np.argmax(action_values.cpu().data.numpy())
            self.qnetwork_local.train()
        return action

    def step(self, state, action, reward, next_state, done):
        # store state, action, reward, next_state tuple for experience replay
        self.memory.store(state, action, reward, next_state, done)
        self.i_step = (self.i_step + 1) % UPDATE_EVERY
        if self.i_step == 0:
            # sample from the experience replay container and update parameters of network
            if len(self.memory) >= BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                # update the target network
                self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def learn(self, experiences, GAMMA):
        states, actions, rewards, next_states, dones = experiences
        qval_curr = self.qnetwork_local.forward(states).gather(1, actions)
        qval_next= torch.max(self.qnetwork_target.forward(next_states).detach(), dim=1)[0].unsqueeze(1)
        target = rewards + (GAMMA * qval_next) * (1 - dones)

        loss = F.mse_loss(qval_curr, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, dqn_curr, dqn_target, tau):
        for curr_param, target_param in zip(dqn_curr.parameters(), dqn_target.parameters()):
            target_param.data.copy_(target_param.data * (1-tau) + curr_param.data * tau)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def store(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


