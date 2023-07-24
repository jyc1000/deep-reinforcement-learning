from Model1 import Actor
from Model1 import Critic
import random
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import copy

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 100
GAMMA = 0.99
LR_ACTOR = 0.003
LR_CRITIC = 0.003
TAU = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.optimizer_critic = torch.optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.i_step = 0

        self.noise = OUNoise(self.action_size, random_seed)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.i_step = (self.i_step + 1) % 4
        if self.i_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def learn(self, experience, gamma):
        state, action, reward, next_state, done = experience
        # update critic
        # get target Q value
        next_action = self.actor_target(next_state)
        target_Q = reward + gamma * self.critic_target.forward(next_state, next_action) * (1 - done)

        local_Q = self.critic_local.forward(state, action)

        objective_critic = F.mse_loss(target_Q, local_Q)

        self.optimizer_critic.zero_grad()
        objective_critic.backward()
        self.optimizer_critic.step()

        # update actor

        actor_action = self.actor_local(state)
        objective_actor = -self.critic_local.forward(state, actor_action).mean()

        self.optimizer_actor.zero_grad()
        objective_actor.backward()
        self.optimizer_actor.step()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

