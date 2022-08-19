import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from collections import namedtuple, deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'terminal'])
        self.seed = random.seed(seed)

    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float().to(DEVICE)
        terminals = torch.from_numpy(np.vstack([exp.terminal for exp in experiences if exp is not None])).float().to(DEVICE)

        return states, actions, rewards, next_states, terminals

    def add_memory(self, state, action, reward, next_state, terminal):
        # this will add an experience to memory
        experience = self.experience(state, action, reward, next_state, terminal)
        self.memory.append(experience)


    def __len__(self):
        return len(self.memory)