from collections import namedtuple, deque
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transform
import warnings

from collections import namedtuple, deque
from models import DQN, Duel_DQN, NoisyLinear
from memory import ReplayMemory

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LL:
    def __init__(self, state_size, action_size, seed, batch_size=64, gamma=0.99, learning_rate=1e-4,
                 buffer_size=int(1e5), n_every=4, tau=1e-3, device = DEVICE, noisy = False, dueling = False, dDQN = False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(0)
        self.batch_size = 64
        self.buffer_size = int(1e5)
        self.n_update = n_every
        self.Loss = 0
        
        #hyperparameters
        self.tau = tau
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        #RAINBOW
        self.dDQN = dDQN
        noisy = noisy
        dueling = dueling
  
        #network type
        if dueling == True:
            self.policy_net = Duel_DQN(state_size = self.state_size, action_size = self.action_size, noisy = noisy).to(DEVICE)
            self.target_net = Duel_DQN(state_size = self.state_size, action_size = self.action_size, noisy = noisy).to(DEVICE)
        else:
            self.policy_net = DQN(state_size = self.state_size, action_size = self.action_size, noisy = noisy).to(DEVICE)
            self.target_net = DQN(state_size = self.state_size, action_size = self.action_size, noisy = noisy).to(DEVICE)

        self.memory = ReplayMemory(self.action_size, self.buffer_size, self.batch_size, self.seed)

        #get dict for policy
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() #set to evalution model
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)#use adam adaptive gradient descent optimizer
        self.n_step = 0  # initialize timestep var

    def step(self, state, action, reward, next_state, terminal):
        self.memory.add_memory(state, action, reward, next_state, terminal) #add to memory
        self.n_step = (self.n_step + 1) % self.n_update #update per time steps
        if self.n_step == 0: #get experience
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.train_model(experiences, self.gamma)

    def decide_action(self, state, epsilon=0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def soft_update(self, policy_net, target_net, tau):
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
    
    def train_model(self, experiences, gamma):
        if self.dDQN == True:
            states, actions, rewards, next_states, terminal_runs = experiences
            _, double_actions = self.policy_net(next_states).detach().max(1)
            Q_net_targets = self.target_net(next_states).detach().gather(1, double_actions.unsqueeze(1))
            Q_targets = rewards + (gamma * Q_net_targets * (1 - terminal_runs))  # done = 0 for False and 1 True
            Q_pred = self.policy_net(states).gather(1, (actions.type(torch.LongTensor).to(DEVICE)))
            loss = F.mse_loss(Q_pred, Q_targets)
            self.Loss = loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.soft_update(policy_net=self.policy_net, target_net=self.target_net, tau=self.tau)
        
        else:
            states, actions, rewards, next_states, terminal_runs = experiences
            Q_net_targets = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (gamma * Q_net_targets * (1 - terminal_runs))
            Q_pred = self.policy_net(states).gather(1, (actions.type(torch.LongTensor)).to(DEVICE))
            loss = F.mse_loss(Q_pred, Q_targets)
            self.Loss = loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.soft_update(policy_net=self.policy_net, target_net=self.target_net, tau=self.tau)

    def get_Loss(self):
        return self.Loss