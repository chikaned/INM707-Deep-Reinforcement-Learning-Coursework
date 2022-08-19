import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class NoisyLinear(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(NoisyLinear, self).__init__()

        #class vars
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.sigma_init = 0.5
        self.weight_mu = nn.Parameter(torch.FloatTensor(output_dims, input_dims))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_dims, input_dims))
        self.register_buffer('weight_epsilon', torch.FloatTensor(output_dims, input_dims))
        self.bias_mu = nn.Parameter(torch.FloatTensor(output_dims))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_dims))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_dims))
        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.input_dims)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.input_dims))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.output_dims))  #output_dims

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.input_dims)
        epsilon_j = self.scale_noise(self.output_dims)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        x = torch.randn(size)  # torch.randn
        x = x.sign().mul(x.abs().sqrt())
        return x


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=64, noisy = False):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        if noisy:
            self.out = NoisyLinear(hidden_dim, action_size)
        else:
            self.out = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.out(x)
        return action


class Duel_DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim = 64, noisy = False):
        super(Duel_DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        if noisy:
            self.V = NoisyLinear(hidden_dim, 1)
            self.A = NoisyLinear(hidden_dim, action_size)
        else:
            self.V = nn.Linear(hidden_dim, 1)
            self.A = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        V = self.V(x) 
        A = self.A(x)  
        action = V + (A - torch.mean(A, dim=-1, keepdim=True))  
        return action