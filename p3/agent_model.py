# Impala CNN. Same as that used in procgen papers except we've added batchnorm. Each module tested separately during construction.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=1):
        super(Agent, self).__init__()

        #actor neural network
        self.a_linear1 = nn.Linear(num_inputs, hidden_size)
        self.a_linear2 = nn.Linear(hidden_size, hidden_size)
        #self.a_linear3 = nn.Linear(hidden_size, hidden_size)
        self.a_linear3 = nn.Linear(hidden_size, num_outputs)
        
        #critic neural network
        self.c_linear1 = nn.Linear(num_inputs, hidden_size)
        self.c_linear2 = nn.Linear(hidden_size, hidden_size)
        #self.c_linear3 = nn.Linear(hidden_size, hidden_size)
        self.c_linear3 = nn.Linear(hidden_size, 1)
        
        #variance
        self.std=std

        
    def forward(self, x, action=None):
        
        #state propagation
        a=F.relu(self.a_linear1(x))
        a=F.relu(self.a_linear2(a))
        #a=F.relu(self.a_linear3(a))
        a=F.tanh(self.a_linear3(a))
        
        c=F.relu(self.c_linear1(x))
        c=F.relu(self.c_linear2(c))
        #c=F.relu(self.c_linear3(c))
        c=self.c_linear3(c)

          
            
        dist = torch.distributions.Normal(a, self.std) #generating normal distribution with mean a and variance self.std
        
        if action==None:
            action = dist.sample() #generate random action
            
        log_prob=dist.log_prob(action) #calculate action log propability with respect to the density function for each action coordinate
        log_prob = torch.sum(log_prob, dim=1, keepdim=True) #average probabilities over 2 coordinates
        
        
        return action, log_prob, c
    
