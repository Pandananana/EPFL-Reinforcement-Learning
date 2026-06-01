# DQN implementation based on the DeepMind paper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import deque


import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import gym

# For visualization
from gym.wrappers.monitoring import video_recorder

# Experience replay buffer class
class ExperienceReplayBuffer:
    def __init__(self, capacity):
        # deque automatically acts as a FIFO queue when maxlen is reached
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition to the replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly samples a batch of transitions."""
        # random.sample picks unique transitions without replacement
        batch = random.sample(self.buffer, batch_size)
        
        # 'zip(*batch)' elegantly unzips the list of tuples into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)

# DQN architecture (input: state, output: all q values associated with that state)
class DQN(nn.Module):
    def __init__(self, input_features, number_of_actions):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(in_features= input_features, out_features= 128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=number_of_actions)

    def forward(self, x):
       """
       Define the forward pass of the neural network.

       Parameters:
           x: Input tensor.

       Returns:
           torch.Tensor
               The output tensor after passing through the network.
       """
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.output(x)

       return x
    
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found, reverting back to CPU")

device = "mps" if torch.mps.is_available() else "cpu"
# device = "mps" if torch.backends.mps.is_available() else "cpu"
model = DQN(in_channels=16, num_classes=4).to(device)

env = gym.make('CartPole-v0')
env.seed(0)

# DQN algorithm loop
buffer = ExperienceReplayBuffer(capacity=1000000)
q_values = []

def select_action(state, epsilon):
    if (np.random.random() < epsilon):
        # Choose explorative random action
        action = env.action_space.sample()

    else:
        # Choose the greedy action
        action = np.argmax(q_values)
    return action

num_episodes=1000
for e in range(num_episodes):
 # Iterate over training batches
   print(f"Episode [{e + 1}/{num_episodes}]")

   for batch_index, (data, targets) in enumerate(tqdm(dataloader_train)):
       data = data.to(device)
       targets = targets.to(device)
       scores = model(data)
       loss = criterion(scores, targets)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()