import os
import time
import torch
from torch import nn
import numpy as np
import gymnasium as gym

from torch import optim
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

# Visualization for the trained TD3 agent

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.output = nn.Linear(300, action_dim)
        
        # Environment specific scaler factor which corresponds to the highest value an action can take numerically
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Use tanh to bound the output to [-1, 1], then scale it with the env specific max action
        action = self.max_action * torch.tanh(self.output(x))
        return action

def evaluate_action(state, actor, device):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = actor(state_tensor).cpu().data.numpy().flatten()
    return action

def watch_agent(env_name, device, episodes=5):
    print(f"\nLoading trained agent for: {env_name}")
    
    env = gym.make(env_name, render_mode="human")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    actor = Actor(state_dim, action_dim, max_action).to(device)
    
    # Load the trained weights
    weights_path = os.path.join("models", f"td3_{env_name.lower().replace('-', '_')}_actor.pth")
    try:
        actor.load_state_dict(torch.load(weights_path, map_location=device))
        actor.eval() # Set torch to eval
        print(f"Successfully loaded {weights_path}")
    except FileNotFoundError:
        print(f"Could not find {weights_path}! Did you run the training loop?")
        return

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = evaluate_action(state, actor, device)
            
            # Take the step in the environment
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Slow down the rendering slightly so it's easier to watch
            time.sleep(0.02)
            
        print(f"Evaluation Episode {episode+1}/{episodes} - Score: {total_reward:.1f}")
        
    env.close()

# Run the visualizer
device = "mps" if torch.backends.mps.is_available() else "cpu"

watch_agent('Pendulum-v1', device)
watch_agent('MountainCarContinuous-v0', device)