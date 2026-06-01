import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import deque
import os

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import gymnasium as gym

# ==========================================
# 1. CORE DQN COMPONENTS
# ==========================================
class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_features, number_of_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_features, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=number_of_actions)
    def forward(self, x):
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       return self.output(x)
    
def select_action(state, epsilon, model, env, device):
    if (np.random.random() < epsilon):
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
    return action

def train_step(model, optimizer, criterion, buffer, batch_size, gamma, device):
    if len(buffer) < batch_size: return
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    
    q_values = model(states).gather(1, actions)
    with torch.no_grad():
        next_q_values = model(next_states)
        max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
        y = rewards + gamma * max_next_q * (1 - dones)
        
    loss = criterion(q_values, y) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ==========================================
# 2. ENCAPSULATED TRAINING FUNCTION
# ==========================================
def train_agent(env_name, params, num_episodes=300, device="cpu"):
    """Trains a single agent and returns the learning curve."""
    env = gym.make(env_name)
    
    # Using a fixed seed ensures the comparison is purely based on hyperparameters,
    # not on one agent getting a "lucky" random start.
    seed = 42 
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
        
    input_features = env.observation_space.shape[0]
    number_of_actions = env.action_space.n
    
    model = DQN(input_features, number_of_actions).to(device)
    buffer = ExperienceReplayBuffer(capacity=params['buffer_capacity'])
    optimizer = optim.RMSprop(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss() 
    
    batch_size = params['batch_size']
    gamma = params['gamma']
    epsilon = 1.0
    epsilon_min = 0.1
    replay_start_size = 1000
    epsilon_decay = (1.0 - epsilon_min) / params['exploration_steps']
    
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed)
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, epsilon, model, env, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)

            if len(buffer) >= replay_start_size:
                train_step(model, optimizer, criterion, buffer, batch_size, gamma, device)
                epsilon = max(epsilon_min, epsilon - epsilon_decay)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(episode_rewards[-50:])
            print(f"    Episode {episode + 1}/{num_episodes} | Epsilon: {epsilon:.2f} | Avg Score (last 50): {avg_score:.1f}")

    env.close() 
    return episode_rewards, model

# ==========================================
# 3. LOAD JSONs, EVALUATE, AND PLOT
# ==========================================
if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    TARGET_ENV = 'CartPole-v1' 
    NUM_EPISODES = 1000 
    
    # 1. Define the 3 JSON files to load
    json_files = {
        "Agent 1": "agent1_params.json",
        "Agent 2": "agent2_params.json",
        "Agent 3": "agent3_params.json"
    }
    
    agents_to_test = {}
    
    # Load all 3 files into our dictionary
    print("\nLoading Hyperparameters...")
    for agent_name, filename in json_files.items():
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing {filename}! Please ensure all 3 JSON files are in the same folder as this script.")
            
        with open(filename, "r") as f:
            agents_to_test[agent_name] = json.load(f)
            print(f"  [OK] Loaded {filename} for {agent_name}")

    # Dictionary to hold the raw learning curves for plotting
    all_agents_results = {}

    # 2. Training Loop (One run per agent)
    for agent_name, params in agents_to_test.items():
        print(f"\n==========================================")
        print(f"Evaluating: {agent_name}")
        print(f"Params: LR={params['lr']:.4f}, Buf={params['buffer_capacity']}, Gamma={params['gamma']}")
        print(f"==========================================")
        
        learning_curve, trained_model = train_agent(
            TARGET_ENV, params, num_episodes=NUM_EPISODES, device=device
        )
        
        all_agents_results[agent_name] = learning_curve

    # 3. Generate the Comparative Plot
    print("\nGenerating comparative plot...")
    plt.figure(figsize=(12, 7))
    
    # Define distinct colors for the 3 agents
    colors = ['blue', 'red', 'green']
    
    for (agent_name, curve), color in zip(all_agents_results.items(), colors):
        # Apply a rolling average to smooth the plot lines
        smoothed_curve = pd.Series(curve).rolling(20, min_periods=1).mean()

        # Optional: Plot the raw, noisy scores slightly faded in the background
        plt.plot(curve, color=color, alpha=0.15)
        
        # Plot the thick Smoothed Line
        plt.plot(smoothed_curve, color=color, linewidth=2.5, label=f'{agent_name}')

    plt.title(f"DQN Performance Comparison on {TARGET_ENV}", fontsize=16, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward (Smoothed Window=20)", fontsize=12)
    
    # Place legend outside so it doesn't cover up the curves
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save and show
    os.makedirs("plots", exist_ok=True)
    plot_filename = os.path.join("plots", f"comparative_eval_{TARGET_ENV.lower().replace('-', '_')}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved comparative evaluation plot to {plot_filename}")
    
    plt.show()