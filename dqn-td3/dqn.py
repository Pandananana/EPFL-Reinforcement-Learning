# DQN implementation based on the DeepMind paper

import os
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

import gymnasium as gym

# For visualization
from gym.wrappers.monitoring import video_recorder

# Output directories (relative to where the script is run, i.e. dqn-td3/)
MODELS_DIR = "models"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"

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
        batch = random.sample(self.buffer, batch_size)
        
        # 'zip(*batch)' is magic for unzipping the list of tuples into separate lists
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
    
# The epsilon greedy policy
def select_action(state, epsilon, model, env, device):
    if (np.random.random() < epsilon):
        # Choose explorative random action
        action = env.action_space.sample()

    else:
        # Choose the greedy action
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

    return action

def train_step(model, optimizer, criterion, buffer, batch_size, gamma, device):
    # Check if we have enough that for a batch to update our approximations
    if len(buffer) < batch_size:
        return
    
    # Sample a batch from the replay buffer
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    # Convert everything to PyTorch tensors and send to device
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)

    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    
    q_values = model(states).gather(1, actions)
    
    # Bellman update
    with torch.no_grad():

        next_q_values = model(next_states)
        
        max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
        
        # (1 -dones) handles the case when the state is a terminal state (done = 1) => there is no next state!
        y = rewards + gamma * max_next_q * (1 - dones)
    
    # Optimization
    loss = criterion(q_values, y) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Running the actual algorithm
# Select the backend for the model
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found, reverting back to CPU")

device = "mps" if torch.mps.is_available() else "cpu"

# List of environments to benchmark
environments = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']

# Dictionary to store the learning curves for our final plot
benchmark_results = {}

for env_name in environments:
    print(f"\n{'='*40}")
    print(f"Initializing Benchmark: {env_name}")
    print(f"{'='*40}")
    
    # Initialize the specific environment
    env = gym.make(env_name)
    input_features = env.observation_space.shape[0]
    number_of_actions = env.action_space.n
    
    # Re-initialise everything for the specific env
    model = DQN(input_features, number_of_actions).to(device)
    buffer = ExperienceReplayBuffer(capacity=1000000)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() 
    
    # Parameter setup (more or less according to the original paper) each time a game starts
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    replay_start_size = 1000

    # Linear decay amount per step
    exploration_steps = 10000 
    epsilon_decay = (1.0 - epsilon_min) / exploration_steps 
    num_episodes = 1000
    
    episode_rewards = []

    # THE loop
    for episode in range(num_episodes):
        # Reset the environment for a new game
        state, _ = env.reset() # In gymnasium reset() returns a tuple instead of the state array, thanks Sam Altman...
        total_reward = 0
        done = False

        while not done:
            # Select and take an action based on the epsilon-greedy policy
            action = select_action(state, epsilon, model, env, device)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Evaluate done
            done = terminated or truncated

            # Store the transition in the replay buffer
            buffer.push(state, action, reward, next_state, done)

            # Perform the training step and end the episode
            # BUT only train and decay epsilon if we have pre-populated the buffer!
            if len(buffer) >= replay_start_size:
                train_step(model, optimizer, criterion, buffer, batch_size, gamma, device)

                # Decay epsilon to decrease exploration with each step
                epsilon = max(epsilon_min, epsilon - epsilon_decay)

            state = next_state
            total_reward += reward


        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}/{num_episodes} - Score: {total_reward} - Epsilon: {epsilon:.2f}")

    # Save the trained model dynamically based on the environment
    os.makedirs(MODELS_DIR, exist_ok=True)
    filename = os.path.join(MODELS_DIR, f"dqn_{env_name.lower().replace('-', '_')}_weights.pth")
    torch.save(model.state_dict(), filename)
    print(f"Saved agent to {filename}")
    
    # Store the results
    benchmark_results[env_name] = episode_rewards


# Save training data for later inspection
results_df = pd.DataFrame(benchmark_results)

os.makedirs(RESULTS_DIR, exist_ok=True)
csv_filename = os.path.join(RESULTS_DIR, "dqn_benchmark_results.csv")
results_df.to_csv(csv_filename, index_label="Episode")
print(f"Saved raw training data to {csv_filename}")

# Plot the results
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

for i, (env_name, rewards) in enumerate(benchmark_results.items()):
    # Calculate the smoothed moving average
    smoothed_rewards = pd.Series(rewards).rolling(50, min_periods=1).mean()
    
    # Noisy scores to the background
    axs[i].plot(rewards, alpha=0.2, color='gray', label='Raw Score')
    
    # Plot the smooth curve over it
    axs[i].plot(smoothed_rewards, color='blue', linewidth=2, label='50-Ep Average')
    
    # Format each individual subplot
    axs[i].set_title(f"{env_name} DQN Training Curve", fontsize=14, fontweight='bold')
    axs[i].set_xlabel("Episode")
    axs[i].set_ylabel("Total Reward")
    axs[i].legend()
    axs[i].grid(True, alpha=0.3)

# Add h_pad to explicitly enforce vertical spacing between the graphs
plt.tight_layout(h_pad=4.0) 

os.makedirs(PLOTS_DIR, exist_ok=True)
plot_path = os.path.join(PLOTS_DIR, "dqn_benchmark_results.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved benchmark plot to {plot_path}")

plt.show()