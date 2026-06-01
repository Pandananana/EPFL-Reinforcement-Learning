# TD3 implementation based on the 2018 Fujimoto et al paper, may the gods of AC help us
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import deque

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import gymnasium as gym

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

# What we do here is basically Policy Iteration, so first we update our values based on the current policy (i.e. we do policy evaluation)
# Then based on those values we do the policy improvement obtaining the new policy and rinse and repeat!

# The actor (policy) network (takes a state as an input and outputs an action)
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
    
# The two critic (value) networks implemented together (they take a state-action pair as input and output the corresponding Q value)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1
        # The paper tells us to concatenate state and action at the first layer
        self.q1_fc1 = nn.Linear(state_dim + action_dim, 400)
        self.q1_fc2 = nn.Linear(400, 300)
        self.q1_output = nn.Linear(300, 1)

        # Q2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, 400)
        self.q2_fc2 = nn.Linear(400, 300)
        self.q2_output = nn.Linear(300, 1)

    def forward(self, state, action):
        # Concatenate the state and action arrays horizontally
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_output(q1)

        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_output(q2)

        return q1, q2

    def Q1(self, state, action):
        """
        A helper function that only returns Q1. 
        We will need this when updating the actor (for the delayed policy update), since the actor is 
        optimized only with respect to Q1 according to the paper.
        """
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_output(q1)
        return q1
    
def select_action(state, actor, max_action, expl_noise, device):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        # Get the action from the actor and convert it back to a numpy array
        action = actor(state_tensor).cpu().data.numpy().flatten()
    
    # Add Gaussian noise for exploration, scaled by the environment's max_action
    noise = np.random.normal(0, max_action * expl_noise, size=action.shape)
    
    # Clip the action to ensure the noise didn't push it beyond the physics limits
    action = np.clip(action + noise, -max_action, max_action)
    
    return action
    
def train_step(actor, target_actor, actor_optimizer, 
               critic, target_critic, critic_optimizer, 
               buffer, batch_size, gamma, max_action, 
               policy_noise, noise_clip, policy_freq, tau, total_iterations, device):
    
    if len(buffer) < batch_size:
        return
    
    # Sample from the buffer
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    # Convert to PyTorch tensors
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    # Training the critics i.e. doing policy evaluation based on the current actor
    # For this we use target networks as anchors which are needed because of the continuous function approximation
    # Without them, updating the value of a specific state would also alter the estimation of another state unwillingly
    # Because in this case the states are not discrete!
    with torch.no_grad():
        # Get target actions from actor_target
        next_actions = target_actor(next_states)
        
        # Create Gaussian noise, clip it, and add it to the target actions
        noise = (torch.randn_like(actions) * policy_noise).clamp(-noise_clip, noise_clip)
        next_actions = (next_actions + noise).clamp(-max_action, max_action)
        
        # Get the q value predictions from the target critic(s) and take their min
        target_q1, target_q2 = target_critic(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2) # This is q_s_t+1
        
        # Do the Bellman update (and account for terminal states by multiplying with (1 - dones))
        y = rewards + gamma * target_q * (1 - dones) # y = q_s_t

    # Get q value predictions from the critics for q_s_t
    current_q1, current_q2 = critic(states, actions)
    
    # Calculate the loss for both critics
    critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
    
    # Optimize the critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Policy iteration i.e. training the actor
    # We only update the Actor every 'policy_freq' iterations
    if total_iterations % policy_freq == 0:
        
        # We ask the critic (Q1) to grade the proposed action and based on that we mnimise the negative score
        actor_loss = -critic.Q1(states, actor(states)).mean()
        
        # Optimize the actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        # We blend the target and non-target networks by executing a soft update (Polyak averaging)
        for param, target_param in zip(critic.parameters(), target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        for param, target_param in zip(actor.parameters(), target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# THE algorithm
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Envs to benchmark
environments = ['Pendulum-v1', 'MountainCarContinuous-v0']
benchmark_results = {}

for env_name in environments:
    print(f"\n{'='*40}")
    print(f"Initializing TD3 Benchmark: {env_name}")
    print(f"{'='*40}")
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    
    # Initialise actor and target actor
    actor = Actor(state_dim, action_dim, max_action).to(device)
    target_actor = Actor(state_dim, action_dim, max_action).to(device)
    target_actor.load_state_dict(actor.state_dict()) # Ensure that the target actor is exactly the same as the OG one
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3) 
    
    # Initialise critics and target critics
    critic = Critic(state_dim, action_dim).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)
    target_critic.load_state_dict(critic.state_dict()) # Ensure that the target critic is the same as the OG one
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3) 
    
    buffer = ExperienceReplayBuffer(capacity=1000000)
    
    # Parameter definitons (according to the paper)
    batch_size = 100         
    gamma = 0.99
    tau = 0.005              
    policy_noise = 0.2       
    noise_clip = 0.5         
    policy_freq = 2          
    expl_noise = 0.1         
    replay_start_size = 10000 # Fill buffer with random actions before training begins
    num_episodes = 200       
    
    episode_rewards = []
    total_iterations = 0

    for episode in range(num_episodes):
        state, _ = env.reset() 
        total_reward = 0
        done = False

        while not done:
            total_iterations += 1
            
            # Pure random exploration for the first few steps to populate the buffer
            if len(buffer) < replay_start_size:
                action = env.action_space.sample()
            else:
                action = select_action(state, actor, max_action, expl_noise, device)
                
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)

            # Train only if we have enough data in the buffer
            if len(buffer) >= replay_start_size:
                train_step(actor, target_actor, actor_optimizer, 
                           critic, target_critic, critic_optimizer, 
                           buffer, batch_size, gamma, max_action, 
                           policy_noise, noise_clip, policy_freq, tau, total_iterations, device)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes} - Avg Score (last 10): {avg_score:.1f}")

    # Save the trained model weights dynamically
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(actor.state_dict(), os.path.join(MODELS_DIR, f"td3_{env_name.lower().replace('-', '_')}_actor.pth"))
    torch.save(critic.state_dict(), os.path.join(MODELS_DIR, f"td3_{env_name.lower().replace('-', '_')}_critic.pth"))
    print(f"Saved agent for {env_name}")
    
    benchmark_results[env_name] = episode_rewards

# Save training data for later inspection
results_df = pd.DataFrame(benchmark_results)

os.makedirs(RESULTS_DIR, exist_ok=True)
csv_filename = os.path.join(RESULTS_DIR, "td3_benchmark_results.csv")
results_df.to_csv(csv_filename, index_label="Episode")
print(f"Saved raw training data to {csv_filename}")

# Plotting the results and hoping for the best
fig, axs = plt.subplots(len(environments), 1, figsize=(10, 5 * len(environments)))

# Handle the case where there is only one environment
if len(environments) == 1:
    axs = [axs]

for i, (env_name, rewards) in enumerate(benchmark_results.items()):
    smoothed_rewards = pd.Series(rewards).rolling(10, min_periods=1).mean()
    axs[i].plot(rewards, alpha=0.2, color='gray', label='Raw Score')
    axs[i].plot(smoothed_rewards, color='blue', linewidth=2, label='10-Ep Average')
    axs[i].set_title(f"{env_name} TD3 Training Curve", fontsize=14, fontweight='bold')
    axs[i].set_xlabel("Episode")
    axs[i].set_ylabel("Total Reward")
    axs[i].legend()
    axs[i].grid(True, alpha=0.3)

plt.tight_layout()

os.makedirs(PLOTS_DIR, exist_ok=True)
plot_path = os.path.join(PLOTS_DIR, "td3_benchmark_results.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved benchmark plot to {plot_path}")
plt.show()