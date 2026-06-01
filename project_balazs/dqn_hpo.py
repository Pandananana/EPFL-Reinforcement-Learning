# DQN with optuna HPO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import deque
import optuna 
import json

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import gymnasium as gym

# Experience replay buffer class
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

# DQN architecture
class DQN(nn.Module):
    def __init__(self, input_features, number_of_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features= input_features, out_features= 128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=number_of_actions)

    def forward(self, x):
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.output(x)
       return x
    
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
    if len(buffer) < batch_size:
        return
    
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

# New training function make it usable for HPO
def train_agent(env_name, params, num_episodes=300, device="cpu"):
    env = gym.make(env_name)
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
        state, _ = env.reset()
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
        
        # Progress tracker
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(episode_rewards[-50:])
            print(f"    Episode {episode + 1}/{num_episodes} | Avg Score: {avg_score:.1f} | Epsilon: {epsilon:.2f}")
        
    final_score = np.mean(episode_rewards[-50:])
    env.close() 
    
    # Return the model so it can be saved it later
    return final_score, episode_rewards, model


# HPO
if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found! Using Apple Silicon.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device found! Using NVIDIA GPU.")
    else:
        device = torch.device("cpu")
        print("MPS/CUDA device not found, reverting back to CPU")

    TARGET_ENV = 'CartPole-v1' 
    NUM_TRIALS = 50      
    NUM_EPISODES = 800 
    all_trials_data = {}

    def objective(trial):
        params = {
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'gamma': trial.suggest_categorical('gamma', [0.90, 0.95, 0.99, 0.999, 0.9999]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
            'exploration_steps': trial.suggest_int('exploration_steps', 2000, 15000),
            'buffer_capacity': trial.suggest_categorical('buffer_capacity', [2000, 10000, 50000, 100000])
        }
        
        print(f"\n--- Starting Trial {trial.number} ---")
        for k, v in params.items():
            print(f"  {k}: {v}")
            
        score, learning_curve, _ = train_agent(TARGET_ENV, params, num_episodes=NUM_EPISODES, device=device)
        
        all_trials_data[trial.number] = {
            'params': params,
            'curve': learning_curve,
            'score': score
        }
        return score

    print(f"\nStarting Optuna Hyperparameter Optimization on {TARGET_ENV}...")
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=NUM_TRIALS)

    print("\n==========================================")
    print("HPO COMPLETE. Best Configuration Found:")
    print("==========================================")
    print(f"Best Score: {study.best_value:.1f}")
    
    # Save the best parameters
    best_params_filename = f"best_params_{TARGET_ENV.lower().replace('-', '_')}.json"
    with open(best_params_filename, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Saved best parameters to {best_params_filename}")

    # Train final agent and ave weights
    print(f"\nTraining final agent with best parameters to save weights...")

    final_score, final_curve, best_model = train_agent(TARGET_ENV, study.best_params, num_episodes=1000, device=device)
    
    weights_filename = f"best_dqn_{TARGET_ENV.lower().replace('-', '_')}.pth"
    torch.save(best_model.state_dict(), weights_filename)
    print(f"Saved best agent weights to {weights_filename}")
    
    # Plotting
    print("\nGenerating comparative learning curve plot...")
    plt.figure(figsize=(14, 8))
    
    sorted_trials = sorted(all_trials_data.items(), key=lambda x: x[1]['score'])
    
    for rank, (trial_num, data) in enumerate(sorted_trials):
        curve = pd.Series(data['curve']).rolling(20, min_periods=1).mean()
        
        if rank >= len(sorted_trials) - 3:
            linewidth = 3.0
            alpha = 1.0
            label = (f"Rank {len(sorted_trials) - rank} (Score: {data['score']:.1f})\n"
                     f"LR: {data['params']['lr']:.4f} | Buf: {data['params']['buffer_capacity']} | $\gamma$: {data['params']['gamma']}\n"
                     f"Batch: {data['params']['batch_size']} | Expl Steps: {data['params']['exploration_steps']}")
        else:
            linewidth = 1.0
            alpha = 0.2
            label = None 
            
        plt.plot(curve, linewidth=linewidth, alpha=alpha, label=label)

    plt.title(f"DQN Hyperparameter Optimization: {TARGET_ENV}", fontsize=18, fontweight='bold')
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Moving Average Reward (Window=20)", fontsize=14)
    
    plt.legend(loc="best", title="Top Configurations", fontsize=9, title_fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"dqn_hpo_{TARGET_ENV.lower().replace('-', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved comparative plot to {filename}")
    plt.show()