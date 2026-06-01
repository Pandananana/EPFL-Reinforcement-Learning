# TD3 implementation with optuna HPO
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

# TD3 as before
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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.output = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.output(x))
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, 400)
        self.q1_fc2 = nn.Linear(400, 300)
        self.q1_output = nn.Linear(300, 1)
        # Q2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, 400)
        self.q2_fc2 = nn.Linear(400, 300)
        self.q2_output = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1_output(F.relu(self.q1_fc2(F.relu(self.q1_fc1(sa)))))
        q2 = self.q2_output(F.relu(self.q2_fc2(F.relu(self.q2_fc1(sa)))))
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1_output(F.relu(self.q1_fc2(F.relu(self.q1_fc1(sa)))))
    
def select_action(state, actor, max_action, expl_noise, device):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = actor(state_tensor).cpu().data.numpy().flatten()
    noise = np.random.normal(0, max_action * expl_noise, size=action.shape)
    return np.clip(action + noise, -max_action, max_action)
    
def train_step(actor, target_actor, actor_optimizer, 
               critic, target_critic, critic_optimizer, 
               buffer, batch_size, gamma, max_action, 
               policy_noise, noise_clip, policy_freq, tau, total_iterations, device):
    if len(buffer) < batch_size: return
    
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        next_actions = target_actor(next_states)
        noise = (torch.randn_like(actions) * policy_noise).clamp(-noise_clip, noise_clip)
        next_actions = (next_actions + noise).clamp(-max_action, max_action)
        target_q1, target_q2 = target_critic(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2) 
        y = rewards + gamma * target_q * (1 - dones) 

    current_q1, current_q2 = critic(states, actions)
    critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    if total_iterations % policy_freq == 0:
        actor_loss = -critic.Q1(states, actor(states)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        for param, target_param in zip(critic.parameters(), target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(actor.parameters(), target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Training functionalities
def train_agent(env_name, params, num_episodes=100, device="cpu"):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    
    # Initialize networks
    actor = Actor(state_dim, action_dim, max_action).to(device)
    target_actor = Actor(state_dim, action_dim, max_action).to(device)
    target_actor.load_state_dict(actor.state_dict())
    actor_optimizer = optim.Adam(actor.parameters(), lr=params['lr_actor']) 
    
    critic = Critic(state_dim, action_dim).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())
    critic_optimizer = optim.Adam(critic.parameters(), lr=params['lr_critic']) 
    
    buffer = ExperienceReplayBuffer(capacity=params['buffer_capacity'])
    
    # Extract HPO parameters
    batch_size = params['batch_size']
    gamma = params['gamma']
    tau = params['tau']
    policy_noise = params['policy_noise']
    expl_noise = params['expl_noise']
    
    # For safety
    noise_clip = 0.5         
    policy_freq = 2          
    replay_start_size = 10000 
    
    episode_rewards = []
    total_iterations = 0

    for episode in range(num_episodes):
        state, _ = env.reset() 
        total_reward = 0
        done = False

        while not done:
            total_iterations += 1
            if len(buffer) < replay_start_size:
                action = env.action_space.sample()
            else:
                action = select_action(state, actor, max_action, expl_noise, device)
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)

            if len(buffer) >= replay_start_size:
                train_step(actor, target_actor, actor_optimizer, 
                           critic, target_critic, critic_optimizer, 
                           buffer, batch_size, gamma, max_action, 
                           policy_noise, noise_clip, policy_freq, tau, total_iterations, device)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(episode_rewards[-10:])
            print(f"    Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.1f}")

    # Use the last 20 episodes to evaluate asymptotic performance
    final_score = np.mean(episode_rewards[-20:])
    env.close()
    
    # Return actor so we can save the model weights
    return final_score, episode_rewards, actor

# Optuna magic
if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    print(f"Using device: {device}")

    TARGET_ENV = 'MountainCarContinuous-v0' 
    NUM_TRIALS = 15      
    NUM_EPISODES = 160
    
    all_trials_data = {}

    def objective(trial):
        # The TD3 Search Space
        params = {
            'lr_actor': trial.suggest_float('lr_actor', 1e-4, 3e-3, log=True),
            'lr_critic': trial.suggest_float('lr_critic', 1e-4, 3e-3, log=True),
            'gamma': trial.suggest_categorical('gamma', [0.99, 0.999, 0.9999]),
            'tau': trial.suggest_categorical('tau', [0.005, 0.01, 0.05]),
            'batch_size': trial.suggest_categorical('batch_size', [100, 256]),
            'buffer_capacity': trial.suggest_categorical('buffer_capacity', [50000, 100000]),
            
            'policy_noise': trial.suggest_categorical('policy_noise', [0.1, 0.2, 0.3]),
            'expl_noise': trial.suggest_categorical('expl_noise', [0.1, 0.2, 0.3, 0.5]) # High values for MountainCar!
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
    
    # Save the parameters
    best_params_filename = f"td3_best_params_{TARGET_ENV.lower().replace('-', '_')}.json"
    with open(best_params_filename, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Saved best parameters to {best_params_filename}")

    # Retrain best agent to save weights
    print(f"\nTraining final agent with best parameters to save weights...")
    final_score, final_curve, best_actor = train_agent(TARGET_ENV, study.best_params, num_episodes=150, device=device)
    
    weights_filename = f"td3_best_actor_{TARGET_ENV.lower().replace('-', '_')}.pth"
    torch.save(best_actor.state_dict(), weights_filename)
    print(f"Saved best actor weights to {weights_filename}")
    
    # Plotting
    print("\nGenerating comparative learning curve plot...")
    plt.figure(figsize=(14, 8))
    
    sorted_trials = sorted(all_trials_data.items(), key=lambda x: x[1]['score'])
    
    for rank, (trial_num, data) in enumerate(sorted_trials):
        curve = pd.Series(data['curve']).rolling(10, min_periods=1).mean()
        
        if rank >= len(sorted_trials) - 3:
            linewidth = 3.0
            alpha = 1.0
            label = (f"Rank {len(sorted_trials) - rank} (Score: {data['score']:.1f})\n"
                     f"Actor LR: {data['params']['lr_actor']:.4f} | Expl Noise: {data['params']['expl_noise']}\n"
                     f"$\gamma$: {data['params']['gamma']} | Tau: {data['params']['tau']}")
        else:
            linewidth = 1.0
            alpha = 0.2
            label = None 
            
        plt.plot(curve, linewidth=linewidth, alpha=alpha, label=label)

    plt.title(f"TD3 Hyperparameter Optimization: {TARGET_ENV}", fontsize=18, fontweight='bold')
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Moving Average Reward (Window=10)", fontsize=14)
    
    plt.legend(loc="lower right", title="Top 3 Configurations", fontsize=10, title_fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # According to the env
    plt.ylim(-100, 100) 
    
    plt.tight_layout()
    
    filename = f"td3_hpo_{TARGET_ENV.lower().replace('-', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved comparative plot to {filename}")
    plt.show()