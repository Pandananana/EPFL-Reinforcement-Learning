import time
import torch
import numpy as np
import gymnasium as gym

from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

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

def evaluate_action(state, model, device):
    """Selects the greedy action (epsilon = 0) based on max Q-value."""
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()
    return action

def watch_dqn_agent(env_name, device, episodes=5):
    print(f"\nLoading trained DQN agent for: {env_name}")
    
    # Initialize environment with human rendering
    env = gym.make(env_name, render_mode="human")
    
    input_features = env.observation_space.shape[0]
    number_of_actions = env.action_space.n
    
    # Recreate the DQN brain
    model = DQN(input_features, number_of_actions).to(device)
    
    # Load the trained weights from your hard drive
    weights_path = f"dqn_{env_name.lower().replace('-', '_')}_weights.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval() # Tell PyTorch we are evaluating, not training
        print(f"Successfully loaded {weights_path}")
    except FileNotFoundError:
        print(f"Could not find {weights_path}! Did you run the DQN training loop?")
        return

    # Watch the agent play!
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Ask the DQN what the best action is
            action = evaluate_action(state, model, device)
            
            # Take the step in the environment
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Slow down the rendering slightly so you can actually see what's happening
            time.sleep(0.03)
            
        print(f"Evaluation Episode {episode+1}/{episodes} - Score: {total_reward:.1f}")
        
    env.close()

if __name__ == '__main__':
    # Run the visualizer!
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Watch CartPole (Should balance the pole flawlessly for 500 steps)
    watch_dqn_agent('CartPole-v1', device)

    # Watch Acrobot (Should pump its legs to swing above the black line)
    watch_dqn_agent('Acrobot-v1', device)

    # Watch MountainCar (Might struggle depending on if it solved the sparse reward during training!)
    watch_dqn_agent('MountainCar-v0', device)