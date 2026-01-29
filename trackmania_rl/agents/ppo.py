import sys
from pathlib import Path

# Add project to path
base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

import time 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle, chain
import gc

from collections import deque

from config_files import config_copy
from trackmania_rl.map_loader import analyze_map_cycle, load_next_map_zone_centers
from trackmania_rl.tmi_interaction import game_instance_manager
from trackmania_rl.utilities import set_random_seed

LLOWED_ACTIONS = [
    2,   # 0: forward
    3,   # 1: left
    4,   # 2: right  
    5,   # 3: forward+left
    6,   # 4: forward+right
    1,   # 5: brake
    0,   # 6: backward
]


class SimplePPONetwork(nn.Module):
    """Lightweight PPO Actor-Critic Network"""
    
    def __init__(self, num_actions=7, float_input_dim=184):  
        super().__init__()
        
        self.float_input_dim = float_input_dim
        
        # Smaller network for faster inference
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        
        # Float processing
        self.float_fc = nn.Linear(float_input_dim, 128)
        
        conv_output_size = 32 * 26 * 36
        
        # Combined
        self.fc_shared = nn.Linear(conv_output_size + 128, 256)
        
        # Heads
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        
    def forward(self, img, float_input):
        if img.dtype == torch.uint8:
            img = img.float()
        img = img / 255.0
        
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        
        f = F.relu(self.float_fc(float_input))
        
        combined = torch.cat([x, f], dim=1)
        shared = F.relu(self.fc_shared(combined))
        
        action_logits = self.actor(shared)
        value = self.critic(shared)
        
        return action_logits, value
    
    def get_action_and_value(self, img, float_input):
        action_logits, value = self.forward(img, float_input)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)
    
    def get_value(self, img, float_input):
        _, value = self.forward(img, float_input)
        return value.squeeze(-1)


class FastPPOInferer:
    
    def __init__(self, network, device, float_input_dim, warmup_episodes=20):
        self.network = network
        self.device = device
        self.warmup_episodes = warmup_episodes
        self.episode_count = 0
        self.step_count = 0
        self.float_input_dim = float_input_dim
        
        self.reset_rollout_storage()
        
        self.action_counts = np.zeros(7)  # Changed to 7
        self.last_action = 0
        self.temperature = 2.0
        
    def reset_rollout_storage(self):
        self.stored_obs = []
        self.stored_floats = []
        self.stored_actions = []
        self.stored_log_probs = []
        self.stored_values = []
        
    def get_exploration_action(self, obs, float_input):
        self.step_count += 1
        
        if self.episode_count < self.warmup_episodes:
            # Random exploration with forward bias
            network_action = np.random.choice([0, 1, 2, 3, 4, 5, 6], 
                                             p=[0.3, 0.1, 0.1, 0.2, 0.2, 0.05, 0.05])
            game_action = ALLOWED_ACTIONS[network_action]
            return (game_action, True, 0.0, np.ones(7) / 7)
        
        try:
            if obs.ndim == 2:
                obs = obs[np.newaxis, :, :]
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            float_tensor = torch.from_numpy(float_input).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                # Get raw logits from network (NO temperature during rollout!)
                action_logits, value = self.network(obs_tensor, float_tensor)
                
                # Create distribution and sample from RAW logits
                dist = torch.distributions.Categorical(logits=action_logits)
                network_action = dist.sample()
                
                # Calculate log_prob from the SAME distribution
                log_prob = dist.log_prob(network_action)
                
                # Get probabilities for diagnostics
                probs = torch.softmax(action_logits, dim=-1)
            
            network_action_int = int(network_action.cpu().item())
            log_prob_float = float(log_prob.cpu().item())
            value_float = float(value.squeeze().cpu().item())
            
            game_action = ALLOWED_ACTIONS[network_action_int]
            
            self.stored_obs.append(obs.copy())
            self.stored_floats.append(float_input.copy())
            self.stored_actions.append(network_action_int)
            self.stored_log_probs.append(log_prob_float)
            self.stored_values.append(value_float)
            
            self.action_counts[network_action_int] += 1
            self.last_action = network_action_int
            
            return (game_action, True, value_float, probs.squeeze().cpu().numpy())
            
        except Exception as e:
            print(f"    ! Inference error: {e}")
            if len(self.stored_obs) > 0:
                self.stored_obs.append(self.stored_obs[-1].copy())
                self.stored_floats.append(self.stored_floats[-1].copy())
                self.stored_actions.append(self.last_action)
                self.stored_log_probs.append(0.0)
                self.stored_values.append(0.0)
            game_action = ALLOWED_ACTIONS[self.last_action]
            return (game_action, True, 0.0, np.ones(7) / 7)
    
    def get_stored_rollout_data(self):
        if len(self.stored_obs) == 0:
            return None
            
        return {
            'obs': np.array(self.stored_obs, dtype=np.uint8),
            'floats': np.array(self.stored_floats, dtype=np.float32),
            'actions': np.array(self.stored_actions, dtype=np.int64),
            'log_probs': np.array(self.stored_log_probs, dtype=np.float32),
            'values': np.array(self.stored_values, dtype=np.float32),
        }
    
    def new_episode(self):
        self.episode_count += 1
        self.step_count = 0
        self.reset_rollout_storage()
        
        
        if self.episode_count > self.warmup_episodes:
            self.temperature = max(1.0, 2.0 - (self.episode_count - self.warmup_episodes) * 0.001)
        
        if self.episode_count % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        if self.episode_count % 10 == 0:
            total = self.action_counts.sum()
            if total > 0:
                action_names = ['Fwd', 'Lft', 'Rgt', 'F+L', 'F+R', 'Brk', 'Bck']
                dist_str = " ".join([f"{action_names[i]}:{int(self.action_counts[i]/total*100):2d}%" for i in range(7)])
                print(f"    Actions: {dist_str} | Temp: {self.temperature:.2f}")
                self.action_counts.fill(0)