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
import matplotlib.pyplot as plt
from collections import deque

from config_files import config_copy
from trackmania_rl.map_loader import analyze_map_cycle, load_next_map_zone_centers
from trackmania_rl.tmi_interaction import game_instance_manager
from trackmania_rl.utilities import set_random_seed

CURRENT_ENTROPY_COEF = 0.10

def ppo_update(network, optimizer, rollout_data, device, epochs=4, batch_size=256):
    global CURRENT_ENTROPY_COEF
    
    obs = torch.from_numpy(rollout_data['obs']).to(device)
    floats = torch.from_numpy(rollout_data['floats']).float().to(device)
    actions = torch.from_numpy(rollout_data['actions']).long().to(device)
    old_log_probs = torch.from_numpy(rollout_data['log_probs']).float().to(device)
    advantages = torch.from_numpy(rollout_data['advantages']).float().to(device)
    returns = torch.from_numpy(rollout_data['returns']).float().to(device)
    
    returns = torch.clamp(returns, -300, 300)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    num_samples = obs.shape[0]
    indices = np.arange(num_samples)
    
    metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'clipfrac': 0, 'value_mean': 0}
    num_updates = 0
    
    for epoch in range(epochs):
        np.random.shuffle(indices)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            mb_idx = indices[start:end]
            
            mb_obs = obs[mb_idx]
            if mb_obs.ndim == 3:
                mb_obs = mb_obs.unsqueeze(1)
            
            action_logits, values = network(mb_obs, floats[mb_idx])
            values = values.squeeze(-1)
            values = torch.clamp(values, -300, 300)
            
            dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions[mb_idx])
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(log_probs - old_log_probs[mb_idx])
            surr1 = ratio * advantages[mb_idx]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.smooth_l1_loss(values, returns[mb_idx])
            loss = policy_loss + 0.5 * value_loss - CURRENT_ENTROPY_COEF * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()
            
            with torch.no_grad():
                clipfrac = ((ratio - 1.0).abs() > 0.2).float().mean()
            
            metrics['policy_loss'] += policy_loss.item()
            metrics['value_loss'] += value_loss.item()
            metrics['entropy'] += entropy.item()
            metrics['clipfrac'] += clipfrac.item()
            metrics['value_mean'] += values.mean().item()
            num_updates += 1
    
    return {k: v / num_updates for k, v in metrics.items()}
