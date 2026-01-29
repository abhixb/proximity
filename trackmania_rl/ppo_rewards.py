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

def compute_enhanced_rewards(rollout_results, stored_data, num_steps):
    rewards = np.zeros(num_steps, dtype=np.float32)
    
    if rollout_results is None:
        rewards.fill(-0.001)
        return rewards
    
    zones = rollout_results.get('current_zone_idx', None)
    speeds = rollout_results.get('speed', None)
    distances_to_line = rollout_results.get('dist_to_refline', None)
    
    if zones is not None:
        if isinstance(zones, list):
            zones = np.array(zones)
    else:
        zones = np.zeros(num_steps)
    
    if speeds is not None:
        if isinstance(speeds, list):
            speeds = np.array(speeds)
    else:
        speeds = np.zeros(num_steps)
    
    if distances_to_line is not None:
        if isinstance(distances_to_line, list):
            distances_to_line = np.array(distances_to_line)
    else:
        distances_to_line = None
    
    if len(zones) < num_steps:
        zones = np.pad(zones, (0, num_steps - len(zones)), constant_values=zones[-1] if len(zones) > 0 else 0)
    if len(speeds) < num_steps:
        speeds = np.pad(speeds, (0, num_steps - len(speeds)), constant_values=0)
    if distances_to_line is not None and len(distances_to_line) < num_steps:
        distances_to_line = np.pad(distances_to_line, (0, num_steps - len(distances_to_line)), 
                                   constant_values=distances_to_line[-1] if len(distances_to_line) > 0 else 0)
    
    max_zone_so_far = 0
    steps_in_same_zone = 0
    
    for i in range(1, num_steps):
        reward = 0.0
        current_zone = int(zones[i])
        prev_zone = int(zones[i-1])
        
        zone_progress = current_zone - prev_zone
        
        if zone_progress > 0:
            base_reward = 0.05
            if current_zone < 20:
                base_reward = 0.06
            elif current_zone < 50:
                base_reward = 0.055
            
            reward += zone_progress * base_reward
            max_zone_so_far = max(max_zone_so_far, current_zone)
            steps_in_same_zone = 0
            
            if i > 1 and zones[i-2] < prev_zone:
                reward += 0.01
                
        elif zone_progress < 0:
            reward -= 0.08 * abs(zone_progress)
            
        else:
            steps_in_same_zone += 1
            reward -= 0.0002
        
        if distances_to_line is not None:
            distance = abs(distances_to_line[i])
            
            magnetic_pull = max(0, 0.004 * (1.0 - (distance / 30.0)))
            precision_reward = 0.006 * np.exp(-distance / 4.0)
            
            reward += (magnetic_pull + precision_reward)
            
            if i > 0:
                dist_prev = abs(distances_to_line[i-1])
                if distance < dist_prev:
                    improvement = dist_prev - distance
                    reward += improvement * 0.003
            
            if distance > 5.0:
                reward -= (distance - 5.0) * 0.001
        
        if current_zone >= max_zone_so_far:
            base_speed_reward = min(speeds[i] / 100.0, 0.004)
            
            if distances_to_line is not None:
                distance = abs(distances_to_line[i])
                speed_multiplier = np.exp(-distance / 4.0)
                base_speed_reward *= speed_multiplier
            
            reward += base_speed_reward
        else:
            reward -= min(speeds[i] / 200.0, 0.0005)
        
        if steps_in_same_zone > 25:
            reward -= 0.002
        if steps_in_same_zone > 60:
            reward -= 0.01
        
        rewards[i] = reward
    
    if rollout_results.get('race_finished', False):
        rewards[-1] += 5.0
        print("    RACE FINISHED")
    
    max_zone_reached = int(zones.max()) if len(zones) > 0 else 0
    if max_zone_reached > 0:
        progress_bonus = (max_zone_reached ** 1.1) * 0.005
        rewards[-1] += progress_bonus
    
    if distances_to_line is not None and max_zone_reached > 5:
        avg_distance = np.mean(np.abs(distances_to_line))
        
        if avg_distance < 3.0:
            racing_line_bonus = (3.0 - avg_distance) ** 2 * max_zone_reached * 0.002
            rewards[-1] += racing_line_bonus
            
            if avg_distance < 1.5:
                rewards[-1] += 0.5
    
    unique_zones = len(set([int(z) for z in zones]))
    if unique_zones >= 5:
        rewards[-1] += 0.05
    if unique_zones >= 10:
        rewards[-1] += 0.1
    if unique_zones >= 20:
        rewards[-1] += 0.2
    if unique_zones >= 30:
        rewards[-1] += 0.4
    if unique_zones >= 50:
        rewards[-1] += 0.8
    
    if rollout_results.get('race_finished', False):
        race_time = rollout_results.get('race_time', 999999) / 1000.0
        if race_time < 90:
            time_bonus = (90 - race_time) * 0.02
            rewards[-1] += time_bonus
    
    return rewards

def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae_lam = 0.0
    
    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
    
    return advantages, advantages + values