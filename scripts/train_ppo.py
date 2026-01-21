"""
simple_ppo_train.py - Single-file PPO trainer for Trackmania
COMPLETE VERSION: 8 Actions + Racing Line Rewards + All Fixes
"""

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

# ACTION MAPPING - 9 ACTIONS WITH BACKWARD
ALLOWED_ACTIONS = [
    2,   # 0: forward
    3,   # 1: left
    4,   # 2: right  
    5,   # 3: forward+left
    6,   # 4: forward+right
    1,   # 5: brake
    0,   # 6: backward
    2,   # 7: forward (duplicate)
    2    # 8: forward (duplicate)
]

# ============================================================================
# PPO NETWORK
# ============================================================================

class SimplePPONetwork(nn.Module):
    """PPO Network with ENHANCED float feature processing"""
    
    def __init__(self, num_actions=9, float_input_dim=184):
        super().__init__()
        
        self.float_input_dim = float_input_dim
        
        # Conv layers for visual processing
        self.conv1 = nn.Conv2d(4, 16, kernel_size=4, stride=2)  # 4 stacked frames
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        
        # ENHANCED float processing - bigger network for lookahead info
        self.float_fc1 = nn.Linear(float_input_dim, 256)  # Increased from 128
        self.float_fc2 = nn.Linear(256, 256)  # Additional layer
        
        conv_output_size = 32 * 26 * 36
        
        # Combined - give float features more weight
        self.fc_shared = nn.Linear(conv_output_size + 256, 512)  # Larger
        self.fc_shared2 = nn.Linear(512, 256)  # Additional layer
        
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
        # Process image
        if img.dtype == torch.uint8:
            img = img.float()
        img = img / 255.0
        
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        
        # ENHANCED float processing - deeper network for lookahead
        f = F.relu(self.float_fc1(float_input))
        f = F.relu(self.float_fc2(f))  # Additional processing
        
        # Combine
        combined = torch.cat([x, f], dim=1)
        shared = F.relu(self.fc_shared(combined))
        shared = F.relu(self.fc_shared2(shared))  # Additional layer
        
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

# ============================================================================
# FAST PPO INFERER
# ============================================================================

class FastPPOInferer:
    """Fast inferer that stores rollout data for learning"""
    
    def __init__(self, network, device, float_input_dim, warmup_episodes=20):
        self.network = network
        self.device = device
        self.warmup_episodes = warmup_episodes
        self.episode_count = 0
        self.step_count = 0
        self.float_input_dim = float_input_dim
        
        self.reset_rollout_storage()
        
        self.action_counts = np.zeros(9)
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
            network_action = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], 
                                             p=[0.25, 0.1, 0.1, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05])
            game_action = ALLOWED_ACTIONS[network_action]
            return (game_action, True, 0.0, np.ones(9) / 9)
        
        try:
            if obs.ndim == 2:
                obs = obs[np.newaxis, :, :]
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            float_tensor = torch.from_numpy(float_input).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                action_logits, value = self.network(obs_tensor, float_tensor)
                action_logits = action_logits / self.temperature
                
                dist = torch.distributions.Categorical(logits=action_logits)
                network_action = dist.sample()
                
                original_logits, _ = self.network(obs_tensor, float_tensor)
                original_dist = torch.distributions.Categorical(logits=original_logits)
                log_prob = original_dist.log_prob(network_action)
                
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
            return (game_action, True, 0.0, np.ones(9) / 9)
    
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
            self.temperature = max(1.0, 2.0 - (self.episode_count - self.warmup_episodes) * 0.003)
        
        if self.episode_count % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        if self.episode_count % 10 == 0:
            total = self.action_counts.sum()
            if total > 0:
                action_names = ['Fwd', 'Lft', 'Rgt', 'F+L', 'F+R', 'Brk', 'Bck', 'Fw2', 'Fw3']
                dist_str = " ".join([f"{action_names[i]}:{int(self.action_counts[i]/total*100):2d}%" for i in range(9)])
                print(f"  Actions: {dist_str} | Temp: {self.temperature:.2f}")
                self.action_counts.fill(0)

# ============================================================================
# RACING LINE REWARD COMPUTATION
# ============================================================================

def compute_enhanced_rewards(rollout_results, stored_data, num_steps):
   
    rewards = np.zeros(num_steps, dtype=np.float32)
    
    if rollout_results is None:
        rewards.fill(-5.0)
        return rewards
    
    # Get data
    zones = rollout_results.get('current_zone_idx', None)
    speeds = rollout_results.get('speed', None)
    distances_to_line = rollout_results.get('dist_to_refline', None)
    
    # Convert to arrays
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
    
    # Pad arrays
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
        
        # === 1. ZONE PROGRESS =
        zone_progress = current_zone - prev_zone
        
        if zone_progress > 0:
            # Moderate reward for zones
            base_reward = 50.0  # Reduced from 300
            if current_zone < 20:
                base_reward = 60.0  # Reduced from 300
            elif current_zone < 50:
                base_reward = 55.0  # Reduced from 240
            
            reward += zone_progress * base_reward
            max_zone_so_far = max(max_zone_so_far, current_zone)
            steps_in_same_zone = 0
            
            # Momentum bonus
            if i > 1 and zones[i-2] < prev_zone:
                reward += 15.0  # Reduced from 50
                
        elif zone_progress < 0:
            # Penalty for going backward
            reward -= 75.0 * abs(zone_progress)  # Reduced from 300
            
        else:
            # Same zone
            steps_in_same_zone += 1
            reward -= 0.2  # Small penalty
        
        # === 2. RACING LINE 
        if distances_to_line is not None:
            distance = abs(distances_to_line[i])
            
           
            racing_line_reward = 6.0 * np.exp(-distance / 3.0)
            reward += racing_line_reward
            
            # Additional penalty for being very far off
            if distance > 5.0:
                reward -= (distance - 5.0) * 1.0  # Increased penalty
        
        # === 3. SPEED 
        if current_zone >= max_zone_so_far:
            # Reward speed in new zones
            base_speed_reward = min(speeds[i] / 100.0, 4.0)  # Max +4.0
            
            # Modulate speed by racing line (go fast on line, slow off line)
            if distances_to_line is not None:
                distance = abs(distances_to_line[i])
                # On line: full speed reward
                # Off line: reduced speed reward
                speed_multiplier = np.exp(-distance / 4.0)
                base_speed_reward *= speed_multiplier
            
            reward += base_speed_reward
        else:
            # Backtracking - small penalty
            reward -= min(speeds[i] / 200.0, 0.5)
        
        # === 4. STUCK PENALTY ===
        if steps_in_same_zone > 15:
            reward -= 3.0
        if steps_in_same_zone > 40:
            reward -= 10.0
        
        rewards[i] = reward
    
    # === END-OF-EPISODE BONUSES ===
    
    if rollout_results.get('race_finished', False):
        rewards[-1] += 2000.0
        print("  RACE FINISHED!")
    
    max_zone_reached = int(zones.max()) if len(zones) > 0 else 0
    if max_zone_reached > 0:
        # Zone progress bonus (reduced importance)
        progress_bonus = (max_zone_reached ** 1.2) * 3.0
        rewards[-1] += progress_bonus
    
    # RACING LINE ADHERENCE BONUS (MAJOR)
    if distances_to_line is not None and max_zone_reached > 5:
        avg_distance = np.mean(np.abs(distances_to_line))
        
        # Big bonus for staying on line throughout episode
        if avg_distance < 3.0:
            # Quadratic bonus: closer = exponentially better
            racing_line_bonus = (3.0 - avg_distance) ** 2 * max_zone_reached * 2.0
            rewards[-1] += racing_line_bonus
            
            if avg_distance < 1.5:
                # Extra bonus for excellent line
                rewards[-1] += 200.0
    
    # Milestones
    unique_zones = len(set([int(z) for z in zones]))
    if unique_zones >= 5:
        rewards[-1] += 30.0
    if unique_zones >= 10:
        rewards[-1] += 60.0
    if unique_zones >= 20:
        rewards[-1] += 120.0
    if unique_zones >= 30:
        rewards[-1] += 240.0
    if unique_zones >= 50:
        rewards[-1] += 480.0
    
    # Time bonus
    if rollout_results.get('race_finished', False):
        race_time = rollout_results.get('race_time', 999999) / 1000.0
        if race_time < 90:
            time_bonus = (90 - race_time) * 20.0
            rewards[-1] += time_bonus
    
    return rewards

# ============================================================================
# COMPUTE GAE
# ============================================================================

def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae_lam = 0.0
    
    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
    
    return advantages, advantages + values

# ============================================================================
# PPO UPDATE
# ============================================================================

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

# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    def __init__(self, save_dir, plot_every=10):
        self.save_dir = save_dir
        self.plot_every = plot_every
        
        self.episodes = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.learning_rates = []
        self.rewards = []
        self.race_times = []
        self.finish_rates = []
        self.steps_per_episode = []
        self.value_means = []
        
        self.reward_window = deque(maxlen=20)
        self.time_window = deque(maxlen=20)
        self.finish_window = deque(maxlen=20)
        
        self.plots_dir = save_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def add_episode(self, episode, metrics, reward, race_time, finished, num_steps, lr):
        self.episodes.append(episode)
        
        if metrics is not None:
            self.policy_losses.append(metrics.get('policy_loss', 0))
            self.value_losses.append(metrics.get('value_loss', 0))
            self.entropies.append(metrics.get('entropy', 0))
            self.value_means.append(metrics.get('value_mean', 0))
        else:
            self.policy_losses.append(np.nan)
            self.value_losses.append(np.nan)
            self.entropies.append(np.nan)
            self.value_means.append(np.nan)
        
        self.learning_rates.append(lr)
        self.rewards.append(reward)
        self.race_times.append(race_time if finished else np.nan)
        self.finish_rates.append(1.0 if finished else 0.0)
        self.steps_per_episode.append(num_steps)
        
        self.reward_window.append(reward)
        if finished:
            self.time_window.append(race_time)
        self.finish_window.append(1.0 if finished else 0.0)
        
    def plot_all(self, episode):
        if len(self.episodes) < 2:
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f'Training Metrics - Episode {episode}', fontsize=16, fontweight='bold')
        
        episodes = np.array(self.episodes)
        
        # Policy Loss
        ax = axes[0, 0]
        valid_pl = np.array([v for v in self.policy_losses if not np.isnan(v)])
        if len(valid_pl) > 0:
            ax.plot(episodes[:len(self.policy_losses)], self.policy_losses, alpha=0.3, color='blue')
            if len(valid_pl) > 10:
                window = min(20, len(valid_pl) // 2)
                smoothed = np.convolve(valid_pl, np.ones(window)/window, mode='valid')
                ax.plot(episodes[:len(smoothed)], smoothed, color='blue', linewidth=2, label='Smoothed')
                ax.legend()
        ax.set_title('Policy Loss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # Value Loss
        ax = axes[0, 1]
        valid_vl = np.array([v for v in self.value_losses if not np.isnan(v)])
        if len(valid_vl) > 0:
            ax.plot(episodes[:len(self.value_losses)], self.value_losses, alpha=0.3, color='red')
            if len(valid_vl) > 10:
                window = min(20, len(valid_vl) // 2)
                smoothed = np.convolve(valid_vl, np.ones(window)/window, mode='valid')
                ax.plot(episodes[:len(smoothed)], smoothed, color='red', linewidth=2, label='Smoothed')
                ax.legend()
            ax.set_ylim(0, min(100, np.percentile(valid_vl, 95) * 1.5))
        ax.set_title('Value Loss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # Entropy
        ax = axes[0, 2]
        valid_ent = np.array([v for v in self.entropies if not np.isnan(v)])
        if len(valid_ent) > 0:
            ax.plot(episodes[:len(self.entropies)], self.entropies, alpha=0.3, color='green')
            if len(valid_ent) > 10:
                window = min(20, len(valid_ent) // 2)
                smoothed = np.convolve(valid_ent, np.ones(window)/window, mode='valid')
                ax.plot(episodes[:len(smoothed)], smoothed, color='green', linewidth=2, label='Smoothed')
                ax.legend()
        ax.set_title('Entropy')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Entropy')
        ax.grid(True, alpha=0.3)
        
        # Learning Rate
        ax = axes[1, 0]
        ax.plot(episodes, self.learning_rates, color='purple', linewidth=2)
        ax.set_title('Learning Rate')
        ax.set_xlabel('Episode')
        ax.set_ylabel('LR')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Reward
        ax = axes[1, 1]
        ax.plot(episodes, self.rewards, alpha=0.3, color='orange')
        if len(self.rewards) > 10:
            window = min(20, len(self.rewards) // 2)
            smoothed = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
            ax.plot(episodes[:len(smoothed)], smoothed, color='orange', linewidth=2, label='Smoothed')
            ax.legend()
        ax.set_title('Episode Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.grid(True, alpha=0.3)
        
        # Race Time
        ax = axes[1, 2]
        finished_times = [(e, t) for e, t in zip(episodes, self.race_times) if not np.isnan(t)]
        if finished_times:
            ep_fin, times_fin = zip(*finished_times)
            ax.scatter(ep_fin, times_fin, alpha=0.5, color='cyan', s=20)
            if len(times_fin) > 3:
                window = min(10, len(times_fin) // 2)
                smoothed = np.convolve(times_fin, np.ones(window)/window, mode='valid')
                ax.plot(ep_fin[:len(smoothed)], smoothed, color='cyan', linewidth=2, label='Smoothed')
                ax.legend()
        ax.set_title('Race Time (Finished)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Time (s)')
        ax.grid(True, alpha=0.3)
        
        # Finish Rate
        ax = axes[2, 0]
        if len(self.finish_rates) > 10:
            window = min(20, len(self.finish_rates) // 2)
            smoothed = np.convolve(self.finish_rates, np.ones(window)/window, mode='valid')
            ax.plot(episodes[:len(smoothed)], smoothed, color='magenta', linewidth=2)
        else:
            ax.plot(episodes, self.finish_rates, color='magenta', linewidth=2)
        ax.set_title('Finish Rate')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Steps per Episode
        ax = axes[2, 1]
        ax.plot(episodes, self.steps_per_episode, alpha=0.5, color='brown')
        if len(self.steps_per_episode) > 10:
            window = min(20, len(self.steps_per_episode) // 2)
            smoothed = np.convolve(self.steps_per_episode, np.ones(window)/window, mode='valid')
            ax.plot(episodes[:len(smoothed)], smoothed, color='brown', linewidth=2, label='Smoothed')
            ax.legend()
        ax.set_title('Steps per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.grid(True, alpha=0.3)
        
        # Summary
        ax = axes[2, 2]
        ax.axis('off')
        
        avg_reward = np.mean(self.reward_window) if self.reward_window else 0
        avg_time = np.mean(self.time_window) if self.time_window else 0
        avg_finish = np.mean(self.finish_window) if self.finish_window else 0
        
        latest_vl = "N/A"
        if len(self.value_losses) > 0 and not np.isnan(self.value_losses[-1]):
            latest_vl = f"{self.value_losses[-1]:.2f}"
        
        summary_text = f"""
        Rolling Averages (last 20):
        
        Avg Reward: {avg_reward:.2f}
        Avg Time: {avg_time:.2f}s
        Finish Rate: {avg_finish*100:.1f}%
        
        Latest Episode:
        Reward: {self.rewards[-1]:.2f}
        Steps: {self.steps_per_episode[-1]}
        Value Loss: {latest_vl}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        plot_path = self.plots_dir / f"training_ep{episode}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        latest_path = self.plots_dir / "latest.png"
        plt.savefig(latest_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Plots saved to {plot_path.name}")
    
    def get_summary_stats(self):
        return {
            'avg_reward': np.mean(self.reward_window) if self.reward_window else 0,
            'avg_time': np.mean(self.time_window) if self.time_window else 0,
            'finish_rate': np.mean(self.finish_window) if self.finish_window else 0,
        }

# ============================================================================
# GAME MANAGER
# ============================================================================

class RobustGameManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.tmi = None
        self.connection_attempts = 0
        self.max_attempts = 3
        
    def connect(self):
        print("Connecting to TMInterface...")
        
        for attempt in range(self.max_attempts):
            try:
                self.tmi = game_instance_manager.GameInstanceManager(
                    game_spawning_lock=None,
                    running_speed=config_copy.running_speed,
                    run_steps_per_action=config_copy.tm_engine_step_per_action,
                    max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
                    max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
                    tmi_port=config_copy.base_tmi_port,
                )
                print(f"Connected on attempt {attempt + 1}")
                self.connection_attempts = 0
                return True
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{self.max_attempts} failed: {e}")
                if attempt < self.max_attempts - 1:
                    time.sleep(3)
                else:
                    print("Failed to connect")
                    return False
    
    def safe_rollout(self, exploration_policy, map_path, zone_centers):
        if self.tmi is None:
            if not self.connect():
                return None, None
        
        try:
            rollout_results, end_race_stats = self.tmi.rollout(
                exploration_policy=exploration_policy,
                map_path=map_path,
                zone_centers=zone_centers,
                update_network=lambda: None,
            )
            return rollout_results, end_race_stats
            
        except Exception as e:
            print(f"Rollout error: {e}")
            self.connection_attempts += 1
            
            if self.connection_attempts >= 3:
                print("Too many failures, reconnecting...")
                self.tmi = None
                time.sleep(2)
                self.connect()
            
            return None, None

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 70)
   
    
    set_random_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    actual_float_dim = config_copy.float_input_dim
    print(f"Float input dim: {actual_float_dim}\n")
    
    network = SimplePPONetwork(num_actions=9, float_input_dim=actual_float_dim).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4, eps=1e-5)
    
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    print(f"Action mapping: {ALLOWED_ACTIONS}")
    print(f"Brake at action 5, Backward at action 6\n")
    
    inferer = FastPPOInferer(network, device, actual_float_dim, warmup_episodes=20)
    
    game_mgr = RobustGameManager(base_dir)
    if not game_mgr.connect():
        print("Cannot start training")
        return
    
    save_dir = base_dir / "save" / config_copy.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_tracker = MetricsTracker(save_dir, plot_every=10)
    
    set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
    map_cycle_iter = cycle(chain(*config_copy.map_cycle))
    zone_centers_filename = None
    
    print("=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)
    
    network.train()
    best_time = float('inf')
    total_updates = 0
    total_steps_trained = 0
    best_zone_ever = 0
    episodes_since_improvement = 0
    
    entropy_coef = 0.10
    min_entropy_coef = 0.02
    
    for episode in range(1, 10000):
        inferer.new_episode()
        
        next_map_tuple = next(map_cycle_iter)
        if next_map_tuple[2] != zone_centers_filename:
            try:
                zone_centers = load_next_map_zone_centers(next_map_tuple[2], base_dir)
                zone_centers_filename = next_map_tuple[2]
            except Exception as e:
                print(f"Map load error: {e}")
                continue
        
        map_name, map_path, _, is_explo, fill_buffer = next_map_tuple
        mode = "EXPLO" if is_explo else "EVAL"
        
        print(f"\n[Ep {episode}] {mode} {map_name}")
        
        rollout_results, end_race_stats = game_mgr.safe_rollout(
            exploration_policy=inferer.get_exploration_action,
            map_path=map_path,
            zone_centers=zone_centers
        )
        
        stored_data = inferer.get_stored_rollout_data()
        
        if stored_data is None or len(stored_data['obs']) < 10:
            print("No data collected")
            continue
        
        num_steps = len(stored_data['obs'])
        
        if rollout_results is not None:
            race_time = end_race_stats.get('race_time', 0) / 1000
            finished = end_race_stats.get('race_finished', False)
            
            rewards = compute_enhanced_rewards(rollout_results, stored_data, num_steps)
            
            print(f"  {num_steps} steps, {race_time:.1f}s, {'FINISH' if finished else 'DNF'}")
            print(f"  Total reward: {rewards.sum():.2f}")
            
            if 'current_zone_idx' in rollout_results:
                zones = rollout_results['current_zone_idx']
                if isinstance(zones, np.ndarray):
                    max_zone = int(zones.max())
                elif isinstance(zones, list):
                    max_zone = max([int(z) for z in zones]) if zones else 0
                else:
                    max_zone = 0
                print(f"  Max zone: {max_zone}")
                
                if max_zone > best_zone_ever:
                    best_zone_ever = max_zone
                    episodes_since_improvement = 0
                    print(f"  NEW BEST ZONE: {best_zone_ever}")
                else:
                    episodes_since_improvement += 1
            
            # Racing line stats
            if 'dist_to_refline' in rollout_results:
                distances = rollout_results['dist_to_refline']
                if isinstance(distances, list):
                    distances = np.array(distances)
                if isinstance(distances, np.ndarray) and len(distances) > 0:
                    avg_dist = np.mean(np.abs(distances))
                    max_dist = np.max(np.abs(distances))
                    print(f"  Racing line: avg={avg_dist:.2f} max={max_dist:.2f}")
        else:
            rewards = np.full(num_steps, -5.0, dtype=np.float32)
            race_time = 0
            finished = False
            print(f"  Crashed")
        
        if finished and race_time < best_time:
            best_time = race_time
            print(f"  NEW BEST TIME: {best_time:.1f}s")
        
        current_lr = optimizer.param_groups[0]['lr']
        trained_metrics = None
        
        if fill_buffer and num_steps > 50:
            try:
                last_obs = stored_data['obs'][-1]
                if last_obs.ndim == 2:
                    last_obs = last_obs[np.newaxis, :, :]
                last_obs_t = torch.from_numpy(last_obs).unsqueeze(0).float().to(device)
                last_float_t = torch.from_numpy(stored_data['floats'][-1]).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    last_value = network.get_value(last_obs_t, last_float_t).cpu().item()
                    last_value = np.clip(last_value, -300, 300)
                
                dones = np.zeros(num_steps, dtype=np.float32)
                dones[-1] = 1.0
                
                advantages, returns = compute_gae(
                    rewards, 
                    stored_data['values'], 
                    dones, 
                    last_value
                )
                
                train_data = {
                    'obs': stored_data['obs'],
                    'floats': stored_data['floats'],
                    'actions': stored_data['actions'],
                    'log_probs': stored_data['log_probs'],
                    'advantages': advantages,
                    'returns': returns,
                }
                
                global CURRENT_ENTROPY_COEF
                CURRENT_ENTROPY_COEF = entropy_coef
                
                metrics = ppo_update(network, optimizer, train_data, device)
                trained_metrics = metrics
                total_updates += 1
                total_steps_trained += num_steps
                
                if metrics['entropy'] < 0.3:
                    entropy_coef = min(entropy_coef * 1.05, 0.15)
                    print(f"  Low entropy! Increasing to {entropy_coef:.3f}")
                elif metrics['entropy'] > 1.0:
                    entropy_coef = max(entropy_coef * 0.98, min_entropy_coef)
                
                print(f"  Trained | PL:{metrics['policy_loss']:.3f} VL:{metrics['value_loss']:.2f} "
                      f"Ent:{metrics['entropy']:.3f} (coef:{entropy_coef:.3f})")
                
            except Exception as e:
                print(f"Training error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  Skipped training")
        
        metrics_tracker.add_episode(
            episode=episode,
            metrics=trained_metrics,
            reward=rewards.sum(),
            race_time=race_time,
            finished=finished,
            num_steps=num_steps,
            lr=current_lr
        )
        
        if episode % metrics_tracker.plot_every == 0:
            metrics_tracker.plot_all(episode)
            stats = metrics_tracker.get_summary_stats()
            print(f"\n  Rolling Avg: Reward={stats['avg_reward']:.1f} "
                  f"Finish={stats['finish_rate']*100:.0f}%")
            print(f"  Best Zone: {best_zone_ever} | No improvement: {episodes_since_improvement} eps")
        
        if episodes_since_improvement > 100 and episode % 10 == 0:
            print(f"  No improvement for {episodes_since_improvement} episodes (best: {best_zone_ever})")
        
        if episode % 25 == 0:
            save_path = save_dir / f"ppo_ep{episode}.pt"
            torch.save({
                'episode': episode,
                'network': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_time': best_time,
                'best_zone': best_zone_ever,
                'total_updates': total_updates,
                'total_steps_trained': total_steps_trained,
            }, save_path)
            print(f"\nSaved checkpoint")

if __name__ == "__main__":
    main()
