"""
simple_ppo_train.py - Single-file PPO trainer for Trackmania
FIXED: Only 6 forward actions + Enhanced reward system + Improved Architecture
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

from config_files import config_copy
from trackmania_rl.map_loader import analyze_map_cycle, load_next_map_zone_centers
from trackmania_rl.tmi_interaction import game_instance_manager
from trackmania_rl.utilities import set_random_seed

# ACTION MAPPING - Network outputs 0-5, maps to game actions
ALLOWED_ACTIONS = [2, 3, 4, 5, 6, 2]  # forward, left, right, fwd+left, fwd+right, forward

# ============================================================================
# PPO NETWORK - 6 ACTIONS ONLY - IMPROVED ARCHITECTURE
# ============================================================================

class SimplePPONetwork(nn.Module):
    """Improved PPO Actor-Critic Network with larger capacity"""
    
    def __init__(self, num_actions=6, float_input_dim=184):
        super().__init__()
        
        self.float_input_dim = float_input_dim
        
        # Deeper conv network for better feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Larger float processing
        self.float_fc1 = nn.Linear(float_input_dim, 256)
        self.float_fc2 = nn.Linear(256, 256)
        
        conv_output_size = 64 * 26 * 36
        
        # Larger shared network for complex racing behaviors
        self.fc_shared1 = nn.Linear(conv_output_size + 256, 512)
        self.fc_shared2 = nn.Linear(512, 512)
        
        # Heads
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Small gain for actor to start uniform
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        
    def forward(self, img, float_input):
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        
        # Conv processing
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        
        # Float processing with deeper network
        f = F.relu(self.float_fc1(float_input))
        f = F.relu(self.float_fc2(f))
        
        # Larger shared network
        combined = torch.cat([x, f], dim=1)
        shared = F.relu(self.fc_shared1(combined))
        shared = F.relu(self.fc_shared2(shared))
        
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
# FAST PPO INFERER - STORES DATA DURING ROLLOUT
# ============================================================================

class FastPPOInferer:
    """Fast inferer that stores rollout data for learning"""
    
    def __init__(self, network, device, float_input_dim, warmup_episodes=5):
        self.network = network
        self.device = device
        self.warmup_episodes = warmup_episodes
        self.episode_count = 0
        self.step_count = 0
        self.float_input_dim = float_input_dim
        
        # Storage for current rollout
        self.reset_rollout_storage()
        
        # Action counters
        self.action_counts = np.zeros(6)
        self.last_action = 0  # Network action 0 = game action 2 (forward)
        
    def reset_rollout_storage(self):
        """Clear rollout storage for new episode"""
        self.stored_obs = []
        self.stored_floats = []
        self.stored_actions = []
        self.stored_log_probs = []
        self.stored_values = []
        
    def get_exploration_action(self, obs, float_input):
        """FAST action selection that stores data for PPO"""
        self.step_count += 1
        
        # WARMUP: Random forward actions
        if self.episode_count < self.warmup_episodes:
            network_action = np.random.choice([0, 1, 2, 3, 4, 5], 
                                             p=[0.3, 0.15, 0.15, 0.15, 0.15, 0.1])
            game_action = ALLOWED_ACTIONS[network_action]
            return (game_action, True, 0.0, np.ones(6) / 6)
        
        try:
            # Fast preprocessing
            if obs.ndim == 2:
                obs = obs[np.newaxis, :, :]
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            float_tensor = torch.from_numpy(float_input).unsqueeze(0).float().to(self.device)
            
            # Get action and value
            with torch.no_grad():
                action_logits, value = self.network(obs_tensor, float_tensor)
                
                # NO MASKING NEEDED - only 6 actions, all forward!
                
                # Sample action
                dist = torch.distributions.Categorical(logits=action_logits)
                network_action = dist.sample()
                log_prob = dist.log_prob(network_action)
                probs = torch.softmax(action_logits, dim=-1)
            
            network_action_int = int(network_action.cpu().item())
            log_prob_float = float(log_prob.cpu().item())
            value_float = float(value.squeeze().cpu().item())
            
            # MAP to game action
            game_action = ALLOWED_ACTIONS[network_action_int]
            
            # STORE DATA (store network action 0-5)
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
            return (game_action, True, 0.0, np.ones(6) / 6)
    
    def get_stored_rollout_data(self):
        """Get stored rollout data"""
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
        
        # Clear CUDA cache periodically
        if self.episode_count % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Print action distribution
        if self.episode_count % 10 == 0:
            total = self.action_counts.sum()
            if total > 0:
                dist_str = " ".join([f"{int(self.action_counts[i]/total*100):2d}" for i in range(6)])
                print(f"  Action %: [{dist_str}]")
                self.action_counts.fill(0)

# ============================================================================
# SIMPLE REWARD COMPUTATION
# ============================================================================

def compute_simple_rewards(rollout_results, stored_data, num_steps):
    """
    SIMPLE rewards:
    1. Reward for advancing zones (going forward)
    2. That's it!
    """
    rewards = np.zeros(num_steps, dtype=np.float32)
    
    if rollout_results is not None and 'current_zone_idx' in rollout_results:
        zones = rollout_results['current_zone_idx']
        
        for i in range(1, num_steps):
            # Reward for advancing zones
            zone_progress = zones[i] - zones[i-1]
            if zone_progress > 0:
                rewards[i] = zone_progress * 10.0  # BIG reward for going forward
            elif zone_progress < 0:
                rewards[i] = zone_progress * 20.0  # BIG penalty for going backward
            else:
                rewards[i] = 0.1  # Small reward for staying alive
    else:
        # Crashed
        rewards.fill(-1.0)
    
    return rewards

# ============================================================================
# REWARD NORMALIZER
# ============================================================================

class RewardNormalizer:
    """Running normalization of rewards for stable training"""
    
    def __init__(self, epsilon=1e-8, clip_range=10.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon
        self.clip_range = clip_range
        
    def update(self, rewards):
        """Update running statistics"""
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
        
    def normalize(self, rewards):
        """Normalize rewards using running statistics"""
        normalized = (rewards - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return np.clip(normalized, -self.clip_range, self.clip_range)

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
# PPO UPDATE - IMPROVED HYPERPARAMETERS
# ============================================================================

def ppo_update(network, optimizer, rollout_data, device, epochs=6, batch_size=64):
    """Improved PPO update with better hyperparameters"""
    
    obs = torch.from_numpy(rollout_data['obs']).float().to(device)
    floats = torch.from_numpy(rollout_data['floats']).float().to(device)
    actions = torch.from_numpy(rollout_data['actions']).long().to(device)
    old_log_probs = torch.from_numpy(rollout_data['log_probs']).float().to(device)
    advantages = torch.from_numpy(rollout_data['advantages']).float().to(device)
    returns = torch.from_numpy(rollout_data['returns']).float().to(device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    num_samples = obs.shape[0]
    indices = np.arange(num_samples)
    
    metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'clipfrac': 0}
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
            
            dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions[mb_idx])
            entropy = dist.entropy().mean()
            
            # Tighter clipping for more stable updates (±10%)
            ratio = torch.exp(log_probs - old_log_probs[mb_idx])
            surr1 = ratio * advantages[mb_idx]
            surr2 = torch.clamp(ratio, 0.9, 1.1) * advantages[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Standard value loss coefficient (0.5)
            value_loss = 0.5 * ((values - returns[mb_idx]) ** 2).mean()
            
            # Combined loss with entropy bonus
            loss = policy_loss + value_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()
            
            with torch.no_grad():
                clipfrac = ((ratio - 1.0).abs() > 0.1).float().mean()
            
            metrics['policy_loss'] += policy_loss.item()
            metrics['value_loss'] += value_loss.item()
            metrics['entropy'] += entropy.item()
            metrics['clipfrac'] += clipfrac.item()
            num_updates += 1
    
    return {k: v / num_updates for k, v in metrics.items()}

# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================

class LinearLRScheduler:
    """Linear learning rate decay"""
    
    def __init__(self, optimizer, initial_lr, final_lr, total_updates):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_updates = total_updates
        self.current_update = 0
        
    def step(self):
        self.current_update += 1
        frac = min(1.0, self.current_update / self.total_updates)
        lr = self.initial_lr + (self.final_lr - self.initial_lr) * frac
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# ============================================================================
# ROBUST GAME MANAGER
# ============================================================================

class RobustGameManager:
    """Manages TMInterface with automatic reconnection"""
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.tmi = None
        self.connection_attempts = 0
        self.max_attempts = 3
        
    def connect(self):
        """Connect or reconnect to TMInterface"""
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
                print(f"✓ Connected on attempt {attempt + 1}")
                self.connection_attempts = 0
                return True
                
            except Exception as e:
                print(f"✗ Attempt {attempt + 1}/{self.max_attempts} failed: {e}")
                if attempt < self.max_attempts - 1:
                    time.sleep(3)
                else:
                    print("Failed to connect after all attempts")
                    return False
    
    def safe_rollout(self, exploration_policy, map_path, zone_centers):
        """Execute rollout with error handling"""
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
            print(f"✗ Rollout error: {e}")
            self.connection_attempts += 1
            
            # Reconnect after 3 failures
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
    print("PPO TRAINING - 6 FORWARD ACTIONS + IMPROVED ARCHITECTURE")
    print("=" * 70)
    
    set_random_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Optimize CUDA for speed
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    # Get correct float input dimension
    actual_float_dim = config_copy.float_input_dim
    print(f"Float input dim: {actual_float_dim}\n")
    
    # Create larger network with ONLY 6 ACTIONS
    network = SimplePPONetwork(num_actions=6, float_input_dim=actual_float_dim).to(device)
    
    # Lower initial learning rate with scheduling
    initial_lr = 2e-4
    final_lr = 5e-5
    total_training_updates = 5000  # Adjust based on expected training length
    
    optimizer = torch.optim.Adam(network.parameters(), lr=initial_lr, eps=1e-5)
    lr_scheduler = LinearLRScheduler(optimizer, initial_lr, final_lr, total_training_updates)
    
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    print(f"Action mapping: Network 0-5 → Game {ALLOWED_ACTIONS}")
    print(f"🚫 BACKWARD MOVEMENT IMPOSSIBLE - Only 6 forward actions")
    print(f"💰 SIMPLE REWARDS: +10 per zone forward, -20 per zone backward")
    print(f"📚 Improved: Larger network (512 units), 6 epochs, batch=64, clip=0.1")
    print(f"📉 Learning rate: {initial_lr} → {final_lr} over {total_training_updates} updates\n")
    
    # Create reward normalizer
    reward_normalizer = RewardNormalizer()
    
    # Create fast inferer
    inferer = FastPPOInferer(network, device, actual_float_dim, warmup_episodes=5)
    
    # Create robust game manager
    game_mgr = RobustGameManager(base_dir)
    if not game_mgr.connect():
        print("Cannot start training without game connection")
        return
    
    # Map cycle
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
    
    for episode in range(1, 10000):
        inferer.new_episode()
        
        # Get map
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
        
        # Execute rollout
        rollout_results, end_race_stats = game_mgr.safe_rollout(
            exploration_policy=inferer.get_exploration_action,
            map_path=map_path,
            zone_centers=zone_centers
        )
        
        # Get stored data from inferer
        stored_data = inferer.get_stored_rollout_data()
        
        if stored_data is None or len(stored_data['obs']) < 10:
            print("✗ No data collected (too short)")
            continue
        
        num_steps = len(stored_data['obs'])
        
        # COMPUTE SIMPLE REWARDS
        if rollout_results is not None:
            race_time = end_race_stats.get('race_time', 0) / 1000
            finished = end_race_stats.get('race_finished', False)
            
            # Simple reward: forward=good, backward=bad
            rewards = compute_simple_rewards(rollout_results, stored_data, num_steps)
            
            # Update and normalize rewards
            reward_normalizer.update(rewards)
            normalized_rewards = reward_normalizer.normalize(rewards)
            
            print(f"  {num_steps} steps, {race_time:.1f}s, {'FINISH' if finished else 'DNF'}")
            print(f"  💰 Total reward: {rewards.sum():.2f} (norm: {normalized_rewards.sum():.2f})")
        else:
            # CRASHED
            rewards = np.full(num_steps, -1.0, dtype=np.float32)
            normalized_rewards = reward_normalizer.normalize(rewards)
            race_time = 0
            finished = False
            print(f"  ⚠ Crashed/Incomplete rollout")
        
        if finished and race_time < best_time:
            best_time = race_time
            print(f"  🏆 NEW BEST: {best_time:.1f}s")
        
        # ALWAYS TRAIN
        if fill_buffer and num_steps > 50:
            try:
                # Get last value for bootstrapping
                last_obs = stored_data['obs'][-1]
                if last_obs.ndim == 2:
                    last_obs = last_obs[np.newaxis, :, :]
                last_obs_t = torch.from_numpy(last_obs).unsqueeze(0).float().to(device)
                last_float_t = torch.from_numpy(stored_data['floats'][-1]).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    last_value = network.get_value(last_obs_t, last_float_t).cpu().item()
                
                # Compute GAE with normalized rewards
                dones = np.zeros(num_steps, dtype=np.float32)
                dones[-1] = 1.0
                
                advantages, returns = compute_gae(
                    normalized_rewards,  # Use normalized rewards
                    stored_data['values'], 
                    dones, 
                    last_value
                )
                
                # Prepare training data
                train_data = {
                    'obs': stored_data['obs'],
                    'floats': stored_data['floats'],
                    'actions': stored_data['actions'],
                    'log_probs': stored_data['log_probs'],
                    'advantages': advantages,
                    'returns': returns,
                }
                
                # Update network
                metrics = ppo_update(network, optimizer, train_data, device)
                
                # Update learning rate
                current_lr = lr_scheduler.step()
                
                total_updates += 1
                total_steps_trained += num_steps
                
                print(f"  ✓ Trained | PL:{metrics['policy_loss']:.3f} VL:{metrics['value_loss']:.3f} "
                      f"Ent:{metrics['entropy']:.3f} LR:{current_lr:.2e}")
                
            except Exception as e:
                print(f"✗ Training error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  Skipped training (fill_buffer={fill_buffer}, steps={num_steps})")
        
        # Save checkpoint
        if episode % 25 == 0:
            save_path = base_dir / "save" / config_copy.run_name / f"ppo_ep{episode}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'episode': episode,
                'network': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_time': best_time,
                'total_updates': total_updates,
                'total_steps_trained': total_steps_trained,
                'reward_normalizer': {
                    'mean': reward_normalizer.mean,
                    'var': reward_normalizer.var,
                    'count': reward_normalizer.count,
                }
            }, save_path)
            print(f"\n💾 Saved | Ep:{episode} Updates:{total_updates} Steps:{total_steps_trained} Best:{best_time:.1f}s")

if __name__ == "__main__":
    main()
