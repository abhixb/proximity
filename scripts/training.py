"""
simple_ppo_train.py - Single-file PPO trainer for Trackmania
FIXED: Learns from EVERY rollout, even partial/crashed ones
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

# Import action names for debugging
try:
    from config_files.inputs_list import keyboard_actions_list
    ACTION_NAMES = [str(action) for action in keyboard_actions_list]
    print("Action mapping loaded:")
    for i, name in enumerate(ACTION_NAMES):
        print(f"  Action {i}: {name}")
    print()
except:
    ACTION_NAMES = [f"Action_{i}" for i in range(12)]

# ============================================================================
# PPO NETWORK
# ============================================================================

class SimplePPONetwork(nn.Module):
    """Lightweight PPO Actor-Critic Network"""
    
    def __init__(self, num_actions=12, float_input_dim=184):
        super().__init__()
        
        self.float_input_dim = float_input_dim
        
        # Smaller network for faster inference
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        
        # Float processing
        self.float_fc = nn.Linear(float_input_dim, 128)
        
        # Calculate conv output size dynamically
        # After conv layers: (120-4)/2+1 = 59, (160-4)/2+1 = 79
        #                   (59-4)/2+1 = 28, (79-4)/2+1 = 38
        #                   (28-3)/1+1 = 26, (38-3)/1+1 = 36
        # conv_out = 32 * 26 * 36 = 29952
        conv_output_size = 32 * 26 * 36
        
        # Combined (smaller to speed up)
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
        
        # Small gain for actor to start uniform
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        
    def forward(self, img, float_input):
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        
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
        self.action_counts = np.zeros(12)
        self.last_action = 2  # Start with forward
        
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
            action = np.random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
                                     p=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
            return (action, True, 0.0, np.ones(12) / 12)
        
        try:
            # Fast preprocessing
            if obs.ndim == 2:
                obs = obs[np.newaxis, :, :]
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            float_tensor = torch.from_numpy(float_input).unsqueeze(0).float().to(self.device)
            
            # Get action and value (STORE THIS TIME!)
            with torch.no_grad():
                action_logits, value = self.network(obs_tensor, float_tensor)
                
                # DEBUG: Print raw logits on first few steps
                if self.step_count <= 3:
                    logits_np = action_logits.squeeze().cpu().numpy()
                    print(f"  [Step {self.step_count}] Raw logits: {logits_np.round(2)}")
                
                # AGGRESSIVE ACTION MASKING - EXPANDED
                # Based on debug output: Action 7 is causing backward movement!
                
                # Action 0: Usually "no input" - mask at start
                if self.step_count < 100:
                    action_logits[:, 0] = -1e9
                
                # MASK ALL BACKWARD/BRAKE ACTIONS
                # Typical TrackMania action mapping:
                # 0=none, 1=down, 2=up, 3=left, 4=right, 5=up+left, 6=up+right
                # 7=down+left, 8=down+right, 9=left+brake, 10=right+brake, etc.
                
                backward_actions = [1, 7, 8]  # down, down+left, down+right
                for action_idx in backward_actions:
                    action_logits[:, action_idx] = -1e9
                
                # Also mask pure brake actions if they exist (usually 9, 10, 11)
                # Uncomment if still having issues:
                # action_logits[:, 9] = -1e9
                # action_logits[:, 10] = -1e9
                # action_logits[:, 11] = -1e9
                
                # DEBUG: Print masked logits
                if self.step_count <= 3:
                    masked_logits = action_logits.squeeze().cpu().numpy()
                    print(f"  [Step {self.step_count}] Masked logits: {masked_logits.round(2)}")
                
                # Sample action and get log_prob
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                probs = torch.softmax(action_logits, dim=-1)
                
                # DEBUG: Print selected action
                if self.step_count <= 3:
                    print(f"  [Step {self.step_count}] Selected action: {action.item()}")
                    print(f"  [Step {self.step_count}] Probs: {probs.squeeze().cpu().numpy().round(3)}")
            
            action_int = int(action.cpu().item())
            
            # HARD CHECK: If somehow backward actions got selected, override them
            backward_actions = [1, 7, 8]  # down, down+left, down+right
            if action_int in backward_actions:
                print(f"  ❌ EMERGENCY: Action {action_int} (backward) was selected! Overriding to 2 (forward)")
                action_int = 2  # Force forward instead
                log_prob_float = 0.0
            else:
                log_prob_float = float(log_prob.cpu().item())
            value_float = float(value.squeeze().cpu().item())
            
            # STORE DATA FOR TRAINING
            self.stored_obs.append(obs.copy())
            self.stored_floats.append(float_input.copy())
            self.stored_actions.append(action_int)
            self.stored_log_probs.append(log_prob_float)
            self.stored_values.append(value_float)
            
            self.action_counts[action_int] += 1
            self.last_action = action_int
            
            return (action_int, True, value_float, probs.squeeze().cpu().numpy())
            
        except Exception as e:
            print(f"    ! Inference error: {e}")
            # Still store something so lengths match
            if len(self.stored_obs) > 0:
                self.stored_obs.append(self.stored_obs[-1].copy())
                self.stored_floats.append(self.stored_floats[-1].copy())
                self.stored_actions.append(self.last_action)
                self.stored_log_probs.append(0.0)
                self.stored_values.append(0.0)
            return (self.last_action, True, 0.0, np.ones(12) / 12)
    
    def get_stored_rollout_data(self):
        """Get stored rollout data"""
        if len(self.stored_obs) == 0:
            return None
        
        # Check if agent went backward too much (safety check)
        backward_actions = [1, 7, 8]  # down, down+left, down+right
        backward_count = sum(1 for a in self.stored_actions if a in backward_actions)
        if backward_count > 10:
            print(f"    ⚠ Warning: Agent used backward actions {backward_count} times")
            
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
                dist_str = " ".join([f"{int(self.action_counts[i]/total*100):2d}" for i in range(12)])
                print(f"  Action %: [{dist_str}]")
                backward_pct = self.action_counts[1] / total * 100
                if backward_pct > 1.0:
                    print(f"  ⚠ BACKWARD USED {backward_pct:.1f}% of the time!")
                self.action_counts.fill(0)

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

def ppo_update(network, optimizer, rollout_data, device, epochs=3, batch_size=256):
    """Faster PPO update with fewer epochs"""
    
    obs = torch.from_numpy(rollout_data['obs']).float().to(device)
    floats = torch.from_numpy(rollout_data['floats']).float().to(device)
    actions = torch.from_numpy(rollout_data['actions']).long().to(device)
    old_log_probs = torch.from_numpy(rollout_data['log_probs']).float().to(device)
    advantages = torch.from_numpy(rollout_data['advantages']).float().to(device)
    returns = torch.from_numpy(rollout_data['returns']).float().to(device)
    
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
            
            ratio = torch.exp(log_probs - old_log_probs[mb_idx])
            surr1 = ratio * advantages[mb_idx]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = 0.5 * ((values - returns[mb_idx]) ** 2).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
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
            num_updates += 1
    
    return {k: v / num_updates for k, v in metrics.items()}

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
# MAIN TRAINING LOOP - LEARNS FROM EVERY ROLLOUT
# ============================================================================

def main():
    print("=" * 70)
    print("PPO TRAINING - Learns from ALL rollouts (even crashed ones)")
    print("=" * 70)
    
    # Print action mappings for debugging
    print("\n🎮 ACTION MAPPINGS:")
    try:
        from config_files.inputs_list import keyboard_actions_list
        for i, action in enumerate(keyboard_actions_list):
            print(f"  [{i:2d}] {action}")
        print("\n  Looking for BACKWARD action (usually has 'down' or 'brake' without forward)...")
        for i, action in enumerate(keyboard_actions_list):
            action_str = str(action).lower()
            if 'down' in action_str or ('brake' in action_str and 'up' not in action_str):
                print(f"  ⚠️  Action {i} might be BACKWARD: {action}")
    except:
        print("  Could not load action list")
    print()
    
    set_random_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Optimize CUDA for speed
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    # Get correct float input dimension
    actual_float_dim = config_copy.float_input_dim
    print(f"Float input dim: {actual_float_dim}")
    print(f"  (27 + 3*{config_copy.n_zone_centers_in_inputs} + 4*{config_copy.n_prev_actions_in_inputs} + 4*{config_copy.n_contact_material_physics_behavior_types} + 1)\n")
    
    # Create network with CORRECT dimension
    network = SimplePPONetwork(num_actions=12, float_input_dim=actual_float_dim).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4, eps=1e-5)
    
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}\n")
    
    # Create fast inferer with CORRECT dimension
    inferer = FastPPOInferer(network, device, actual_float_dim, warmup_episodes=5)
    print("Warmup: 5 episodes with random forward actions")
    print("🚫 BACKWARD ACTIONS PERMANENTLY DISABLED: [1, 7, 8]")
    print("   (down, down+left, down+right)")
    print("🚫 No-input disabled for first 100 steps\n")
    
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
    print("TRAINING STARTED - Learning from EVERY run (crashed or not)")
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
        
        # Get stored data from inferer (THIS IS KEY!)
        stored_data = inferer.get_stored_rollout_data()
        
        # Even if rollout crashed, we have stored data!
        if stored_data is None or len(stored_data['obs']) < 10:
            print("✗ No data collected (too short)")
            continue
        
        num_steps = len(stored_data['obs'])
        
        # Get rewards from rollout_results if available
        if rollout_results is not None and 'rewards' in rollout_results:
            rewards = np.array(rollout_results['rewards'], dtype=np.float32)
            race_time = end_race_stats.get('race_time', 0) / 1000
            finished = end_race_stats.get('race_finished', False)
            
            # PENALTY: Heavily penalize backward actions
            backward_actions = [1, 7, 8]  # down, down+left, down+right
            for i, action in enumerate(stored_data['actions']):
                if action in backward_actions:  # Backward action
                    rewards[i] -= 10.0  # HUGE penalty for going backward
                if action == 0 and i < 100:  # No input at start
                    rewards[i] -= 1.0  # Smaller penalty for being idle at start
                    
        else:
            # CRASHED: Give penalty rewards
            rewards = np.full(num_steps, -0.5, dtype=np.float32)  # Increased penalty
            race_time = 0
            finished = False
            print(f"  ⚠ Crashed/Incomplete rollout")
            
            # Extra penalty for backward actions in crashed runs
            backward_actions = [1, 7, 8]  # down, down+left, down+right
            for i, action in enumerate(stored_data['actions']):
                if action in backward_actions:
                    rewards[i] -= 20.0  # MASSIVE penalty if backward caused crash
        
        print(f"  {num_steps} steps, {race_time:.1f}s, {'FINISH' if finished else 'DNF/CRASH'}")
        
        if finished and race_time < best_time:
            best_time = race_time
            print(f"  🏆 NEW BEST: {best_time:.1f}s")
        
        # ALWAYS TRAIN (even on crashed runs!)
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
                
                # Compute GAE
                dones = np.zeros(num_steps, dtype=np.float32)
                dones[-1] = 1.0  # Always treat end as done (crashed or finished)
                
                advantages, returns = compute_gae(
                    rewards, 
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
                total_updates += 1
                total_steps_trained += num_steps
                
                avg_reward = rewards.mean()
                print(f"  ✓ Trained | PL:{metrics['policy_loss']:.3f} VL:{metrics['value_loss']:.3f} "
                      f"Ent:{metrics['entropy']:.3f} | Avg reward:{avg_reward:.3f}")
                
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
            }, save_path)
            print(f"\n💾 Saved | Ep:{episode} Updates:{total_updates} Steps:{total_steps_trained} Best:{best_time:.1f}s")

if __name__ == "__main__":
    main()