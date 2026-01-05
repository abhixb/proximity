    """
    improved_ppo_trackmania.py - Enhanced PPO trainer for TrackMania
    - Eliminates ALL backward actions (backward + brake)
    - Optimized for reference line following
    - Enhanced reward shaping for racing performance
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

    # ============================================================================
    # ACTION FILTERING - REMOVE BACKWARD ACTIONS
    # ============================================================================

    def get_forward_only_actions():
        """
        Filter actions to only allow forward movement.
        Removes: backward (down), brake, and any combinations with these.
        
        Based on typical TrackMania controls:
        0=none, 1=down(back), 2=up(forward), 3=left, 4=right, 
        5=up+left, 6=up+right, 7=down+left, 8=down+right,
        9=left+brake, 10=right+brake, 11=brake
        """
        try:
            from config_files.inputs_list import keyboard_actions_list
            
            forward_actions = []
            forward_action_mapping = {}  # Maps new idx -> original idx
            
            print("\n🚫 FILTERING ACTIONS - Removing backward/brake:")
            for i, action in enumerate(keyboard_actions_list):
                action_str = str(action).lower()
                
                # Check if action contains backward or brake indicators
                is_backward = ('down' in action_str and 'up' not in action_str)
                is_brake = ('brake' in action_str)
                
                if not (is_backward or is_brake):
                    forward_action_mapping[len(forward_actions)] = i
                    forward_actions.append(action)
                    print(f"  ✓ Action {len(forward_actions)-1} -> Original {i}: {action}")
                else:
                    print(f"  ✗ Removed Original {i}: {action} (backward/brake)")
            
            print(f"\n✓ Using {len(forward_actions)} forward-only actions\n")
            return forward_actions, forward_action_mapping
        
        except Exception as e:
            print(f"Warning: Could not load action list: {e}")
            # Fallback: assume actions 2-6 are safe forward actions
            return None, {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}

    FORWARD_ACTIONS, FORWARD_ACTION_MAP = get_forward_only_actions()
    NUM_FORWARD_ACTIONS = len(FORWARD_ACTION_MAP)

    # ============================================================================
    # ENHANCED PPO NETWORK WITH REFERENCE LINE FEATURES
    # ============================================================================

    class EnhancedPPONetwork(nn.Module):
        """
        PPO Network optimized for reference line following.
        Enhanced feature extraction for spatial reasoning.
        """
        
        def __init__(self, num_actions, float_input_dim=184):
            super().__init__()
            
            self.float_input_dim = float_input_dim
            self.num_actions = num_actions
            
            # Visual processing - deeper for better spatial understanding
            self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(64)
            
            # Calculate conv output: (120,160) -> (59,79) -> (28,38) -> (26,36)
            conv_output_size = 64 * 26 * 36  # 59904
            
            # Float processing - enhanced for reference line features
            # The float input contains zone centers (reference line waypoints)
            self.float_fc1 = nn.Linear(float_input_dim, 256)
            self.float_fc2 = nn.Linear(256, 128)
            
            # Combined processing
            self.fc_shared1 = nn.Linear(conv_output_size + 128, 512)
            self.fc_shared2 = nn.Linear(512, 256)
            
            # Separate heads for better learning
            self.actor_fc = nn.Linear(256, 128)
            self.actor = nn.Linear(128, num_actions)
            
            self.critic_fc = nn.Linear(256, 128)
            self.critic = nn.Linear(128, 1)
            
            self._initialize_weights()
            
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
            # Small initialization for actor output (encourages exploration at start)
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
            nn.init.constant_(self.actor.bias, 0)
            
        def forward(self, img, float_input):
            # Normalize image
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            
            # Visual stream with batch norm
            x = F.relu(self.bn1(self.conv1(img)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = x.reshape(x.size(0), -1)
            
            # Float stream - deeper processing
            f = F.relu(self.float_fc1(float_input))
            f = F.relu(self.float_fc2(f))
            
            # Combine streams
            combined = torch.cat([x, f], dim=1)
            shared = F.relu(self.fc_shared1(combined))
            shared = F.relu(self.fc_shared2(shared))
            
            # Actor head
            actor_features = F.relu(self.actor_fc(shared))
            action_logits = self.actor(actor_features)
            
            # Critic head
            critic_features = F.relu(self.critic_fc(shared))
            value = self.critic(critic_features)
            
            return action_logits, value
        
        def get_action_and_value(self, img, float_input, deterministic=False):
            action_logits, value = self.forward(img, float_input)
            dist = torch.distributions.Categorical(logits=action_logits)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            return action, log_prob, entropy, value.squeeze(-1)
        
        def get_value(self, img, float_input):
            _, value = self.forward(img, float_input)
            return value.squeeze(-1)
        
        def evaluate_actions(self, img, float_input, actions):
            """Evaluate actions for PPO update"""
            action_logits, value = self.forward(img, float_input)
            dist = torch.distributions.Categorical(logits=action_logits)
            
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
            
            return log_prob, entropy, value.squeeze(-1)

    # ============================================================================
    # ENHANCED REWARD COMPUTATION WITH CURIOSITY
    # ============================================================================

    def compute_enhanced_rewards(rollout_data, zone_centers, current_zone_indices, episode_num):
        """
        Compute rewards that strongly encourage reference line following.
        Uses curriculum learning to gradually increase difficulty over time.

        Checkpoint extraction:
        - Prefer using map-level zone_centers (world coords) from map_loader.
        - Convert to relative coords by subtracting car world position (floats[0:3]).
        - If zone_centers missing or car pos unavailable, fall back to legacy embedded tail in floats.
        """
        rewards = []
        num_steps = len(rollout_data['obs'])
        
        # Curriculum: Start with simpler rewards, add complexity over time
        curriculum_stage = min(episode_num // 50, 4)  # 0-4 stages
        
        max_zone_reached = max(current_zone_indices) if len(current_zone_indices) > 0 else 0
        
        # Debug counters
        checkpoint_bonuses = 0
        proximity_rewards_sum = 0
        warned_no_zone_coords = False
        
        # Prepare zone_centers array if provided
        zone_centers_arr = None
        if zone_centers is not None:
            try:
                zone_centers_arr = np.asarray(zone_centers, dtype=np.float32)
            except Exception:
                zone_centers_arr = None
        
        n_zones = config_copy.n_zone_centers_in_inputs
        
        for i in range(num_steps):
            reward = 0.0
            
            floats = np.asarray(rollout_data['floats'][i], dtype=np.float32)
            zone_idx = int(current_zone_indices[i])
            
            zone_coords = None
            
            # Primary: use provided zone_centers (map_loader) and car world pos floats[0:3]
            if zone_centers_arr is not None and floats.size >= 3:
                car_pos = floats[0:3]
                # choose window starting at current zone index
                start = max(0, zone_idx)
                end = start + n_zones
                # clip slice and pad with last point if necessary
                slice_pts = zone_centers_arr[start:end]
                if slice_pts.shape[0] < n_zones:
                    if zone_centers_arr.shape[0] > 0:
                        pad_count = n_zones - slice_pts.shape[0]
                        pad = np.repeat(zone_centers_arr[-1][np.newaxis, :], pad_count, axis=0)
                        slice_pts = np.vstack((slice_pts, pad))
                    else:
                        slice_pts = np.zeros((n_zones, 3), dtype=np.float32)
                zone_coords = (slice_pts - car_pos[np.newaxis, :]).astype(np.float32)
            
            # Fallback: legacy embedded zone coords at tail of floats
            if zone_coords is None:
                try:
                    zone_coords_start = -(3 * n_zones + 1)
                    tail = floats[zone_coords_start:-1]
                    if tail.size == 3 * n_zones:
                        zone_coords = tail.reshape(n_zones, 3).astype(np.float32)
                except Exception:
                    zone_coords = None
            
            # Final fallback: zeros (warn once)
            if zone_coords is None:
                if not warned_no_zone_coords:
                    print("Warning: cannot extract zone checkpoint coords (no zone_centers/car pos/embedded tail). Using zeros.")
                    warned_no_zone_coords = True
                zone_coords = np.zeros((n_zones, 3), dtype=np.float32)
            
            # distance to current reference point
            dist_to_reference = float(np.linalg.norm(zone_coords[0]))
            
            # STAGE 0-1: Focus purely on forward progress (exploration phase)
            if curriculum_stage <= 1:
                # Massive bonus for checkpoint progress
                if i > 0 and current_zone_indices[i] > current_zone_indices[i-1]:
                    reward += 20.0  # HUGE early bonus
                    checkpoint_bonuses += 1
                
                # Reward just being close to ANY checkpoint
                if dist_to_reference < 30.0:
                    proximity_bonus = 1.0
                    reward += proximity_bonus
                    proximity_rewards_sum += proximity_bonus
                
                # Reward for being at furthest zone reached (encourage exploration)
                if zone_idx == max_zone_reached:
                    reward += 5.0  # Exploration bonus
            
            # STAGE 2+: Add precision requirements
            else:
                # Progressive checkpoint bonus
                if i > 0 and current_zone_indices[i] > current_zone_indices[i-1]:
                    reward += 15.0
                    checkpoint_bonuses += 1
                
                # Distance-based reward (tighter over time)
                max_good_dist = max(15.0 - curriculum_stage * 2, 5.0)
                if dist_to_reference < max_good_dist:
                    proximity_reward = 3.0 * (1.0 - dist_to_reference / max_good_dist)
                    reward += proximity_reward
                    proximity_rewards_sum += proximity_reward
                else:
                    penalty_factor = 0.5 if curriculum_stage < 3 else 1.0
                    reward -= penalty_factor * (dist_to_reference / max_good_dist)
            
            # Speed rewards (always active) - keep original assumption floats[6:9] == car_velocity
            forward_speed = 0.0
            try:
                car_velocity = floats[6:9]
                if car_velocity.size >= 3:
                    forward_speed = float(car_velocity[2])
            except Exception:
                forward_speed = 0.0
            
            if forward_speed > 0:
                speed_reward = min(forward_speed / 80.0, 1.5)
                reward += speed_reward
            else:
                reward -= 5.0  # Heavy backward penalty
            
            # Orientation reward - are we pointing toward next checkpoint?
            if zone_coords.shape[0] > 1:
                to_next_checkpoint = zone_coords[1] - zone_coords[0]
                norm = np.linalg.norm(to_next_checkpoint)
                if norm > 0.1:
                    to_next_checkpoint = to_next_checkpoint / norm
                    car_forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    alignment = float(np.dot(car_forward, to_next_checkpoint))
                    reward += max(alignment, 0.0) * 0.5  # Bonus for pointing right way
            
            # Stability (STAGE 3+)
            if curriculum_stage >= 3 and 'car_gear_and_wheels' in rollout_data:
                try:
                    wheels = np.asarray(rollout_data['car_gear_and_wheels'][i])
                    wheels_on_ground = wheels[4:8].sum()
                    if wheels_on_ground >= 3:
                        reward += 0.3
                    elif wheels_on_ground < 2:
                        reward -= 1.0
                except Exception:
                    pass
            
            # Time penalty (light at first, heavier later)
            time_penalty = -0.002 * (1 + curriculum_stage * 0.5)
            reward += time_penalty
            
            rewards.append(reward)
        
        rewards_array = np.array(rewards, dtype=np.float32)
        print(f"    💰 Reward stats: Stage={curriculum_stage} | Checkpoints={checkpoint_bonuses} | "
            f"Proximity={proximity_rewards_sum:.1f} | MaxZone={max_zone_reached}")
        
        return rewards_array

    # ============================================================================
    # FAST PPO COLLECTOR
    # ============================================================================

    class FastPPOCollector:
        """Collects rollout data with forward-only actions and exploration strategies"""
        
        def __init__(self, network, device, float_input_dim, num_actions, warmup_episodes=10):
            self.network = network
            self.device = device
            self.warmup_episodes = warmup_episodes
            self.episode_count = 0
            self.step_count = 0
            self.float_input_dim = float_input_dim
            self.num_actions = num_actions
            
            self.reset_rollout_storage()
            self.action_counts = np.zeros(num_actions)
            
            # Entropy-based exploration
            self.entropy_bonus = 0.2  # Start high
            self.entropy_decay = 0.995  # Decay per episode
            
        def reset_rollout_storage(self):
            """Clear storage for new episode"""
            self.stored_obs = []
            self.stored_floats = []
            self.stored_actions = []
            self.stored_log_probs = []
            self.stored_values = []
            self.stored_zone_indices = []
            
        def get_exploration_action(self, obs, float_input):
            """Select action with entropy-based exploration"""
            self.step_count += 1
            
            # EXTENDED WARMUP: More random exploration
            if self.episode_count < self.warmup_episodes:
                # Bias toward forward actions in warmup
                action_probs = np.ones(self.num_actions) / self.num_actions
                action_probs[0] *= 2.0  # Favor first forward action (usually just forward)
                action_probs /= action_probs.sum()
                action = np.random.choice(self.num_actions, p=action_probs)
                original_action = FORWARD_ACTION_MAP[action]
                return (original_action, True, 0.0, np.ones(12) / 12)
            
            try:
                if obs.ndim == 2:
                    obs = obs[np.newaxis, :, :]
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
                float_tensor = torch.from_numpy(float_input).unsqueeze(0).float().to(self.device)
                
                with torch.no_grad():
                    action_logits, value = self.network(obs_tensor, float_tensor)
                    
                    # Add entropy bonus to encourage exploration
                    # Soften the distribution by reducing logit magnitudes
                    if self.entropy_bonus > 0.01:
                        action_logits = action_logits / (1.0 + self.entropy_bonus)
                    
                    dist = torch.distributions.Categorical(logits=action_logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                action_int = int(action.cpu().item())
                log_prob_float = float(log_prob.cpu().item())
                value_float = float(value.cpu().item())
                
                original_action = FORWARD_ACTION_MAP[action_int]
                
                self.stored_obs.append(obs.copy())
                self.stored_floats.append(float_input.copy())
                self.stored_actions.append(action_int)
                self.stored_log_probs.append(log_prob_float)
                self.stored_values.append(value_float)
                
                self.action_counts[action_int] += 1
                
                probs = np.zeros(12)
                probs[original_action] = 1.0
                
                return (original_action, True, value_float, probs)
                
            except Exception as e:
                print(f"    ! Inference error: {e}")
                fallback_action = FORWARD_ACTION_MAP[0]
                return (fallback_action, True, 0.0, np.ones(12) / 12)
        
        def get_stored_rollout_data(self):
            """Get collected data"""
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
            
            # Decay exploration
            self.entropy_bonus *= self.entropy_decay
            
            if self.episode_count % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            if self.episode_count % 10 == 0:
                total = self.action_counts.sum()
                if total > 0:
                    print(f"  Exploration: {self.entropy_bonus:.3f} | Actions: ", end="")
                    for i in range(self.num_actions):
                        if self.action_counts[i] > 0:
                            pct = int(self.action_counts[i] / total * 100)
                            orig_idx = FORWARD_ACTION_MAP[i]
                            print(f"{orig_idx}:{pct}% ", end="")
                    print()
                    self.action_counts.fill(0)

    # ============================================================================
    # GAE COMPUTATION
    # ============================================================================

    def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        return advantages, returns

    # ============================================================================
    # PPO UPDATE
    # ============================================================================

    def ppo_update(network, optimizer, rollout_data, device, 
                epochs=4, batch_size=256, clip_coef=0.2, target_kl=0.015):
        """PPO update with early stopping on KL divergence"""
        
        # Convert to tensors
        obs = torch.from_numpy(rollout_data['obs']).float().to(device)
        floats = torch.from_numpy(rollout_data['floats']).float().to(device)
        actions = torch.from_numpy(rollout_data['actions']).long().to(device)
        old_log_probs = torch.from_numpy(rollout_data['log_probs']).float().to(device)
        advantages = torch.from_numpy(rollout_data['advantages']).float().to(device)
        returns = torch.from_numpy(rollout_data['returns']).float().to(device)
        old_values = rollout_data['old_values']  # Already a tensor
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        num_samples = obs.shape[0]
        indices = np.arange(num_samples)
        
        metrics = {
            'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 
            'clipfrac': 0, 'approx_kl': 0, 'epochs_done': 0
        }
        num_updates = 0
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_approx_kl = []
            
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                mb_idx = indices[start:end]
                
                mb_obs = obs[mb_idx]
                if mb_obs.ndim == 3:
                    mb_obs = mb_obs.unsqueeze(1)
                
                # Evaluate actions
                log_probs, entropy, values = network.evaluate_actions(
                    mb_obs, floats[mb_idx], actions[mb_idx]
                )
                
                # Policy loss with clipping
                ratio = torch.exp(log_probs - old_log_probs[mb_idx])
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping (helps stability)
                value_pred_clipped = old_values[mb_idx] + torch.clamp(
                    values - old_values[mb_idx], -clip_coef, clip_coef
                )
                value_loss_clipped = ((value_pred_clipped - returns[mb_idx]) ** 2)
                value_loss_unclipped = ((values - returns[mb_idx]) ** 2)
                value_loss = 0.5 * torch.max(value_loss_clipped, value_loss_unclipped).mean()
                
                # Total loss with entropy bonus
                entropy_bonus = entropy.mean()
                loss = policy_loss + 0.5 * value_loss - 0.02 * entropy_bonus
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                optimizer.step()
                
                # Metrics
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
                    approx_kl = (old_log_probs[mb_idx] - log_probs).mean()
                    epoch_approx_kl.append(approx_kl.item())
                
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy_bonus.item()
                metrics['clipfrac'] += clipfrac.item()
                num_updates += 1
            
            # Early stopping based on KL
            mean_kl = np.mean(epoch_approx_kl)
            metrics['approx_kl'] += mean_kl
            metrics['epochs_done'] = epoch + 1
            
            if mean_kl > 1.5 * target_kl:
                print(f"    Early stop at epoch {epoch+1} (KL={mean_kl:.4f})")
                break
        
        return {k: v / num_updates if k != 'epochs_done' else v for k, v in metrics.items()}

    # ============================================================================
    # GAME MANAGER
    # ============================================================================

    class RobustGameManager:
        """Manages TMInterface with reconnection"""
        
        def __init__(self, base_dir):
            self.base_dir = base_dir
            self.tmi = None
            
        def connect(self):
            """Connect to TMInterface"""
            print("Connecting to TMInterface...")
            try:
                self.tmi = game_instance_manager.GameInstanceManager(
                    game_spawning_lock=None,
                    running_speed=config_copy.running_speed,
                    run_steps_per_action=config_copy.tm_engine_step_per_action,
                    max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
                    max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
                    tmi_port=config_copy.base_tmi_port,
                )
                print("✓ Connected")
                return True
            except Exception as e:
                print(f"✗ Connection failed: {e}")
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
                return None, None

    # ============================================================================
    # MAIN TRAINING LOOP
    # ============================================================================

    def main():
        print("=" * 80)
        print("ENHANCED PPO TRAINING - Reference Line Following")
        print("=" * 80)
        print(f"\n✓ Forward-only actions: {NUM_FORWARD_ACTIONS} actions")
        print(f"✓ Backward/brake actions: REMOVED")
        print(f"✓ Reference line tracking: ENABLED\n")
        
        set_random_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}\n")
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Get float dimension
        actual_float_dim = config_copy.float_input_dim
        print(f"Float input dim: {actual_float_dim}\n")
        
        # Create network
        network = EnhancedPPONetwork(
            num_actions=NUM_FORWARD_ACTIONS,
            float_input_dim=actual_float_dim
        ).to(device)
        
        # Create optimizer with higher learning rate for exploration
        optimizer = torch.optim.Adam(network.parameters(), lr=5e-4, eps=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000, eta_min=1e-5
        )
        
        print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}\n")
        
        # Create collector with extended warmup
        collector = FastPPOCollector(
            network, device, actual_float_dim, NUM_FORWARD_ACTIONS, warmup_episodes=10
        )
        
        # Create game manager
        game_mgr = RobustGameManager(base_dir)
        if not game_mgr.connect():
            print("Cannot start without game connection")
            return
        
        # Map cycle
        set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
        map_cycle_iter = cycle(chain(*config_copy.map_cycle))
        zone_centers_filename = None
        
        print("=" * 80)
        print("TRAINING STARTED")
        print("=" * 80)
        
        network.train()
        best_time = float('inf')
        total_updates = 0
        
        for episode in range(1, 10000):
            collector.new_episode()
            
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
            mode = "TRAIN" if is_explo else "EVAL"
            
            print(f"\n[Ep {episode}] {mode} {map_name}")
            
            # Execute rollout
            rollout_results, end_race_stats = game_mgr.safe_rollout(
                exploration_policy=collector.get_exploration_action,
                map_path=map_path,
                zone_centers=zone_centers
            )
            
            # Get stored data
            stored_data = collector.get_stored_rollout_data()
            
            if stored_data is None or len(stored_data['obs']) < 10:
                print("✗ No data collected")
                continue
            
            num_steps = len(stored_data['obs'])
            
            # Compute enhanced rewards WITH CURRICULUM
            if rollout_results is not None and 'current_zone_idx' in rollout_results:
                # CRITICAL: Actually compute rewards based on the rollout!
                current_zones = rollout_results['current_zone_idx'][:num_steps]
                
                # Get race time and finish status
                race_time = end_race_stats.get('race_time', 0) / 1000
                finished = end_race_stats.get('race_finished', False)
                
                # Compute curriculum-based rewards
                rewards = compute_enhanced_rewards(
                    stored_data, 
                    zone_centers,
                    current_zones,
                    episode  # Pass episode number for curriculum
                )
                
                # Track progress
                max_zone = max(current_zones)
                furthest_zone = rollout_results.get('furthest_zone_idx', 0)
                
                print(f"  ✓ Rewards computed: min={rewards.min():.2f} max={rewards.max():.2f} mean={rewards.mean():.2f}")
            else:
                # Crashed - heavier penalty but still encourage any progress
                rewards = np.full(num_steps, -3.0, dtype=np.float32)
                race_time = 0
                finished = False
                max_zone = 0
                furthest_zone = 0
                print("  ⚠ Crashed/Incomplete - Penalty rewards assigned")
            
            avg_reward = rewards.mean()
            curriculum_stage = min(episode // 50, 4)
            print(f"  {num_steps} steps, {race_time:.1f}s, {'FINISH' if finished else 'DNF'}")
            print(f"  Reward: {avg_reward:.2f} | MaxZone: {max_zone} | Stage: {curriculum_stage}")
            
            if finished and race_time < best_time:
                best_time = race_time
                print(f"  🏆 NEW BEST: {best_time:.1f}s")
            
            # Train
            if fill_buffer and num_steps > 50:
                try:
                    # Get last value
                    last_obs = stored_data['obs'][-1]
                    if last_obs.ndim == 2:
                        last_obs = last_obs[np.newaxis, :, :]
                    last_obs_t = torch.from_numpy(last_obs).unsqueeze(0).float().to(device)
                    last_float_t = torch.from_numpy(stored_data['floats'][-1]).unsqueeze(0).float().to(device)
                    
                    with torch.no_grad():
                        last_value = network.get_value(last_obs_t, last_float_t).cpu().item()
                    
                    # Compute GAE
                    dones = np.zeros(num_steps, dtype=np.float32)
                    dones[-1] = 1.0
                    
                    # Store old values for clipped value loss
                    old_values = torch.from_numpy(stored_data['values']).float().to(device)
                    
                    advantages, returns = compute_gae(
                        rewards, 
                        stored_data['values'], 
                        dones, 
                        last_value,
                        gamma=0.99,
                        gae_lambda=0.95
                    )
                    
                    # Prepare training data
                    train_data = {
                        'obs': stored_data['obs'],
                        'floats': stored_data['floats'],
                        'actions': stored_data['actions'],
                        'log_probs': stored_data['log_probs'],
                        'advantages': advantages,
                        'returns': returns,
                        'old_values': old_values,
                    }
                    
                    # Update with adaptive learning
                    metrics = ppo_update(network, optimizer, train_data, device)
                    total_updates += 1
                    
                    # Step LR scheduler
                    lr_scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    print(f"  ✓ PL:{metrics['policy_loss']:.3f} VL:{metrics['value_loss']:.3f} "
                        f"Ent:{metrics['entropy']:.3f} KL:{metrics['approx_kl']:.4f} "
                        f"LR:{current_lr:.2e}")
                    
                except Exception as e:
                    print(f"✗ Training error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Save checkpoint
            if episode % 25 == 0:
                save_path = base_dir / "save" / config_copy.run_name / f"ppo_enhanced_ep{episode}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'episode': episode,
                    'network': network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_time': best_time,
                    'total_updates': total_updates,
                    'action_mapping': FORWARD_ACTION_MAP,
                }, save_path)
                print(f"\n💾 Saved | Ep:{episode} Updates:{total_updates} Best:{best_time:.1f}s")

    if __name__ == "__main__":
        main()