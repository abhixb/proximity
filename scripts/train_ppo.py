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

###############################################################################
from trackmania_rl.ppo_rewards import compute_enhanced_rewards, compute_gae 
from trackmania_rl.agents.ppo import SimplePPONetwork , FastPPOInferer , ALLOWED_ACTIONS
from trackmania_rl.ppo_metrics import MetricsTracker
from trackmania_rl.multiprocess.ppo_learner import ppo_update 
from trackmania_rl.tmi_interaction.ppo_instances_manager import  RobustGameManager


def main():
    print("=" * 70)
    print("PPO TRAINING - FIXED VERSION")
    print("7 Actions (removed duplicates)")
    print("Fixed log-prob calculation")
    print("Smarter actor reset logic")
    print("=" * 70)
    
    set_random_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    actual_float_dim = config_copy.float_input_dim
    print(f"Float input dim: {actual_float_dim}\n")
    
    network = SimplePPONetwork(num_actions=7, float_input_dim=actual_float_dim).to(device)
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
    zone_centers = None
    
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
