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
from trackmania_rl.agents.ppo import SimplePPONetwork , FastPPOInferer
from trackmania_rl.multiprocess.ppo_learner import ppo_update 
from trackmania_rl.tmi_interaction.ppo_instances_manager import  RobustGameManager


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
        
        print(f"    Plots saved to {plot_path.name}")
    
    def get_summary_stats(self):
        return {
            'avg_reward': np.mean(self.reward_window) if self.reward_window else 0,
            'avg_time': np.mean(self.time_window) if self.time_window else 0,
            'finish_rate': np.mean(self.finish_window) if self.finish_window else 0,
        }