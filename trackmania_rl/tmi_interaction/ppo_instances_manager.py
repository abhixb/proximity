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
