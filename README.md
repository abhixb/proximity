



## Proximity 
Proximity is a fork of [Linesight]( https://github.com/Linesight-RL/linesight )
Proximity specifically aims to build and fine-tune a PPO architecture to train agents effectively within that ecosystem.

Why PPO?
We utilize Proximal Policy Optimization because it offers a balance between ease of tuning, sample complexity, and performance. It prevents the agent from making drastically detrimental policy updates, ensuring smoother convergence in complex environments like those provided by Linesight.

## Trackmania

Trackmania is a racing game that sacrifices some of the realism of sim-racers for a wide variety of track types with all kinds of tricks like wall riding, stunt jumps and wallbangs. Furthermore, Trackmania was designed for equality of input devices which means that keyboard inputs are a viable way to play and therefore that discrete input algorithms like DQN can be applied. In other words, Trackmania is a deep game which can serve as a benchmark to work on any RL algorithm.

## additions made 
```
proximity/
├──           
├── config_files/config_ppo          # Hyperparameter configuration files             
├── scripts/train_ppo.py          # Main training loop
└── trackmania_rl/
    ├── agents/ppo_agent.py #ppo agent implementation 
    ├── multiprocesses/
        ├──ppo_learner.py
    ├──ppo_metrics.py
    ├──ppo_rewards.py
    
```
## `config_ppo.py`

- PPO hyperparameters and training settings  
- Network architecture and optimizer configuration  

This file contains all tunable parameters for PPO, including discounting, GAE, clipping, entropy and value loss weights, rollout sizes, batch sizes, and optimization settings. Centralizing configuration ensures reproducibility and allows experiments to be modified without changing training logic.

---

## `train_ppo.py`

- Entry point for training  
- Coordinates data collection, learning, and logging  

This file drives the entire PPO training process. It initializes the environment and all core components, collects rollouts through environment interaction, triggers PPO updates when sufficient data is available, logs metrics, and saves checkpoints. It is responsible for control flow, not model or learning details.

---

## `ppo_agent.py`

- Actor–critic policy definition  
- Action sampling and value prediction  

This file defines the PPO policy and value networks and their forward inference behavior. It maps observations to action distributions and state-value estimates, handling log-probabilities and inference modes. The agent is purely a policy representation and does not perform optimization.

---

## `ppo_learner.py`

- PPO loss computation  
- Gradient-based policy and value updates  

This file implements the PPO algorithm itself. It computes advantages, applies the clipped surrogate objective, evaluates value and entropy losses, and performs optimizer steps over multiple epochs and minibatches. Learning logic is fully isolated from environment interaction and policy definition.

---

## `ppo_metrics.py`

- Training diagnostics and visualization  
- Metric aggregation and logging  

This file tracks key PPO signals such as episodic return, policy loss, value loss, entropy, and KL divergence. It provides logging and visualization utilities that are critical for debugging instability, collapse, or poor convergence.

---

## `ppo_rewards.py`

- Task-specific reward shaping  
- Reward scaling and decomposition  

This file defines how raw environment feedback is transformed into learning signals. It contains dense shaping terms, penalties, and terminal bonuses tailored to the task. Separating reward logic allows rapid iteration without modifying PPO mechanics.




