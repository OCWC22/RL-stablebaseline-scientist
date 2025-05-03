https://docs.google.com/document/d/1Luw_jWPiLfq-uG9idGC5CdSG8ywxLiuI5SHga57vyjY/edit?tab=t.0


Replicating Standard Reinforcement Learning Benchmarks on CartPole-v1 with Stable Baselines3: PPO, DQN, and A2C
1. Introduction
Objective: This report provides a detailed technical guide for replicating standard Reinforcement Learning (RL) experiments using three foundational algorithms: Proximal Policy Optimization (PPO), Deep Q-Network (DQN), and Advantage Actor-Critic (A2C). The focus is on applying these algorithms to the classic CartPole-v1 control benchmark environment, utilizing the capabilities of the Stable Baselines3 (SB3) library.
Context: Deep Reinforcement Learning has demonstrated significant success across various domains, yet reproducing published results remains a persistent challenge. Small implementation details can drastically affect performance, often more so than the algorithmic differences themselves.1 This underscores the need for standardized benchmark environments, such as those provided by the Gymnasium (formerly OpenAI Gym) interface, and reliable, well-tested algorithm implementations.
Stable Baselines3 (SB3) Introduction: Stable Baselines3 emerges as a critical tool in addressing the reproducibility challenge in RL. It offers a curated set of reliable, state-of-the-art model-free RL algorithm implementations in PyTorch.2 SB3 aims to simplify the process of replicating research, establishing robust baselines for new ideas, and making advanced RL techniques accessible to a broader community, akin to a "scikit-learn for RL".3 The library emphasizes ease of use through a unified structure and clean API, allowing agent training with minimal code.1 However, it is important to note that while SB3 simplifies implementation, a foundational understanding of RL concepts is assumed for effective utilization.3 The performance of each implemented algorithm has been tested and benchmarked.2
Scope: This report concentrates specifically on the PPO, DQN, and A2C algorithms as implemented in SB3, applied to the CartPole-v1 environment. It draws upon the functionalities demonstrated in the Stable Baselines3 documentation and associated resources, providing both conceptual explanations and practical, runnable code examples.
2. Prerequisites: Environment and Setup
Before implementing the RL algorithms, the necessary software must be installed, and the target environment, CartPole-v1, must be understood.
Installation:
The primary libraries required are gymnasium (the current standard for RL environments, succeeding OpenAI Gym) and stable-baselines3. Installation is typically performed using pip:

Bash


# Install Gymnasium (core RL environment interface)
pip install gymnasium

# Install Stable Baselines3 (core library)
pip install stable-baselines3

# Optionally, install with extras for wider compatibility (e.g., Atari, plotting)
pip install stable-baselines3[extra]


Recent versions of Stable Baselines3 require Python 3.8 or higher.5 The [extra] option installs additional dependencies such as atari-py (for Atari environments), opencv-python (for image processing), and tensorboard (for logging), which might be useful for broader experimentation but are not strictly necessary for CartPole-v1.3
The CartPole-v1 Environment:
CartPole-v1 is a classic control problem often used as an initial benchmark in RL research.
Description: The environment consists of a cart that can move horizontally along a frictionless track. A pole is attached to the cart via an unactuated joint. The objective is to balance the pole vertically by applying horizontal forces (+1 or -1) to the cart.9
Reward: A reward of +1 is given for every timestep the pole remains upright within the defined limits.9
State Space (Observation Space): The state of the environment is represented by a 4-dimensional continuous vector containing:
Cart Position (range typically observed: -2.4 to 2.4, though bounds are wider)
Cart Velocity
Pole Angle (range typically observed: approx. -12 to +12 degrees, or -0.2095 to +0.2095 radians)
Pole Angular Velocity This corresponds to a gymnasium.spaces.Box space.3
Action Space: The agent can choose between two discrete actions: 0. Push cart to the left
Push cart to the right This corresponds to a gymnasium.spaces.Discrete(2) space.3
Termination Conditions: An episode ends if:
The pole angle exceeds ±12 degrees (approx. ±0.2095 radians) from the vertical.11
The cart position moves more than 2.4 units from the center.11
The episode length exceeds 500 timesteps (for CartPole-v1; earlier versions might have used 200).11
The simplicity of CartPole-v1, characterized by its low-dimensional state space, discrete actions, and unambiguous reward signal, makes it an ideal environment for initial testing and debugging of RL algorithms and implementations.9 Its prevalence in tutorials and benchmarks provides a common ground for comparing results. Successfully training an agent to achieve the maximum score (typically 500, corresponding to surviving 500 steps) indicates that the basic RL setup is functioning correctly before moving to more complex challenges.15
Code Snippet: Environment Creation:
Creating an instance of the environment is straightforward using Gymnasium:

Python


import gymnasium as gym

# Create the CartPole-v1 environment
# render_mode="rgb_array" is often used for recording videos or visualizing without a pop-up window
# render_mode="human" will display the environment in a window
env = gym.make("CartPole-v1", render_mode="human") # Or "rgb_array"
print(f"Observation Space: {env.observation_space}")
print(f"Action Space: {env.action_space}")

# Reset the environment to get the initial observation
obs, info = env.reset()
print(f"Initial Observation: {obs}")

# Take a random action
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Observation after random action: {obs}")
print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

env.close()


This code demonstrates how to instantiate the environment, inspect its spaces, and interact with it for a single step.5
3. Stable Baselines3 Core Concepts for Replication
Understanding a few core components of SB3 is essential for effective use and replication.
Key Components:
Algorithms: These are the classes implementing specific RL algorithms, such as stable_baselines3.PPO, stable_baselines3.DQN, and stable_baselines3.A2C.9 SB3 provides a unified structure and API across these algorithms, simplifying experimentation.2 Each algorithm typically involves methods for collecting experience (collect_rollouts) and updating the model (train).18
Policies: These define the agent's decision-making process, usually implemented as neural networks. SB3 provides standard policy networks like "MlpPolicy" (Multi-Layer Perceptron) for vector-based inputs and "CnnPolicy" for image-based inputs.9 It's important to understand that in SB3 terminology, "policy" often refers to the overarching class that manages all necessary networks (actor, critic, target networks) and their optimizers, not just the action-selection network.18
Vectorized Environments (VecEnv): SB3 internally utilizes VecEnv wrappers to handle potentially multiple environment instances running in parallel. Even when training on a single environment, SB3 often wraps it in a DummyVecEnv, which runs the environment in the main process. For true parallelization across CPU cores, SubprocVecEnv can be used.9 This vectorization is key to the efficiency of algorithms like A2C and PPO.
Buffers: Algorithms store collected experiences in buffers before using them for updates. On-policy algorithms like PPO and A2C use a RolloutBuffer, which typically stores experiences from the latest policy iteration and is discarded after the gradient update. Off-policy algorithms like DQN use a ReplayBuffer, which stores a larger history of transitions that are sampled randomly for updates, decoupling data collection from learning.17
The MlpPolicy Architecture:
For environments with vector-based observations like CartPole-v1, "MlpPolicy" is the standard choice.9
Components:
Features Extractor: By default, MlpPolicy uses stable_baselines3.common.torch_layers.FlattenExtractor. This layer simply takes the input observation vector and flattens it (which often results in no change for already flat vectors like CartPole's) before passing it to the main network body.19 Observations undergo pre-processing (like normalization if enabled, or one-hot encoding for discrete observations, though not relevant for CartPole's continuous state) before reaching the extractor.18
Network Body (net_arch): This consists of fully connected (dense) layers that process the features extracted. The architecture (number and size of hidden layers) is controlled by the net_arch parameter within policy_kwargs.19
Algorithm-Specific Defaults: The default architecture activated by "MlpPolicy" varies depending on the algorithm, reflecting common practices and empirical performance:
For PPO, A2C, and DQN, the default net_arch is typically two hidden layers with 64 units each (``).20
For algorithms like SAC, TD3, and DDPG (off-policy, continuous action space), the defaults are often larger (e.g., for SAC, for TD3/DDPG).20 This difference highlights that even the default "MlpPolicy" incorporates some level of algorithm-specific optimization.
Policy/Value Heads:
For Actor-Critic algorithms (PPO, A2C), the net_arch parameter allows specifying shared layers followed by separate layers for the policy network (actor, outputs action distribution parameters) and the value network (critic, outputs state value estimate). For example, net_arch=[dict(pi=, vf=)] would mean separate 64-unit layers after the feature extractor, while net_arch=[128, dict(pi=, vf=)] would have a shared 128-unit layer first.19 The default (``) implies shared layers. Feature extractors are also shared by default in on-policy algorithms to save computation.20
For DQN, which is value-based, the net_arch defines the architecture of the Q-network, which outputs Q-values for each discrete action.29
Customization: Users can override defaults by passing a dictionary to the policy_kwargs argument during model initialization. This allows specifying net_arch, activation_fn (e.g., torch.nn.ReLU), optimizer_class, etc..19
The use of standardized policy names like "MlpPolicy" provides a convenient abstraction. However, it is beneficial to be aware that this simple string activates potentially different underlying network structures and sharing strategies optimized for the specific algorithm class (on-policy actor-critic vs. off-policy value-based). This design choice balances ease of use with sensible, empirically validated defaults, abstracting complexity while providing a solid starting point for experiments. Fine-grained control remains available through policy_kwargs for advanced users or specific problem requirements.
4. Implementing and Training PPO
Proximal Policy Optimization (PPO) is a widely used on-policy actor-critic algorithm recognized for its robustness, good performance across a variety of tasks, and relative ease of tuning compared to other methods.9 Key ideas include clipping the objective function to prevent destructively large policy updates and performing multiple optimization epochs on the same batch of collected data (rollout) to improve sample efficiency compared to simpler policy gradient methods.23 Being on-policy, it requires collecting new trajectories with the current policy for each update cycle. While often less sample efficient than off-policy algorithms like DQN or SAC in terms of total environment interactions, PPO can be significantly faster in terms of wall-clock time, especially when parallelized using vectorized environments.9
Code Implementation:

Python


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# --- Environment Setup ---
# Use make_vec_env to handle environment vectorization
# n_envs=1 creates a DummyVecEnv (single process)
# For parallel execution, increase n_envs and consider SubprocVecEnv
env_id = "CartPole-v1"
n_envs = 1 # Start with 1, can increase for faster training if CPU allows
env = make_vec_env(env_id, n_envs=n_envs)

# --- Model Instantiation ---
# Use MlpPolicy for CartPole's vector observations
# verbose=1 provides training progress updates
# Specify hyperparameters here or use defaults/RL Zoo values
model_ppo = PPO("MlpPolicy", env, verbose=1)

# --- Training ---
# Define the total number of timesteps for training
total_timesteps_ppo = 25000 # Adjust as needed, RL Zoo uses 100k
model_ppo.learn(total_timesteps=total_timesteps_ppo, progress_bar=True)

# --- Saving ---
model_ppo.save("ppo_cartpole_model")
print(f"PPO model saved to ppo_cartpole_model.zip")

# --- Cleanup ---
env.close()


Key PPO Hyperparameters in SB3:
Understanding the main hyperparameters is crucial for tuning and replication 23:
learning_rate: Step size for the optimizer (default: 0.0003). Can be a constant or a schedule (a function of training progress).
n_steps: Number of steps collected per environment before each policy update. The total rollout buffer size is n_steps * n_envs (default: 2048). Must ensure n_steps * n_envs > 1.30
batch_size: Size of the mini-batches used during the optimization epochs (default: 64). The rollout buffer is divided into mini-batches.
n_epochs: Number of optimization epochs performed on the collected rollout data (default: 10).
gamma: Discount factor for future rewards (default: 0.99).
gae_lambda: Factor for Generalized Advantage Estimation (GAE), balancing bias and variance in advantage calculation (default: 0.95).
clip_range: The PPO clipping parameter, limiting the policy change per update (default: 0.2). Can be a constant or a schedule.
ent_coef: Coefficient for the entropy bonus in the loss, encouraging exploration (default: 0.0).
vf_coef: Coefficient for the value function loss in the total loss calculation (default: 0.5).
max_grad_norm: Maximum norm for gradient clipping to prevent large updates (default: 0.5).
normalize_advantage: Whether to normalize the calculated advantages (default: True).
Hyperparameters for CartPole-v1:
While default hyperparameters provide a starting point, achieving benchmark performance often requires values tuned specifically for the environment and algorithm combination.39 The RL Baselines3 Zoo is a primary resource for such tuned parameters, derived from systematic optimization (often using tools like Optuna).40 These are typically stored in YAML files within the Zoo repository (e.g., hyperparameters/ppo.yml).40 Comparing these tuned values against the SB3 defaults reveals which parameters are most impactful for a given task.
Table 1: PPO Hyperparameters for CartPole-v1

Parameter
SB3 Default Value
RL Zoo Tuned Value
Description
policy
"MlpPolicy"
"MlpPolicy"
Network type (MLP for vector inputs)
n_timesteps
N/A
100,000
Total training duration
n_envs
N/A (Assumed 1 if not VecEnv)
8
Number of parallel environments
n_steps
2048
32
Steps per env per update (Rollout buffer size = n_steps * n_envs)
batch_size
64
256
Mini-batch size for optimization epochs
n_epochs
10
20
Number of optimization epochs per rollout
gamma
0.99
0.98
Discount factor
gae_lambda
0.95
0.8
GAE lambda parameter
clip_range
0.2
lin_0.2
PPO clipping parameter (linear schedule from 0.2)
ent_coef
0.0
0.0
Entropy coefficient
learning_rate
0.0003
lin_0.001
Optimizer learning rate (linear schedule from 0.001)
normalize_advantage
True
True
Whether to normalize advantages (Not explicitly listed in 44, assumed True based on SB3 default)
vf_coef
0.5
0.5
Value function loss coefficient (Not explicitly listed in 44, assumed 0.5 based on SB3 default)

Note: lin_X denotes a linear schedule decreasing from X to 0 over the course of training.
The comparison highlights significant adjustments for CartPole: using more parallel environments (n_envs=8), a much smaller rollout per update (n_steps=32), a larger mini-batch size (batch_size=256), more optimization epochs (n_epochs=20), adjusted discounting and GAE (gamma=0.98, gae_lambda=0.8), and scheduled learning rate and clip range. These changes collectively aim to improve stability and learning speed for this specific, relatively simple environment.
Code with Tuned Hyperparameters:

Python


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import linear_schedule

# --- Environment Setup ---
env_id = "CartPole-v1"
n_envs = 8 # Use 8 parallel environments as per RL Zoo
env = make_vec_env(env_id, n_envs=n_envs)

# --- Tuned Hyperparameters [44] ---
# Note: SB3 handles linear schedules directly if passed as functions
ppo_tuned_params = {
    "n_steps": 32,
    "batch_size": 256,
    "n_epochs": 20,
    "gamma": 0.98,
    "gae_lambda": 0.8,
    "clip_range": linear_schedule(0.2), # SB3 interprets schedule functions
    "ent_coef": 0.0,
    "learning_rate": linear_schedule(0.001), # SB3 interprets schedule functions
    "vf_coef": 0.5, # Assuming default
    "normalize_advantage": True # Assuming default
}

# --- Model Instantiation ---
model_ppo_tuned = PPO("MlpPolicy",
                      env,
                      verbose=1,
                      tensorboard_log="./ppo_cartpole_tuned_tensorboard/",
                      **ppo_tuned_params)

# --- Training ---
total_timesteps_ppo_tuned = 100000 # Use RL Zoo's budget
model_ppo_tuned.learn(total_timesteps=total_timesteps_ppo_tuned, progress_bar=True)

# --- Saving ---
model_ppo_tuned.save("ppo_cartpole_tuned_model")
print(f"Tuned PPO model saved to ppo_cartpole_tuned_model.zip")

# --- Cleanup ---
env.close()


5. Implementing and Training DQN
Deep Q-Network (DQN) is a seminal off-policy, value-based algorithm primarily designed for environments with discrete action spaces.17 It revolutionized the field by successfully combining Q-learning with deep neural networks to learn control policies directly from high-dimensional sensory input (like pixels, though we use vector input here). Key innovations include the use of a replay buffer to store past experiences and break correlations in sequential data, and a separate target network to stabilize learning by providing fixed Q-value targets during updates.17 Exploration is typically handled using an epsilon-greedy strategy, where the agent takes a random action with probability ϵ (which usually decays over time) and the greedy action (highest Q-value) otherwise. The standard SB3 implementation provides vanilla DQN, without common extensions like Double DQN or Dueling DQN built-in, although these could potentially be implemented via customization.29
Code Implementation:

Python


import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# --- Environment Setup ---
# DQN typically uses a single environment for interaction,
# but SB3 still benefits from the VecEnv wrapper structure.
env_id = "CartPole-v1"
env = make_vec_env(env_id, n_envs=1) # DQN is off-policy, often uses n_envs=1

# --- Model Instantiation ---
# Use MlpPolicy for CartPole's vector observations
# verbose=1 provides training progress updates
model_dqn = DQN("MlpPolicy", env, verbose=1)

# --- Training ---
# Define the total number of timesteps for training
total_timesteps_dqn = 50000 # Adjust as needed, RL Zoo uses 50k
model_dqn.learn(total_timesteps=total_timesteps_dqn, progress_bar=True, log_interval=4) # log_interval controls logging frequency

# --- Saving ---
model_dqn.save("dqn_cartpole_model")
print(f"DQN model saved to dqn_cartpole_model.zip")

# --- Cleanup ---
env.close()


Key DQN Hyperparameters in SB3:
Effective DQN training relies on careful tuning of parameters related to the replay buffer, target network updates, and exploration 15:
learning_rate: Step size for the Q-network optimizer (default: 0.0001).
buffer_size: Maximum number of transitions stored in the replay buffer (default: 1,000,000).
learning_starts: Number of steps to collect experience before starting gradient updates (default: 50,000 for DQN in v1.0, 100 in master/later versions). This is crucial; training won't begin until this threshold is met.
batch_size: Number of transitions sampled from the replay buffer for each gradient update (default: 32).
tau: Polyak averaging coefficient for soft target network updates (default: 1.0, meaning hard updates). Smaller values (e.g., 0.005) mean softer updates.
gamma: Discount factor for future rewards (default: 0.99).
train_freq: Frequency of model updates. Can be steps or episodes (e.g., (4, "step") means update every 4 steps, default is (4, "step")).
gradient_steps: Number of gradient updates performed per training frequency interval (default: 1). -1 means perform as many updates as steps collected in the interval.
target_update_interval: Frequency (in environment steps) of updating the target network (default: 10,000). For hard updates (tau=1.0).
exploration_fraction: Fraction of the total training timesteps over which ϵ decreases (default: 0.1).
exploration_initial_eps: Starting value of ϵ (default: 1.0).
exploration_final_eps: Final value of ϵ after decaying (default: 0.05).
max_grad_norm: Maximum norm for gradient clipping (default: 10).
policy_kwargs: Dictionary for customizing the policy network (e.g., dict(net_arch=)).
A critical aspect of using DQN (and other off-policy algorithms) is the interplay between total_timesteps and learning_starts. If total_timesteps is set lower than learning_starts, the agent will collect data but never actually perform any learning updates, leading to random performance regardless of other hyperparameter settings.47 This has been a source of confusion for users attempting to replicate results without ensuring a sufficient training budget relative to the buffer warm-up phase.15
Hyperparameters for CartPole-v1:
Tuned hyperparameters for DQN on CartPole-v1 are available from the RL Baselines3 Zoo and associated Hugging Face model cards.40
Table 2: DQN Hyperparameters for CartPole-v1

Parameter
SB3 Default Value
RL Zoo Tuned Value
Description
policy
"MlpPolicy"
"MlpPolicy"
Network type
n_timesteps
N/A
50,000
Total training duration
learning_rate
0.0001
0.0023
Optimizer learning rate
buffer_size
1,000,000
100,000
Replay buffer capacity
learning_starts
100 (master) / 50k (v1.0)
1,000
Steps before learning starts
batch_size
32
64
Mini-batch size for updates
tau
1.0
1.0
Target network update coefficient (Hard update)
gamma
0.99
0.99
Discount factor
train_freq
(4, "step")
(256, "step")
Frequency of training updates (steps)
gradient_steps
1
128
Number of gradient steps per update
target_update_interval
10,000
10
Frequency (steps) of target network updates
exploration_fraction
0.1
0.16
Fraction of training for epsilon decay
exploration_initial_eps
1.0
1.0
Initial epsilon value
exploration_final_eps
0.05
0.04
Final epsilon value
policy_kwargs
None
dict(net_arch=)
Network architecture (2 hidden layers, 256 units each)

The tuned parameters show significant changes compared to defaults: a much higher learning rate, smaller buffer, drastically reduced learning_starts, larger batch size, less frequent but much more intensive training updates (train_freq, gradient_steps), very frequent target network updates, a slightly adjusted exploration schedule, and a specific network architecture. These adjustments reflect the optimization needed to make DQN learn efficiently on the relatively simple CartPole task within a modest timestep budget. Even with tuned parameters, achieving the maximum score of 500 might require careful implementation and sufficient training time, as some users have reported difficulties reaching the theoretical maximum consistently.15
Code with Tuned Hyperparameters:

Python


import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# --- Environment Setup ---
env_id = "CartPole-v1"
env = make_vec_env(env_id, n_envs=1)

# --- Tuned Hyperparameters [50] ---
dqn_tuned_params = {
    "learning_rate": 0.0023,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 64,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": (256, "step"), # Update every 256 steps
    "gradient_steps": 128, # Perform 128 gradient steps per update
    "target_update_interval": 10, # Update target network every 10 steps
    "exploration_fraction": 0.16,
    "exploration_final_eps": 0.04,
    "policy_kwargs": dict(net_arch=) # Specify network architecture
}

# --- Model Instantiation ---
model_dqn_tuned = DQN("MlpPolicy",
                      env,
                      verbose=1,
                      tensorboard_log="./dqn_cartpole_tuned_tensorboard/",
                      **dqn_tuned_params)

# --- Training ---
total_timesteps_dqn_tuned = 50000 # Use RL Zoo's budget
# Ensure total_timesteps > learning_starts
if total_timesteps_dqn_tuned <= dqn_tuned_params["learning_starts"]:
    print(f"Warning: total_timesteps ({total_timesteps_dqn_tuned}) should be greater than learning_starts ({dqn_tuned_params['learning_starts']}) for DQN training.")
    total_timesteps_dqn_tuned = dqn_tuned_params["learning_starts"] + 1000 # Increase budget slightly

model_dqn_tuned.learn(total_timesteps=total_timesteps_dqn_tuned, progress_bar=True, log_interval=10) # Log every 10 episodes

# --- Saving ---
model_dqn_tuned.save("dqn_cartpole_tuned_model")
print(f"Tuned DQN model saved to dqn_cartpole_tuned_model.zip")

# --- Cleanup ---
env.close()


6. Implementing and Training A2C
Advantage Actor-Critic (A2C) is a synchronous, on-policy actor-critic algorithm, often considered a simpler, deterministic variant of the Asynchronous Advantage Actor-Critic (A3C) algorithm.24 Instead of asynchronous updates from multiple workers, A2C typically waits for all parallel environments (managed by a VecEnv) to complete a batch of steps (n_steps) before performing a single, centralized gradient update. This synchronous nature can lead to more stable training compared to A3C, leveraging the benefits of batch processing. Like PPO, it's an on-policy method requiring fresh samples for each update.
Code Implementation:

Python


import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# --- Environment Setup ---
# A2C benefits from parallel environments
env_id = "CartPole-v1"
n_envs = 4 # Use multiple environments for A2C
env = make_vec_env(env_id, n_envs=n_envs)

# --- Model Instantiation ---
# Use MlpPolicy for CartPole's vector observations
# verbose=1 provides training progress updates
model_a2c = A2C("MlpPolicy", env, verbose=1)

# --- Training ---
# Define the total number of timesteps for training
total_timesteps_a2c = 25000 # Adjust as needed, RL Zoo uses 500k
model_a2c.learn(total_timesteps=total_timesteps_a2c, progress_bar=True)

# --- Saving ---
model_a2c.save("a2c_cartpole_model")
print(f"A2C model saved to a2c_cartpole_model.zip")

# --- Cleanup ---
env.close()


Key A2C Hyperparameters in SB3:
The core hyperparameters for A2C are similar to other actor-critic methods, focusing on the balance between policy and value updates, and managing the batch of experiences 24:
learning_rate: Step size for the optimizer (default: 0.0007).
n_steps: Number of steps collected per environment before each update. The batch size for the update is n_steps * n_envs (default: 5). This default is quite small.
gamma: Discount factor for future rewards (default: 0.99).
gae_lambda: Factor for Generalized Advantage Estimation (default: 1.0, which corresponds to standard N-step advantage).
ent_coef: Coefficient for the entropy bonus in the loss (default: 0.0).
vf_coef: Coefficient for the value function loss (default: 0.5).
max_grad_norm: Maximum norm for gradient clipping (default: 0.5).
use_rms_prop: Whether to use RMSprop (default: True) or Adam optimizer. The documentation notes that using a specific RMSprop implementation (RMSpropTFLike) might be needed to match older Stable Baselines performance or improve stability.24
rms_prop_eps: Epsilon parameter for RMSprop (default: 1e-5).
normalize_advantage: Whether to normalize advantages (default: False for A2C).
Hyperparameters for CartPole-v1:
Tuned hyperparameters for A2C on CartPole-v1 can be found in the RL Baselines3 Zoo repository and associated Hugging Face model cards.40
Table 3: A2C Hyperparameters for CartPole-v1

Parameter
SB3 Default Value
RL Zoo/HF Tuned Value
Description
policy
"MlpPolicy"
"MlpPolicy"
Network type
n_timesteps
N/A
500,000
Total training duration
n_envs
N/A (Assumed 1 if not VecEnv)
8
Number of parallel environments
n_steps
5
5 (?)
Steps per env per update (Batch size = n_steps*n_envs)
gamma
0.99
0.99 (?)
Discount factor
gae_lambda
1.0
1.0 (?)
GAE lambda parameter (1.0 = N-step advantage)
ent_coef
0.0
0.0
Entropy coefficient
vf_coef
0.5
0.5 (?)
Value function loss coefficient
learning_rate
0.0007
0.0007 (?)
Optimizer learning rate
normalize_advantage
False
False (?)
Whether to normalize advantages

Note: Values marked with (?) are inferred defaults or common settings, as the Hugging Face card 55 only explicitly lists ent_coef, n_envs, n_timesteps, policy, and normalize=False. The RL Zoo a2c.yml file, which would contain the full set, was inaccessible.56 Users should verify these in the actual Zoo configuration if possible.
The most prominent tuned parameter explicitly listed is n_envs=8, emphasizing the importance of parallel data collection for A2C's synchronous updates. The total training budget is also significantly increased to 500,000 steps.
Code with Tuned Hyperparameters:

Python


import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# --- Environment Setup ---
env_id = "CartPole-v1"
n_envs = 8 # Use 8 parallel environments as per RL Zoo/HF
env = make_vec_env(env_id, n_envs=n_envs)

# --- Tuned Hyperparameters [55] ---
# Using explicitly listed params from [55] and assuming SB3 defaults for others
a2c_tuned_params = {
    "n_steps": 5, # Default
    "gamma": 0.99, # Default
    "gae_lambda": 1.0, # Default
    "ent_coef": 0.0, # From [55]
    "vf_coef": 0.5, # Default
    "learning_rate": 0.0007, # Default
    "normalize_advantage": False, # From [55] (as normalize=False)
    "max_grad_norm": 0.5, # Default
    "use_rms_prop": True, # Default
    "policy_kwargs": None # Assuming default MLPPolicy architecture
}

# --- Model Instantiation ---
model_a2c_tuned = A2C("MlpPolicy",
                      env,
                      verbose=1,
                      tensorboard_log="./a2c_cartpole_tuned_tensorboard/",
                      **a2c_tuned_params)

# --- Training ---
total_timesteps_a2c_tuned = 500000 # Use RL Zoo/HF's budget
model_a2c_tuned.learn(total_timesteps=total_timesteps_a2c_tuned, progress_bar=True)

# --- Saving ---
model_a2c_tuned.save("a2c_cartpole_tuned_model")
print(f"Tuned A2C model saved to a2c_cartpole_tuned_model.zip")

# --- Cleanup ---
env.close()


7. Evaluating Agent Performance
Once models are trained, evaluating their performance rigorously is crucial for understanding their capabilities and comparing them fairly. SB3 provides tools and guidelines for this process.
The evaluate_policy Function:
This helper function is the standard way to assess a trained agent's performance in SB3.57
Purpose: It runs the agent's policy in the specified environment for a set number of episodes and calculates the average performance, typically measured by the mean return (sum of undiscounted rewards per episode).59
Key Parameters 57:
model: The trained agent (or policy object) to evaluate.
env: The environment instance (Gym or VecEnv) for evaluation. If a VecEnv is used, it should ideally contain only one environment instance for standard evaluation, although the function can handle multiple by distributing episodes.57
n_eval_episodes: The number of complete episodes to run for the evaluation (default: 10).
deterministic: A boolean flag. When True (default), the agent selects actions deterministically (choosing the action with the highest probability or Q-value). When False, it samples actions from the policy's output distribution. For evaluating the learned policy's exploitation capability, deterministic=True is strongly recommended.57
render: Boolean flag to visualize the environment during evaluation (default: False). Note that rendering might require specific environment setup or backend configurations.62
return_episode_rewards: If True, returns lists of individual episode returns and lengths instead of the mean and standard deviation (default: False).
Return Values: By default, it returns a tuple containing the mean episode return and the standard deviation of the episode returns over the n_eval_episodes.57
Evaluation Best Practices:
Simply running the trained agent is often insufficient for reliable benchmarking. Several practices are recommended:
Separate Test Environment: Always evaluate the agent on a separate instance of the environment that was not used during training. This prevents any potential state leakage or adaptation to specific training environment quirks.39
The Monitor / RecordEpisodeStatistics Wrapper: This is crucial. The evaluation environment should ideally be wrapped only with gymnasium.wrappers.RecordEpisodeStatistics (or stable_baselines3.common.monitor.Monitor for older versions). This wrapper accurately tracks the original episode returns and lengths before any other wrappers might modify them (e.g., reward scaling, observation normalization applied during training but not desired during evaluation). evaluate_policy explicitly warns if this wrapper is missing, as results might otherwise be misleading.21
Number of Episodes: Performance in RL can be stochastic. Averaging the return over a sufficient number of evaluation episodes (e.g., n_eval_episodes=10 to 20 or more) provides a more statistically reliable estimate of the agent's true performance.57
Deterministic Actions: As mentioned, using deterministic=True in evaluate_policy assesses the policy's learned optimal behavior without the influence of exploration noise used during training.59
Adhering to these practices ensures that the evaluation accurately reflects the agent's capabilities under standardized conditions, making comparisons between different algorithms or hyperparameter sets meaningful. Failure to do so, for example by evaluating with exploration noise (deterministic=False) or on a training environment with modified rewards/termination, can obscure the true performance and hinder reproducibility.
Code Implementation:

Python


import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO, DQN, A2C # Import necessary algorithms
from stable_baselines3.common.evaluation import evaluate_policy

# --- Environment Setup for Evaluation ---
env_id = "CartPole-v1"
# Create a single environment instance for evaluation
eval_env_raw = gym.make(env_id, render_mode="rgb_array") # Use "human" for visual display
# Wrap with RecordEpisodeStatistics for accurate reward/length tracking
eval_env = RecordEpisodeStatistics(eval_env_raw)

# --- Load Trained Models ---
# Ensure the model files (e.g., ppo_cartpole_tuned_model.zip) exist
try:
    model_ppo_loaded = PPO.load("ppo_cartpole_tuned_model", env=eval_env)
    model_dqn_loaded = DQN.load("dqn_cartpole_tuned_model", env=eval_env)
    model_a2c_loaded = A2C.load("a2c_cartpole_tuned_model", env=eval_env)
except FileNotFoundError as e:
    print(f"Error loading model: {e}. Make sure models were trained and saved.")
    exit()

# --- Evaluate Models ---
n_eval = 20 # Number of episodes for evaluation

print("\n--- Evaluating PPO ---")
mean_reward_ppo, std_reward_ppo = evaluate_policy(model_ppo_loaded, eval_env,
                                                  n_eval_episodes=n_eval,
                                                  deterministic=True) # Use deterministic actions
print(f"PPO Mean Reward ({n_eval} episodes): {mean_reward_ppo:.2f} +/- {std_reward_ppo:.2f}")

print("\n--- Evaluating DQN ---")
mean_reward_dqn, std_reward_dqn = evaluate_policy(model_dqn_loaded, eval_env,
                                                  n_eval_episodes=n_eval,
                                                  deterministic=True)
print(f"DQN Mean Reward ({n_eval} episodes): {mean_reward_dqn:.2f} +/- {std_reward_dqn:.2f}")

print("\n--- Evaluating A2C ---")
mean_reward_a2c, std_reward_a2c = evaluate_policy(model_a2c_loaded, eval_env,
                                                  n_eval_episodes=n_eval,
                                                  deterministic=True)
print(f"A2C Mean Reward ({n_eval} episodes): {mean_reward_a2c:.2f} +/- {std_reward_a2c:.2f}")


# --- Cleanup ---
eval_env.close() # Closes the underlying raw env as well


Interpreting Results: For CartPole-v1, the maximum possible score per episode is 500 (corresponding to 500 steps). An agent achieving a mean reward close to 500 over multiple evaluation episodes is considered to have successfully solved the environment.15 The standard deviation provides insight into the consistency of the agent's performance. Due to the stochastic nature of training and initialization, results can vary slightly between different training runs even with the same hyperparameters.
8. Leveraging the Ecosystem and Next Steps
Stable Baselines3 exists within a broader ecosystem designed to support the RL workflow, from initial experimentation to sharing results and deploying agents.
RL Baselines3 Zoo Revisited:
This companion repository is invaluable not only for providing tuned hyperparameters and pre-trained models but also as a framework for running experiments.3
It offers command-line scripts (train.py, enjoy.py, eval.py) that simplify training, evaluation, and visualization using the stored configurations. For instance, training PPO on CartPole using the Zoo's tuned parameters can be initiated via python -m rl_zoo3.train --algo ppo --env CartPole-v1 (assuming the Zoo is installed and configured).41
Experiment Tracking (Weights & Biases):
Tracking experiments (hyperparameters, metrics, code versions) is crucial for reproducibility and analysis. SB3 integrates seamlessly with Weights & Biases (W&B).
Using the WandbCallback allows automatic logging of hyperparameters, losses, rewards, system metrics, and even videos of the agent playing, directly to the W&B platform during training.21 This provides a centralized dashboard for monitoring and comparing runs.
Python
# Example W&B Integration (requires wandb account and login)
# import wandb
# from wandb.integration.sb3 import WandbCallback
#
# config = { # Define hyperparameters
#     "policy_type": "MlpPolicy",
#     "total_timesteps": 25000,
#     "env_id": "CartPole-v1",
#     "algo": "PPO"
# }
# run = wandb.init(project="sb3_cartpole_example", config=config, sync_tensorboard=True, monitor_gym=True, save_code=True)
#
# model = PPO(config["policy_type"], config["env_id"], verbose=1, tensorboard_log=f"runs/{run.id}")
# model.learn(total_timesteps=config["total_timesteps"], callback=WandbCallback(model_save_path=f"models/{run.id}", verbose=2))
# run.finish()


Model Sharing (Hugging Face Hub):
The Hugging Face Hub serves as a central repository for sharing trained models, including many SB3 agents, particularly those from the RL Zoo.65
The huggingface_sb3 library facilitates downloading pre-trained models using load_from_hub and uploading new models using package_to_hub (which bundles the model, config, evaluation results, and a video) or push_to_hub (for simpler file uploads).65 Pre-trained models from the SB3 team are often hosted under the sb3 organization.65
Python
# Example Hugging Face Hub Download (requires huggingface_sb3)
# from huggingface_sb3 import load_from_hub
# repo_id = "sb3/ppo-CartPole-v1" # Example repo
# filename = "ppo-CartPole-v1.zip" # Model file within the repo
# checkpoint = load_from_hub(repo_id, filename)
# loaded_model = PPO.load(checkpoint)

# Example Hugging Face Hub Upload (requires login and huggingface_sb3)
# from huggingface_sb3 import package_to_hub
# package_to_hub(model=model_ppo_tuned, # Your trained model
#                model_name="my-cool-ppo-cartpole",
#                model_architecture="PPO",
#                env_id=env_id,
#                eval_env=eval_env, # Evaluation env
#                repo_id="<your-hf-username>/my-cool-ppo-cartpole", # Target repo
#                commit_message="Initial commit of tuned PPO CartPole model")


Further Exploration:
SB3 Contrib: For algorithms beyond the core SB3 offerings (e.g., Quantile Regression DQN (QR-DQN), Truncated Quantile Critics (TQC), Trust Region Policy Optimization (TRPO)), the sb3-contrib repository provides implementations following the same SB3 API.3
Other Environments: The principles outlined here for CartPole-v1 can be applied to the vast array of environments available in Gymnasium, or to custom-built environments (ensuring they adhere to the Gymnasium API standard).
Hyperparameter Optimization: For new environments or algorithms where tuned parameters are unavailable, the RL Zoo integrates with Optuna, enabling automated hyperparameter searches to find optimal settings.39
The SB3 ecosystem, encompassing the core library, the Zoo, Contrib, and integrations with MLOps tools like W&B and Hugging Face, provides a robust platform. It supports the full lifecycle of RL research and development, allowing users to start with stable core algorithms, leverage community-tuned parameters, explore experimental methods, track progress meticulously, and share results effectively. This integrated design allows the core library to maintain stability while offering clear pathways for advanced usage and contribution.
9. Conclusion
This report has detailed the process for replicating experiments with PPO, DQN, and A2C algorithms on the CartPole-v1 benchmark using the Stable Baselines3 library. We covered the essential steps: setting up the environment, understanding the core components of SB3 (algorithms, policies, vectorized environments), implementing each algorithm with runnable code, considering the crucial role of hyperparameters, and applying rigorous evaluation techniques.
Summary of Process:
Setup: Install gymnasium and stable-baselines3. Understand the dynamics, state/action spaces, and reward structure of CartPole-v1.
Algorithm Implementation: Instantiate PPO, DQN, or A2C using the "MlpPolicy".
Hyperparameter Tuning: Recognize the sensitivity of RL algorithms to hyperparameters. Utilize resources like the RL Baselines3 Zoo or Hugging Face Hub to find tuned parameters for CartPole-v1, comparing them against SB3 defaults to understand key adjustments. Implement training using these parameters.
Training: Execute the model.learn() method for a sufficient number of total_timesteps, ensuring considerations like learning_starts for DQN are met.
Evaluation: Load the saved model and use evaluate_policy on a separate, RecordEpisodeStatistics-wrapped environment instance, employing deterministic=True and averaging over multiple episodes (n_eval_episodes) for reliable performance assessment.
Key Takeaways:
Reliability through Standardization: Libraries like Stable Baselines3 provide reliable, tested implementations that are crucial for reproducibility and establishing strong baselines in RL research.1
Hyperparameters Matter: Achieving benchmark performance almost always requires moving beyond default hyperparameters. Leveraging community resources like the RL Baselines3 Zoo for tuned parameters is highly effective.39
Rigorous Evaluation is Non-Negotiable: Proper evaluation protocols—using separate test environments, the Monitor/RecordEpisodeStatistics wrapper, deterministic actions, and sufficient evaluation episodes—are essential for obtaining meaningful and comparable results.57
Ecosystem Support: The broader SB3 ecosystem (Zoo, Contrib, W&B, Hugging Face) provides comprehensive support for the entire RL workflow, facilitating experimentation, tuning, tracking, and sharing.3
Final Thoughts: By following the steps and leveraging the tools outlined in this report, researchers and practitioners can confidently replicate standard RL benchmarks like CartPole-v1 using PPO, DQN, and A2C within the Stable Baselines3 framework. This foundation enables further exploration of more complex environments, advanced algorithms, and contributions back to the vibrant open-source reinforcement learning community.
Works cited
Stable-Baselines3: Reliable Reinforcement Learning Implementations - Antonin Raffin, accessed May 3, 2025, https://araffin.github.io/post/sb3/
Stable-Baselines3 Docs - Reliable Reinforcement Learning Implementations — Stable Baselines3 2.6.1a0 documentation, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/
Stable Baselines3, accessed May 3, 2025, https://www.ai4europe.eu/sites/default/files/2021-06/README_5.pdf
DLR-RM/stable-baselines3: PyTorch version of Stable Baselines, reliable implementations of reinforcement learning algorithms. - GitHub, accessed May 3, 2025, https://github.com/DLR-RM/stable-baselines3
stable-baselines3 - PyPI, accessed May 3, 2025, https://pypi.org/project/stable-baselines3/
Reinforcement Learning in Python with Stable Baselines 3 - PythonProgramming.net, accessed May 3, 2025, https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/
Stable-Baselines3: Reliable Reinforcement Learning Implementations, accessed May 3, 2025, https://www.jmlr.org/papers/volume22/20-1364/20-1364.pdf
Releases · DLR-RM/rl-baselines3-zoo - GitHub, accessed May 3, 2025, https://github.com/DLR-RM/rl-baselines3-zoo/releases
Stable Baselines3 Tutorial - Getting Started - Colab, accessed May 3, 2025, https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb
rl-tutorial-jnrr19/1_getting_started.ipynb at sb3 - GitHub, accessed May 3, 2025, https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/1_getting_started.ipynb
SwamiKannan/CartPole-using-Stable-Baselines - GitHub, accessed May 3, 2025, https://github.com/SwamiKannan/CartPole-using-Stable-Baselines
CartPole game Reinforcement Learning - Kaggle, accessed May 3, 2025, https://www.kaggle.com/code/saquib7hussain/cartpole-game-reinforcement-learning
Getting Started — Stable Baselines3 2.6.1a0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html
Getting Started — Stable Baselines3 1.0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v1.0/guide/quickstart.html
Why is my SB3 DQN agent unable to learn CartPole-v1 despite using optimal hyperparameters from RLZoo3? · Issue #472 · DLR-RM/rl-baselines3-zoo - GitHub, accessed May 3, 2025, https://github.com/DLR-RM/rl-baselines3-zoo/issues/472
Why is my SB3 DQN agent unable to learn CartPole-v1 despite using optimal hyperparameters from RLZoo3? : r/reinforcementlearning - Reddit, accessed May 3, 2025, https://www.reddit.com/r/reinforcementlearning/comments/1g2xpq9/why_is_my_sb3_dqn_agent_unable_to_learn/
DQN — Stable Baselines3 1.0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v1.0/modules/dqn.html
Developer Guide — Stable Baselines3 2.6.1a0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/master/guide/developer.html
Custom Policy Network — Stable Baselines3 1.0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v1.0/guide/custom_policy.html
Policy Networks — Stable Baselines3 2.6.1a0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
Stable Baselines 3 - Weights & Biases Documentation - Wandb, accessed May 3, 2025, https://docs.wandb.ai/guides/integrations/stable-baselines-3/
Examples — Stable Baselines3 2.6.1a0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
PPO — Stable Baselines3 2.6.1a0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
A2C — Stable Baselines3 2.0.0 documentation, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v2.0.0/modules/a2c.html
A2C — Stable Baselines3 2.6.1a0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
stable-baselines3/docs/modules/a2c.rst at master - GitHub, accessed May 3, 2025, https://github.com/DLR-RM/stable-baselines3/blob/master/docs/modules/a2c.rst
Stable Baselines3 - Easy Multiprocessing - Colab, accessed May 3, 2025, https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb
PPO — Stable Baselines3 2.1.0 documentation, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v2.1.0/modules/ppo.html
DQN — Stable Baselines3 2.6.1a0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
ppo.py - DLR-RM/stable-baselines3 · GitHub, accessed May 3, 2025, https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
DQN — Stable Baselines3 2.0.0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v2.0.0/modules/dqn.html
stable-baselines3/stable_baselines3/dqn/dqn.py at master - GitHub, accessed May 3, 2025, https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py
Custom Policy Network — Stable Baselines3 0.8.0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/sde/guide/custom_policy.html
PPO — Stable Baselines3 2.4.0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v2.4.0/modules/ppo.html
PPO — Stable Baselines3 1.0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v1.0/modules/ppo.html
PPO — Stable Baselines3 0.8.0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/sde/modules/ppo.html
Stable Baselines 3: Default parameters - Stack Overflow, accessed May 3, 2025, https://stackoverflow.com/questions/75509729/stable-baselines-3-default-parameters
Stable Baselines3 PPO() - how to change clip_range parameter during training?, accessed May 3, 2025, https://stackoverflow.com/questions/72483775/stable-baselines3-ppo-how-to-change-clip-range-parameter-during-training
Stable Baselines3 Tutorial - Callbacks and hyperparameter tuning - Google Colab, accessed May 3, 2025, https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/4_callbacks_hyperparameter_tuning.ipynb
RL Baselines3 Zoo, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v0.11.1/guide/rl_zoo.html
RL Baselines3 Zoo, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html
DLR-RM/rl-baselines3-zoo: A training framework for Stable Baselines3 reinforcement learning agents, with hyperparameter optimization and pre-trained agents included. - GitHub, accessed May 3, 2025, https://github.com/DLR-RM/rl-baselines3-zoo
Hyperparameter Tuning - DLR-RM/rl-baselines3-zoo - GitHub, accessed May 3, 2025, https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/docs/guide/tuning.rst
sb3/ppo-CartPole-v1 · Hugging Face, accessed May 3, 2025, https://huggingface.co/sb3/ppo-CartPole-v1
DQN — Stable Baselines3 0.11.1 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v0.11.1/modules/dqn.html
DQN — Stable Baselines3 0.8.0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/sde/modules/dqn.html
DQN example fails · Issue #526 · DLR-RM/stable-baselines3 - GitHub, accessed May 3, 2025, https://github.com/DLR-RM/stable-baselines3/issues/526
DQN and Double DQN with Stable-Baselines3 - Colab, accessed May 3, 2025, https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/dqn_sb3.ipynb
How to get the Q-values in DQN in stable baseline 3? - Stack Overflow, accessed May 3, 2025, https://stackoverflow.com/questions/73239501/how-to-get-the-q-values-in-dqn-in-stable-baseline-3
sb3/dqn-CartPole-v1 · Hugging Face, accessed May 3, 2025, https://huggingface.co/sb3/dqn-CartPole-v1
A2C — Stable Baselines3 2.1.0 documentation, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v2.1.0/modules/a2c.html
A2C — Stable Baselines3 1.0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v1.0/modules/a2c.html
A2C — Stable Baselines3 0.8.0 documentation, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/sde/modules/a2c.html
stable-baselines3/stable_baselines3/a2c/a2c.py at master - GitHub, accessed May 3, 2025, https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/a2c/a2c.py
sb3/a2c-CartPole-v1 · Hugging Face, accessed May 3, 2025, https://huggingface.co/sb3/a2c-CartPole-v1
accessed December 31, 1969, https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparameters/a2c.yml
Evaluation Helper — Stable Baselines3 0.11.1 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v0.11.1/common/evaluation.html
Evaluation Helper — Stable Baselines3 0.8.0 documentation, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/sde/common/evaluation.html
Evaluation Helper — Stable Baselines3 2.6.1a0 documentation, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html
stable_baselines3.common.evaluation — Stable Baselines3 0.11.1 documentation, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v0.11.1/_modules/stable_baselines3/common/evaluation.html
Reinforcement Learning Tips and Tricks - Stable Baselines3 - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
[Question] evaluate_policy with render=True not working · Issue #404 · DLR-RM/stable-baselines3 - GitHub, accessed May 3, 2025, https://github.com/DLR-RM/stable-baselines3/issues/404
renee127/CartPole-v1-optuna-parameters - Hugging Face, accessed May 3, 2025, https://huggingface.co/renee127/CartPole-v1-optuna-parameters
rl-baselines-zoo.ipynb - Colab - Google, accessed May 3, 2025, https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb
Integrations — Stable Baselines3 2.6.1a0 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/master/guide/integrations.html
Using Stable-Baselines3 at Hugging Face, accessed May 3, 2025, https://huggingface.co/docs/hub/stable-baselines3
Integrations — Stable Baselines3 2.2.1 documentation - Read the Docs, accessed May 3, 2025, https://stable-baselines3.readthedocs.io/en/v2.2.1/guide/integrations.html
Welcome to Stable Baselines3 Contrib docs! — Stable Baselines3 - Contrib 2.6.0 documentation, accessed May 3, 2025, https://sb3-contrib.readthedocs.io/
