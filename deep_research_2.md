Replicating Reinforcement Learning Benchmarks on CartPole-v1 using Stable Baselines3I. IntroductionReinforcement Learning (RL) has demonstrated significant success in solving complex sequential decision-making problems. However, reproducing results reported in the literature can be challenging due to variations in implementation details, hyperparameters, and evaluation protocols.1 The Stable Baselines3 (SB3) library aims to address this by providing reliable, well-tested, and easy-to-use implementations of state-of-the-art RL algorithms in PyTorch.3 Complementing SB3, the RL Baselines3 Zoo offers a collection of pre-trained agents and tuned hyperparameters for various standard benchmark environments, further facilitating reproducibility and benchmarking.6This report provides a comprehensive guide on how to replicate benchmark results for three common RL algorithms – Proximal Policy Optimization (PPO), Deep Q-Network (DQN), and Advantage Actor-Critic (A2C) – on the classic CartPole-v1 control environment using Stable Baselines3. The CartPole problem, where the objective is to balance a pole upright on a moving cart, serves as a fundamental benchmark for evaluating RL algorithms.9 This report details the necessary setup, explains core SB3 concepts relevant to this task, provides specific code implementations for each algorithm leveraging hyperparameters from the RL Baselines3 Zoo where available, outlines standardized evaluation procedures, and offers recommendations for robust experimentation. The goal is to enable users to reliably recreate these experiments and understand the key components involved in using the SB3 ecosystem for benchmark replication.II. Environment Setup and Core DependenciesReproducing RL experiments requires careful attention to the software environment, including library versions and dependencies. This section outlines the installation process for SB3 and the RL Baselines3 Zoo, explains how to instantiate the target environment (CartPole-v1) using the Gymnasium library, and introduces the concept of Vectorized Environments used by SB3.

Installation: Stable Baselines3 and the RL Baselines3 Zoo can be installed using pip. It is highly recommended to use a dedicated virtual environment (e.g., using venv or conda) to manage dependencies and avoid conflicts. There are two primary ways to install the Zoo:

As a package: This provides access to the core functionalities and scripts.
Bashpip install rl_zoo3 stable-baselines3[extra] gymnasium


Cloning the repository: This grants full access to the codebase, including hyperparameter files and pre-trained agents (if cloned recursively). This is often preferred for direct use of the Zoo's scripts and hyperparameter files.7
Bashgit clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo
cd rl-baselines3-zoo
pip install -e.[plots,tests] # Installs in editable mode with optional dependencies
# Ensure SB3 and Gymnasium are also installed if not covered by Zoo dependencies
pip install stable-baselines3[extra] gymnasium



The transition from the older gym library to gymnasium is a significant factor in dependency management.12 Stable Baselines3 versions 2.0.0 and later rely on Gymnasium.12 The RL Baselines3 Zoo also specifies dependencies on particular versions of SB3, Gymnasium, and other libraries like sb3_contrib and huggingface_sb3 in its setup files.14 Ensuring that the installed versions of SB3, the Zoo, and Gymnasium are compatible is absolutely critical for successful replication, as mismatches can lead to errors or unexpected behavior.12 Using a virtual environment and potentially pinning specific versions based on the Zoo's requirements 14 is the most reliable approach.


Environment Instantiation: The CartPole-v1 environment is instantiated using the Gymnasium library. Gymnasium serves as the standard interface for RL environments, providing methods to reset the environment, take steps (actions), and receive observations, rewards, and termination/truncation signals.10 The render_mode parameter can be set to "human" for live visualization or "rgb_array" to get pixel data, useful for recording videos.16
Pythonimport gymnasium as gym

env_id = "CartPole-v1"
# Create the environment with human rendering
env = gym.make(env_id, render_mode="human")

# Reset the environment to get the initial observation
observation, info = env.reset()
print(f"Initial Observation: {observation}")

# Example: Take a random action (0=left, 1=right)
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
print(f"Action: {action}, Next Observation: {observation}, Reward: {reward}, Done: {terminated or truncated}")

# It's good practice to close the environment when finished
env.close()



Vectorized Environments (VecEnv): Stable Baselines3 heavily utilizes the concept of Vectorized Environments (VecEnv) internally, especially for on-policy algorithms like PPO and A2C.16 A VecEnv essentially wraps multiple individual environments, allowing the agent to collect experiences from several environments in parallel.11 This significantly speeds up data collection and can improve training stability by averaging gradients over more diverse transitions. SB3 provides helpers like make_vec_env to easily create VecEnv instances.17 The two main types are DummyVecEnv (runs environments sequentially in the main process) and SubprocVecEnv (runs each environment in a separate process, offering true parallelism but with higher overhead).17 For CPU-bound tasks like CartPole with simple policies, SubprocVecEnv can leverage multiple CPU cores effectively.17
Pythonfrom stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

env_id = "CartPole-v1"

# Create 4 parallel environments using DummyVecEnv (default)
vec_env_dummy = make_vec_env(env_id, n_envs=4)
print(f"Using DummyVecEnv with {vec_env_dummy.num_envs} environments.")
obs_dummy = vec_env_dummy.reset()
print(f"Initial observations shape (DummyVecEnv): {obs_dummy.shape}")
vec_env_dummy.close()

# Create 4 parallel environments using SubprocVecEnv
# This requires the code to be run inside `if __name__ == '__main__':` block on some OS
# vec_env_subproc = make_vec_env(env_id, n_envs=4, vec_env_cls=SubprocVecEnv)
# print(f"Using SubprocVecEnv with {vec_env_subproc.num_envs} environments.")
# obs_subproc = vec_env_subproc.reset()
# print(f"Initial observations shape (SubprocVecEnv): {obs_subproc.shape}")
# vec_env_subproc.close()


III. Core Concepts in Stable Baselines3 for CartPoleStable Baselines3 provides standardized components and relies on conventions that simplify the process of setting up and training RL agents. Understanding the default policy networks and the role of the RL Baselines3 Zoo's hyperparameters is key to replicating benchmarks effectively.

Policy Networks (MlpPolicy): For environments like CartPole-v1, which have a 1-dimensional, continuous observation space (specifically Box(4,) representing cart position, cart velocity, pole angle, pole angular velocity) 10, Stable Baselines3 defaults to using the MlpPolicy.16 This policy utilizes a Multi-Layer Perceptron (MLP) – a standard feedforward neural network – to map observations to actions (for the actor/policy) and state values (for the critic, in actor-critic methods). For PPO, DQN, and A2C, the default MlpPolicy architecture consists of two hidden layers, each containing 64 units (neurons).21 The activation function used between these layers is typically the Rectified Linear Unit (ReLU).21 Separate linear layers are added on top of this shared or separate base network to produce the final action probabilities (or Q-values for DQN) and state-value estimates.21 While this default architecture works well for many standard tasks, SB3 allows customization of the network architecture (number of layers, units per layer, activation function) via the policy_kwargs argument during model instantiation.21
Pythonfrom stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn

env_id = "CartPole-v1"
vec_env = make_vec_env(env_id, n_envs=4)

# Using the default MlpPolicy architecture (2 layers of 64 units)
model_default = PPO("MlpPolicy", vec_env, verbose=0)
print("Default PPO Policy Architecture:")
print(model_default.policy)

# Example: Specifying a custom architecture (e.g., 2 layers of 128 units)
# policy_kwargs_custom = dict(net_arch=dict(pi=, vf=), activation_fn=nn.Tanh)
# model_custom = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs_custom, verbose=0)
# print("\nCustom PPO Policy Architecture:")
# print(model_custom.policy)

vec_env.close()



RL Baselines3 Zoo Hyperparameters: The RL Baselines3 Zoo plays a crucial role in achieving benchmark performance by providing a curated set of tuned hyperparameters.6 These hyperparameters, optimized for specific algorithm-environment combinations, are stored in YAML files within the Zoo's hyperparams directory (e.g., hyperparams/ppo.yml, hyperparams/dqn.yml, hyperparams/a2c.yml).7 Using these tuned parameters is generally the recommended approach when aiming to replicate published results or achieve strong performance without extensive manual tuning.11 The Zoo's train.py script is designed to automatically parse these YAML files and instantiate the SB3 models with the specified hyperparameters when training is initiated via the command line.7 For programmatic use, these parameters can be loaded from the YAML files and passed during model initialization.

The combination of standardized default policies like MlpPolicy 21 and the readily available tuned hyperparameters from the Zoo 23 significantly lowers the barrier to entry for using and benchmarking RL algorithms.1 This standardization is a core strength of the SB3 ecosystem, promoting reproducibility and allowing researchers and practitioners to quickly get started.1 However, it's important to recognize that RL algorithms can be highly sensitive to both hyperparameters and network architectures.2 While the defaults and Zoo parameters provide excellent starting points and often yield strong results on benchmarks like CartPole, achieving state-of-the-art performance on novel or more complex problems frequently necessitates further, potentially extensive, hyperparameter optimization and architecture search.11 The Zoo itself facilitates this through its integration with tools like Optuna.8 Thus, users should view the provided parameters as validated baselines for known problems, understanding that customization may be required for different challenges.IV. Proximal Policy Optimization (PPO) ImplementationPPO is a widely used, state-of-the-art RL algorithm known for its balance of sample efficiency, stability, and ease of implementation.18 It is an on-policy, actor-critic method that improves upon algorithms like A2C by using a clipped surrogate objective function to prevent excessively large policy updates, thus promoting more stable learning.18

Algorithm Overview (SB3 Context): In Stable Baselines3, PPO is implemented to work effectively with vectorized environments (VecEnv) for parallel data collection.18 It supports both discrete (like CartPole) and continuous action spaces. Key features include Generalized Advantage Estimation (GAE) for variance reduction and optional value function clipping. It is generally considered robust and often requires less hyperparameter tuning than other algorithms for good performance on many tasks.11


Code Implementation: The following code demonstrates how to initialize, train, save, and load a PPO agent on CartPole-v1 using SB3, incorporating hyperparameters sourced from the RL Baselines3 Zoo.
Pythonimport gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time

env_id = "CartPole-v1"
# PPO benefits from parallel environments for stable gradient estimates
vec_env = make_vec_env(env_id, n_envs=4)

# Hyperparameters from RL Baselines3 Zoo [23]
ppo_params = {
    'n_steps': 2048,        # Num steps per env before update
    'batch_size': 64,         # Minibatch size for SGD
    'n_epochs': 10,           # Num epochs when optimizing surrogate loss
    'gamma': 0.99,          # Discount factor
    'gae_lambda': 0.95,       # Factor for GAE
    'clip_range': 0.2,        # Clipping parameter PPO
    'ent_coef': 0.0,          # Entropy coefficient
    'vf_coef': 0.5,           # Value function coefficient
    'max_grad_norm': 0.5,     # Max norm for gradient clipping
    'learning_rate': 0.0003,  # Learning rate
    'policy_kwargs': None     # Use default MlpPolicy architecture
}
# Total training timesteps from RL Baselines3 Zoo [23]
total_timesteps = 1.0e5

# Instantiate the PPO agent
model = PPO("MlpPolicy", vec_env, verbose=1, **ppo_params)

# Train the agent
print(f"Training PPO on {env_id} for {int(total_timesteps)} timesteps...")
start_time = time.time()
# log_interval=10 means logging will happen every 10 updates
# An update happens every n_envs * n_steps timesteps
model.learn(total_timesteps=int(total_timesteps), log_interval=10)
end_time = time.time()
print(f"PPO Training finished in {end_time - start_time:.2f} seconds.")

# Save the trained model
model_path = "ppo_cartpole_model.zip"
model.save(model_path)
print(f"PPO model saved to {model_path}")

# Demonstrate loading the model
del model # Remove the current model instance
loaded_model = PPO.load(model_path, env=vec_env)
print("PPO model loaded successfully.")
# Verify the loaded model can predict actions
obs = vec_env.reset()
action, _states = loaded_model.predict(obs, deterministic=True)
print(f"Loaded PPO model predicted action: {action}")

vec_env.close()



Hyperparameters (RL Zoo): The RL Baselines3 Zoo provides a set of tuned hyperparameters specifically for PPO on CartPole-v1, found in hyperparams/ppo.yml.23 Using these parameters is recommended for attempting to replicate benchmark performance.
Table 1: PPO Hyperparameters for CartPole-v1 (Source: RL Baselines3 Zoo 23)

HyperparameterValueDescriptionn_timesteps1.0×105Total number of environment interactions for training.policyMlpPolicyStandard MLP policy network for vector observations.learning_rate0.0003Step size for the Adam optimizer.n_steps2048Number of steps collected per environment before each policy update.batch_size64Size of the minibatch used for Stochastic Gradient Descent (SGD) updates.n_epochs10Number of optimization epochs performed on the collected data per update.gamma0.99Discount factor for future rewards.gae_lambda0.95Factor for trade-off in Generalized Advantage Estimation (GAE).clip_range0.2PPO clipping parameter, limits policy change per update.ent_coef0.0Entropy regularization coefficient (encourages exploration).vf_coef0.5Value function loss coefficient in the total loss.max_grad_norm0.5Maximum norm for gradient clipping to prevent exploding gradients.policy_kwargsnullNo custom arguments passed; uses default MlpPolicy architecture.The interaction between `n_steps`, `batch_size`, and `n_epochs` is particularly important for PPO's performance and stability.[18, 23] The total amount of experience gathered before an update is `n_envs * n_steps`. This data roll-out is then iterated over `n_epochs` times, using mini-batches of size `batch_size` for the SGD updates.[18] The Zoo parameters for CartPole (`n_steps=2048`, `batch_size=64`, `n_epochs=10`) [23] indicate a strategy of collecting a relatively large amount of data per update cycle and then performing multiple passes over this data. This approach likely helps stabilize learning by reducing the variance in gradient estimates and allowing the policy and value function to fit the collected data more thoroughly before new, potentially different, data is gathered. Modifying one of these parameters often requires compensatory adjustments to the others to maintain a similar learning dynamic.
V. Deep Q-Network (DQN) ImplementationDQN is a foundational value-based RL algorithm that achieved breakthrough performance on Atari games.20 It is an off-policy algorithm, meaning it learns from transitions stored in a replay buffer, which helps break correlations in the data and improves sample efficiency.

Algorithm Overview (SB3 Context): The SB3 implementation of DQN is primarily designed for environments with discrete action spaces.20 Key components include the replay buffer, a target network (a periodically updated copy of the main Q-network used to stabilize learning), and an epsilon-greedy exploration strategy (where the agent chooses a random action with probability epsilon, which typically decays over time, and the greedy action otherwise).20 While DQN doesn't inherently require parallel environments for the same reasons as on-policy methods, using VecEnv can still be beneficial for speeding up evaluation or potentially for more complex data collection schemes.


Code Implementation: This code shows the setup, training, saving, and loading process for DQN on CartPole-v1, using the hyperparameters provided by the RL Baselines3 Zoo.
Pythonimport gymnasium as gym
from stable_baselines3 import DQN
import time

env_id = "CartPole-v1"
# DQN is off-policy and uses a replay buffer, often trained on a single environment instance
env = gym.make(env_id)

# Hyperparameters from RL Baselines3 Zoo [24]
dqn_params = {
    'learning_rate': 0.0001,        # Learning rate for the Adam optimizer
    'buffer_size': 1000000,       # Size of the replay buffer
    'learning_starts': 1000,        # Timesteps before learning begins
    'batch_size': 32,           # Minibatch size sampled from buffer
    'tau': 1.0,             # Strength of "soft" target network update (1.0 = hard update)
    'gamma': 0.99,            # Discount factor
    'train_freq': 10,           # Update the model every 'train_freq' steps
    'gradient_steps': 1,          # How many gradient steps to perform per update
    'target_update_interval': 10000, # Timesteps between updating target network
    'exploration_fraction': 0.1,    # Fraction of total timesteps over which epsilon decays
    'exploration_final_eps': 0.05,  # Final value of epsilon
    'policy_kwargs': None       # Use default MlpPolicy architecture
}
# Total training timesteps from RL Baselines3 Zoo [24]
total_timesteps = 20000.0

# Instantiate the DQN agent
model = DQN("MlpPolicy", env, verbose=1, **dqn_params)

# Train the agent
print(f"Training DQN on {env_id} for {int(total_timesteps)} timesteps...")
start_time = time.time()
# Default log_interval=4 from SB3 DQN example [20], logs every 4 episodes
model.learn(total_timesteps=int(total_timesteps), log_interval=4)
end_time = time.time()
print(f"DQN Training finished in {end_time - start_time:.2f} seconds.")

# Save the trained model
model_path = "dqn_cartpole_model.zip"
model.save(model_path)
print(f"DQN model saved to {model_path}")

# Demonstrate loading the model
del model # Remove the current model instance
loaded_model = DQN.load(model_path, env=env)
print("DQN model loaded successfully.")
# Verify the loaded model can predict actions
obs, info = env.reset()
action, _states = loaded_model.predict(obs, deterministic=True)
print(f"Loaded DQN model predicted action: {action}")

env.close()



Hyperparameters (RL Zoo): The hyperparams/dqn.yml file in the RL Baselines3 Zoo contains tuned parameters for DQN on CartPole-v1.24
Table 2: DQN Hyperparameters for CartPole-v1 (Source: RL Baselines3 Zoo 24)

HyperparameterValueDescriptionn_timesteps2.0×104Total number of environment interactions for training.policyMlpPolicyStandard MLP policy network for Q-value estimation.learning_rate0.0001Step size for the Adam optimizer.buffer_size1,000,000Maximum size of the replay buffer (stores past experiences).learning_starts1,000Number of steps to collect before starting training updates.batch_size32Number of transitions sampled from the replay buffer per update.tau1.0Target network update rate (1.0 means hard update, copy weights).gamma0.99Discount factor for future rewards.train_freq10Frequency (in steps) at which the Q-network is updated.gradient_steps1Number of gradient updates performed per training frequency.target_update_interval10,000Frequency (in steps) at which the target network is updated.exploration_fraction0.1Fraction of training time over which epsilon decreases linearly.exploration_final_eps0.05Final value of epsilon after decay (minimum exploration rate).policy_kwargsnullNo custom arguments passed; uses default MlpPolicy architecture.Experience suggests that DQN's performance, even on seemingly simple tasks like CartPole, can be quite sensitive to the choice of hyperparameters.[28, 29] Parameters governing the replay buffer (`buffer_size`, `learning_starts`), the target network updates (`target_update_interval`, `tau`), the learning process (`learning_rate`, `batch_size`, `train_freq`, `gradient_steps`), and especially the exploration strategy (`exploration_fraction`, `exploration_final_eps`) can all significantly influence convergence speed and final performance.[24, 29] While the Zoo parameters [24] provide a validated starting point, achieving optimal results might sometimes require further tuning. Some reports suggest CartPole can be unexpectedly challenging for standard DQN configurations [28], potentially requiring adjustments to network size or more aggressive training schedules (e.g., more frequent updates or gradient steps) than the Zoo defaults.[29] This highlights the empirical nature of RL and the importance of careful tuning, even for classic benchmarks.
VI. Advantage Actor-Critic (A2C) ImplementationA2C is the synchronous counterpart to the Asynchronous Advantage Actor-Critic (A3C) algorithm.17 It is an on-policy, actor-critic method that, like PPO, benefits from parallel environments to stabilize learning.

Algorithm Overview (SB3 Context): The SB3 implementation of A2C uses multiple workers (via VecEnv) to collect experience simultaneously.17 It updates the policy and value function after a fixed number of steps (n_steps) per environment. Compared to PPO, A2C is conceptually simpler as it lacks the clipping mechanism, but this can sometimes lead to less stable training or lower sample efficiency.11 Notably, the SB3 documentation warns about potential instability and recommends considering the RMSpropTFLike optimizer (mimicking the TensorFlow implementation used in the original Stable Baselines) if issues arise or if aiming to match results from that older library.2


Code Implementation: The following code sets up, trains, saves, and loads an A2C agent. Since the specific tuned hyperparameters from a2c.yml were inaccessible 32, this implementation relies on the default parameters provided by SB3 17 and information gleaned from the Hugging Face Hub model card for a2c-CartPole-v1.27
Pythonimport gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import time

env_id = "CartPole-v1"
# A2C benefits from parallel envs. The Hugging Face model card [27] used n_envs=8.
# A2C default n_steps is small (5), so more envs help gather sufficient data per update.
n_envs = 8
vec_env = make_vec_env(env_id, n_envs=n_envs)

# Hyperparameters based on SB3 defaults [17] and Hugging Face info [27]
# Note: Tuned hyperparams from rl-baselines3-zoo/hyperparams/a2c.yml were not accessible [32]
a2c_params = {
    'n_steps': 5,           # Default A2C: Num steps per env before update
    'gamma': 0.99,          # Default: Discount factor
    'gae_lambda': 1.0,       # Default: GAE factor (1.0 = standard advantage)
    'ent_coef': 0.0,          # Default: Entropy coefficient
    'vf_coef': 0.5,           # Default: Value function coefficient
    'max_grad_norm': 0.5,     # Default: Max norm for gradient clipping
    'learning_rate': 0.0007,  # Default: Learning rate
    'use_rms_prop': True,     # Default: Use RMSprop optimizer
    'policy_kwargs': None     # Use default MlpPolicy architecture
    # Consider policy_kwargs={'optimizer_class': RMSpropTFLike, 'optimizer_kwargs': dict(eps=1e-5)}
    # if matching older Stable Baselines results or facing instability [17]
}
# Total training timesteps from Hugging Face model card [27]
total_timesteps = 500000.0

# Instantiate the A2C agent
model = A2C("MlpPolicy", vec_env, verbose=1, **a2c_params)

# Train the agent
print(f"Training A2C on {env_id} for {int(total_timesteps)} timesteps...")
start_time = time.time()
# Log less frequently due to longer training duration
# An update happens every n_envs * n_steps = 8 * 5 = 40 timesteps
model.learn(total_timesteps=int(total_timesteps), log_interval=100) # Log every 100 updates
end_time = time.time()
print(f"A2C Training finished in {end_time - start_time:.2f} seconds.")

# Save the trained model
model_path = "a2c_cartpole_model.zip"
model.save(model_path)
print(f"A2C model saved to {model_path}")

# Demonstrate loading the model
del model # Remove the current model instance
loaded_model = A2C.load(model_path, env=vec_env)
print("A2C model loaded successfully.")
# Verify the loaded model can predict actions
obs = vec_env.reset()
action, _states = loaded_model.predict(obs, deterministic=True)
print(f"Loaded A2C model predicted action: {action}")

vec_env.close()



Hyperparameters (RL Zoo/Defaults): Due to the inaccessibility of the a2c.yml file 32, the hyperparameters presented below are primarily the defaults specified in the SB3 A2C documentation 17, supplemented with information from the corresponding model card on the Hugging Face Hub.27 These should be considered a reasonable starting point rather than fully optimized parameters like those available for PPO and DQN from the Zoo.
Table 3: A2C Hyperparameters for CartPole-v1 (Source: SB3 Defaults 17 & Hugging Face 27)


HyperparameterValueSourceDescriptionn_timesteps5.0×105Hugging Face 27Total number of environment interactions for training.policyMlpPolicyHugging Face 27Standard MLP policy network.n_envs8Hugging Face 27Number of parallel environments used for training.learning_rate0.0007SB3 Default 17Step size for the RMSprop optimizer.n_steps5SB3 Default 17Number of steps collected per environment before each policy update.gamma0.99SB3 Default 17Discount factor for future rewards.gae_lambda1.0SB3 Default 17GAE factor (1.0 means standard advantage estimation, no GAE).ent_coef0.0SB3 Default 17Entropy regularization coefficient.vf_coef0.5SB3 Default 17Value function loss coefficient.max_grad_norm0.5SB3 Default 17Maximum norm for gradient clipping.use_rms_propTrueSB3 Default 17Use RMSprop optimizer (vs. Adam).policy_kwargsnullAssumed DefaultNo custom arguments passed; uses default MlpPolicy architecture.
When tuned hyperparameters from a reliable source like the Zoo are unavailable, relying on the library's defaults is a common fallback strategy.[17] However, A2C's default `n_steps=5` is notably small, especially compared to PPO's default/tuned value of 2048.[23] This means A2C performs updates based on very short trajectories, which can increase the variance of gradient estimates and potentially lead to less stable learning. Using a larger number of parallel environments (like `n_envs=8` from the Hugging Face model [27]) helps compensate by providing more data points per update cycle (`n_envs * n_steps`). The recommendation to potentially use `RMSpropTFLike` [17] underscores that A2C can be sensitive to subtle implementation details of the optimizer, a known challenge in ensuring RL reproducibility across different frameworks or versions.[2] The relatively high number of timesteps (500k) reported on the Hugging Face card [27] might reflect the need for extended training to overcome the challenges posed by the small `n_steps` and achieve robust performance on CartPole.
VII. Evaluating Agent PerformanceOnce agents are trained, their performance must be evaluated systematically to compare algorithms and determine if the benchmark criteria have been met. Stable Baselines3 provides a convenient utility for this purpose.

Standard Evaluation Method: The recommended way to evaluate a trained agent in SB3 is using the evaluate_policy function found in stable_baselines3.common.evaluation.19 This function runs the agent's policy in the environment for a specified number of episodes and computes statistics on the obtained rewards.


Key Parameters: Understanding the parameters of evaluate_policy is crucial for conducting meaningful evaluations 33:

model: The trained RL agent (e.g., the loaded PPO, DQN, or A2C model object).
env: The environment used for evaluation. Critically, this should ideally be a separate instance from the training environment(s). This is particularly important if wrappers like VecNormalize were used during training, as evaluation should typically be done with normalization statistics frozen or using saved running averages. For the basic CartPole-v1 without normalization, using gym.make(env_id) is generally sufficient. Wrapping the evaluation environment with Monitor from stable_baselines3.common.monitor is good practice for accurate tracking of episode rewards and lengths, especially if other wrappers are present.33
n_eval_episodes: The number of complete episodes to run for the evaluation. Averaging over a sufficient number (e.g., 100) provides more statistically reliable results.33
deterministic: A boolean indicating whether the agent should select actions deterministically (typically, choosing the action with the highest probability or Q-value) or stochastically (sampling from the policy's output distribution or using epsilon-greedy). For reporting final benchmark performance, deterministic=True is standard practice as it measures the learned policy's exploitation capability.11 Setting it to False evaluates performance including exploration noise.
return_episode_rewards: If False (default), returns the mean and standard deviation of the rewards per episode. If True, returns lists of individual episode rewards and lengths.33
Other parameters like render, callback, reward_threshold, and warn offer further customization.33



Code Example: The following snippet demonstrates how to load a saved model and evaluate it using evaluate_policy.
Pythonimport gymnasium as gym
from stable_baselines3 import PPO # Or DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env_id = "CartPole-v1"
# Create a separate environment instance for evaluation
eval_env = gym.make(env_id)
# Wrap with Monitor for accurate episode statistics
eval_env = Monitor(eval_env)

# Load a previously saved model (replace with the desired model path)
model_path = "ppo_cartpole_model.zip" # Or "dqn_cartpole_model.zip", "a2c_cartpole_model.zip"
try:
    # Determine algorithm from file name or context (adjust as needed)
    if "ppo" in model_path: from stable_baselines3 import PPO as Algo
    elif "dqn" in model_path: from stable_baselines3 import DQN as Algo
    elif "a2c" in model_path: from stable_baselines3 import A2C as Algo
    else: raise ValueError("Could not determine algorithm from model path")

    model = Algo.load(model_path, env=eval_env) # Pass eval_env if needed for VecNormalize stats
    print(f"Evaluating model: {model_path}")

    # Evaluate the agent deterministically over 100 episodes
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
    print(f"\n--- Evaluation Results (Deterministic) ---")
    print(f"Algorithm: {model.__class__.__name__}")
    print(f"Mean reward over 100 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Optionally, evaluate stochastically
    mean_reward_stoch, std_reward_stoch = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=False)
    print(f"\n--- Evaluation Results (Stochastic) ---")
    print(f"Mean reward over 100 episodes: {mean_reward_stoch:.2f} +/- {std_reward_stoch:.2f}")

except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
except Exception as e:
    print(f"An error occurred during evaluation: {e}")

eval_env.close()



Interpreting Results: The CartPole-v1 environment is typically considered "solved" when an agent achieves an average reward of 475 or higher over 100 consecutive evaluation episodes. The maximum possible score per episode in CartPole-v1 is 500 (as the episode terminates after 500 steps even if the pole is still balanced). Therefore, achieving a mean reward close to 500 consistently indicates successful learning.27 When using the tuned hyperparameters from the RL Baselines3 Zoo for PPO 23 and DQN 24, or the defaults/Hugging Face parameters for A2C 27, the expected outcome of the evaluation should approach this target score.

Comparing deterministic (deterministic=True) and stochastic (deterministic=False) evaluation results can provide useful information. Deterministic evaluation measures the pure exploitation performance of the learned policy, which is the standard for reporting benchmark scores.33 Stochastic evaluation reflects performance under the exploration strategy inherent in the policy (sampling for PPO/A2C, epsilon-greedy for DQN).11 A significant drop in performance during stochastic evaluation compared to deterministic evaluation might suggest issues such as insufficient decay of exploration noise during training (e.g., epsilon remaining too high for DQN) or a policy distribution that is overly broad or "peaked" in a suboptimal way. Analyzing both provides a more complete understanding of the agent's behavior.VIII. Consolidated Replication Script and RecommendationsTo facilitate practical replication, this section provides a consolidated Python script combining the training and evaluation steps for PPO, DQN, and A2C on CartPole-v1. It also offers recommendations for extending these basic experiments and adopting robust RL practices.

Runnable Python Script: The script below defines functions to train and evaluate each algorithm sequentially, using the hyperparameters identified previously (Zoo-tuned for PPO/DQN, Defaults/HF for A2C).
Pythonimport gymnasium as gym
import time
import os
import numpy as np

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike # Uncomment if needed for A2C

# --- Configuration ---
ENV_ID = "CartPole-v1"
N_EVAL_EPISODES = 100
SEED = 42 # Set a seed for reproducibility

# --- PPO Configuration ---
PPO_PARAMS = {
    'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99,
    'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.0, 'vf_coef': 0.5,
    'max_grad_norm': 0.5, 'learning_rate': 0.0003, 'policy_kwargs': None, 'seed': SEED
}
PPO_TIMESTEPS = 1.0e5
PPO_N_ENVS = 4
PPO_MODEL_PATH = "ppo_cartpole_model.zip"

# --- DQN Configuration ---
DQN_PARAMS = {
    'learning_rate': 0.0001, 'buffer_size': 1000000, 'learning_starts': 1000,
    'batch_size': 32, 'tau': 1.0, 'gamma': 0.99, 'train_freq': 10,
    'gradient_steps': 1, 'target_update_interval': 10000,
    'exploration_fraction': 0.1, 'exploration_final_eps': 0.05,
    'policy_kwargs': None, 'seed': SEED
}
DQN_TIMESTEPS = 20000.0
DQN_MODEL_PATH = "dqn_cartpole_model.zip"

# --- A2C Configuration ---
A2C_PARAMS = {
    'n_steps': 5, 'gamma': 0.99, 'gae_lambda': 1.0, 'ent_coef': 0.0,
    'vf_coef': 0.5, 'max_grad_norm': 0.5, 'learning_rate': 0.0007,
    'use_rms_prop': True, 'policy_kwargs': None, 'seed': SEED
    # 'policy_kwargs': dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)), # Optional
}
A2C_TIMESTEPS = 500000.0
A2C_N_ENVS = 8
A2C_MODEL_PATH = "a2c_cartpole_model.zip"

# --- Helper Functions ---
def train_agent(algo_class, params, total_timesteps, model_path, env_id, n_envs=1):
    """Trains an SB3 agent."""
    print(f"\n--- Training {algo_class.__name__} ---")
    if n_envs > 1:
        env = make_vec_env(env_id, n_envs=n_envs, seed=SEED)
    else:
        env = Monitor(gym.make(env_id)) # Monitor wrapper for single env training stats
        env.reset(seed=SEED)

    model = algo_class("MlpPolicy", env, verbose=0, **params) # verbose=1 for logs

    start_time = time.time()
    model.learn(total_timesteps=int(total_timesteps), log_interval=200) # Adjust log_interval
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    model.save(model_path)
    print(f"Model saved to {model_path}")
    env.close()
    return model_path

def evaluate_agent(algo_class, model_path, env_id, n_eval_episodes):
    """Evaluates a saved SB3 agent."""
    print(f"\n--- Evaluating {algo_class.__name__} ---")
    eval_env = Monitor(gym.make(env_id)) # Use Monitor for evaluation
    try:
        model = algo_class.load(model_path, env=eval_env) # Pass env if needed for VecNormalize stats
        mean_reward, std_reward = evaluate_policy(model, eval_env,
                                                  n_eval_episodes=n_eval_episodes,
                                                  deterministic=True)
        print(f"Mean reward (deterministic): {mean_reward:.2f} +/- {std_reward:.2f}")

        mean_reward_s, std_reward_s = evaluate_policy(model, eval_env,
                                                      n_eval_episodes=n_eval_episodes,
                                                      deterministic=False)
        print(f"Mean reward (stochastic):  {mean_reward_s:.2f} +/- {std_reward_s:.2f}")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Skipping evaluation.")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
    finally:
        eval_env.close()

# --- Main Execution ---
if __name__ == "__main__":
    # Train and Evaluate PPO
    ppo_path = train_agent(PPO, PPO_PARAMS, PPO_TIMESTEPS, PPO_MODEL_PATH, ENV_ID, n_envs=PPO_N_ENVS)
    evaluate_agent(PPO, ppo_path, ENV_ID, N_EVAL_EPISODES)

    # Train and Evaluate DQN
    dqn_path = train_agent(DQN, DQN_PARAMS, DQN_TIMESTEPS, DQN_MODEL_PATH, ENV_ID, n_envs=1)
    evaluate_agent(DQN, dqn_path, ENV_ID, N_EVAL_EPISODES)

    # Train and Evaluate A2C
    a2c_path = train_agent(A2C, A2C_PARAMS, A2C_TIMESTEPS, A2C_MODEL_PATH, ENV_ID, n_envs=A2C_N_ENVS)
    evaluate_agent(A2C, a2c_path, ENV_ID, N_EVAL_EPISODES)

    print("\n--- Script Finished ---")



Recommendations & Further Steps: While the script provides a direct replication path, robust RL experimentation often involves more sophisticated practices:

Hyperparameter Tuning: The provided hyperparameters are effective starting points, but optimal performance, especially on new environments, often requires tuning.11 The RL Baselines3 Zoo integrates with Optuna, enabling automated hyperparameter searches.8 Exploring this feature is recommended for maximizing agent performance.
Callbacks: Stable Baselines3 offers a powerful callback system for monitoring and controlling training.19 The EvalCallback is particularly useful; it periodically evaluates the agent on a separate environment and saves the best-performing model based on evaluation scores.19 This ensures that the best model found during training is preserved, even if performance degrades later (a common issue in RL 11). Other callbacks like StopTrainingOnRewardThreshold can automate stopping criteria.
Python# Example Usage of EvalCallback (inside training function)
from stable_baselines3.common.callbacks import EvalCallback
eval_env_for_callback = Monitor(gym.make(env_id))
eval_callback = EvalCallback(eval_env_for_callback, best_model_save_path=f'./logs/best_{algo_class.__name__}/',
                             log_path=f'./logs/results_{algo_class.__name__}/', eval_freq=max(int(total_timesteps / 20 / n_envs), 500), # Eval 20 times
                             n_eval_episodes=20, deterministic=True, render=False)
# model.learn(..., callback=eval_callback)


Logging and Visualization: SB3 integrates seamlessly with TensorBoard. By specifying a log directory via the tensorboard_log parameter during model instantiation, detailed metrics like losses, rewards, episode lengths, and exploration parameters can be logged and visualized.6 This is invaluable for understanding learning dynamics, debugging issues, and comparing different runs or hyperparameters.
Environment Wrappers: While CartPole-v1 requires minimal preprocessing, many RL tasks benefit significantly from environment wrappers. VecNormalize standardizes observations and optionally normalizes rewards, often crucial for continuous control tasks using algorithms like PPO or A2C.11 Custom wrappers can be created to modify observations, actions, or rewards (reward shaping).36 Always ensure the Monitor wrapper is applied correctly for accurate logging, especially when other wrappers are used.33
Troubleshooting and Robustness: If replication results deviate significantly from expectations, first verify library versions (SB3, Gymnasium, Zoo).12 RL training is inherently stochastic 11; running experiments with multiple random seeds and averaging results is essential for reliable conclusions. Increasing the training budget (total_timesteps) can sometimes improve performance, especially if the default budget is insufficient for convergence.11 Consulting the SB3 and Zoo documentation, GitHub issues, and community forums (like the RL Discord mentioned in 5) can provide solutions to common problems.


Adopting these practices moves beyond simple single-run replication towards more rigorous and reliable RL experimentation. The infrastructure provided by SB3 and the Zoo, including callbacks, logging integration, and hyperparameter optimization tools, strongly supports these best practices.6IX. ConclusionThis report has outlined a systematic approach to replicating benchmark performance for PPO, DQN, and A2C algorithms on the CartPole-v1 environment using the Stable Baselines3 library and resources from the RL Baselines3 Zoo. By following the steps detailed – including careful environment setup, leveraging standardized components like MlpPolicy, utilizing tuned hyperparameters from the Zoo (or sensible defaults), and employing the evaluate_policy function for assessment – users can reliably recreate these fundamental RL experiments.The key takeaways emphasize the significant value provided by the Stable Baselines3 ecosystem. Its focus on reliable implementations, clear API, and integration with tools like the RL Baselines3 Zoo streamlines the process of benchmarking and reproducing RL results.1 The availability of tuned hyperparameters lowers the barrier to achieving strong performance on standard tasks, while the library's flexibility allows for customization when needed. The importance of meticulous dependency management and standardized evaluation protocols was also highlighted as crucial for reproducibility.While this report focused on CartPole-v1, the principles and tools discussed are broadly applicable. Users are encouraged to extend these methods to other environments within the Gymnasium suite, experiment with customizing policy architectures 21, delve into automated hyperparameter optimization using the Zoo's Optuna integration 8, and utilize callbacks and logging for more robust experimentation. The Stable Baselines3 framework provides a solid foundation for both learning RL concepts and conducting reliable research in the field.