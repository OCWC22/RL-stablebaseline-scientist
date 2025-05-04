import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
model = A2C('MlpPolicy', env, verbose=1)

# Evaluate untrained model
eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Before training: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent
model.learn(total_timesteps=50_000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"After training: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
