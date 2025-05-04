# DQN Version - Copy these cells into your notebook

# For imports cell:
'''
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
'''

# For model creation cell:
'''
env = gym.make("CartPole-v1", render_mode="rgb_array")
model = DQN(MlpPolicy, env, verbose=1)
'''

# For evaluation before training cell:
'''
# Use a separate environment for evaluation
eval_env = gym.make("CartPole-v1", render_mode="rgb_array")

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
'''

# For training cell:
'''
# Train the agent for 50000 steps
model.learn(total_timesteps=50_000)
'''

# For evaluation after training cell:
'''
# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
'''

# For video recording cell (replace the existing record_video function):
'''
define record_video(env_id, model, video_length=500, prefix="", video_folder="videos/"):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make("CartPole-v1", render_mode="rgb_array")])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()
'''

# For recording video cell:
'''
record_video("CartPole-v1", model, video_length=500, prefix="dqn-cartpole")
'''

# For showing video cell:
'''
show_videos("videos", prefix="dqn")
'''
