# tests/test_ppo_training.py
import os
import pytest
import gymnasium as gym
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from src.env_utils import make_cartpole_vec_env, make_eval_env


@pytest.fixture
def temp_model_path():
    """Create a temporary model path for testing."""
    path = "./models/test_ppo_model.zip"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yield path
    # Cleanup: remove the file if it exists
    if os.path.exists(path):
        os.remove(path)


def test_ppo_instantiation():
    """Test that PPO can be instantiated with our environment."""
    env = make_cartpole_vec_env(n_envs=2, seed=42)
    model = PPO("MlpPolicy", env, verbose=0)
    assert model is not None
    assert model.policy is not None
    assert model.env is not None


def test_ppo_short_training():
    """Test that PPO can be trained for a small number of steps."""
    env = make_cartpole_vec_env(n_envs=2, seed=42)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=0,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=64,
        gamma=0.99,
        seed=42
    )
    
    # Train for a small number of steps
    model.learn(total_timesteps=500)
    
    # Check that the model has been updated
    # PPO processes data in batches, so the actual number of timesteps
    # may be slightly higher than requested (rounded up to complete the last batch)
    assert model.num_timesteps >= 500


def test_ppo_save_load(temp_model_path):
    """Test that PPO models can be saved and loaded."""
    # Create and train a model
    env = make_cartpole_vec_env(n_envs=2, seed=42)
    model = PPO("MlpPolicy", env, verbose=0, seed=42)
    model.learn(total_timesteps=500)
    
    # Save the model
    model.save(temp_model_path)
    assert os.path.exists(temp_model_path), f"Model was not saved to {temp_model_path}"
    
    # Load the model
    loaded_model = PPO.load(temp_model_path, env=env)
    assert loaded_model is not None
    assert loaded_model.policy is not None
    assert loaded_model.env is not None


def test_ppo_prediction():
    """Test that PPO can make predictions after training."""
    # Create and train a model
    env = make_cartpole_vec_env(n_envs=1, seed=42)
    model = PPO("MlpPolicy", env, verbose=0, seed=42)
    model.learn(total_timesteps=500)
    
    # Get an observation
    obs = env.reset()
    
    # Make a prediction
    action, _states = model.predict(obs, deterministic=True)
    
    # Check that the action has the correct shape
    assert action.shape == (1,)
    assert action[0] in [0, 1]  # CartPole has two actions


def test_ppo_evaluation():
    """Test that PPO can be evaluated."""
    # Create and train a model
    train_env = make_cartpole_vec_env(n_envs=2, seed=42)
    model = PPO("MlpPolicy", train_env, verbose=0, seed=42)
    model.learn(total_timesteps=500)
    
    # Create an evaluation environment
    eval_env = make_eval_env(seed=43)
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=5, deterministic=True
    )
    
    # Check that the evaluation returned valid results
    assert isinstance(mean_reward, float)
    assert isinstance(std_reward, float)
    assert not np.isnan(mean_reward)
    assert not np.isnan(std_reward)
