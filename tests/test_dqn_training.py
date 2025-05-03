# tests/test_dqn_training.py
import os
import pytest
import gymnasium as gym
import numpy as np
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from src.env_utils import make_eval_env


@pytest.fixture
def temp_model_path():
    """Create a temporary model path for testing."""
    path = "./models/test_dqn_model.zip"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yield path
    # Cleanup: remove the file if it exists
    if os.path.exists(path):
        os.remove(path)


def test_dqn_instantiation():
    """Test that DQN can be instantiated with our environment."""
    env = gym.make("CartPole-v1")
    model = DQN("MlpPolicy", env, verbose=0)
    assert model is not None
    assert model.policy is not None
    assert model.env is not None


def test_dqn_short_training():
    """Test that DQN can be trained for a small number of steps."""
    env = gym.make("CartPole-v1")
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=0,
        learning_rate=2.3e-3,
        buffer_size=1000,  # Smaller buffer for faster testing
        learning_starts=50,  # Start learning earlier for testing
        batch_size=64,
        gamma=0.99,
        seed=42
    )
    
    # Train for a small number of steps
    model.learn(total_timesteps=200)
    
    # Check that the model has been updated
    assert model.num_timesteps == 200


def test_dqn_save_load(temp_model_path):
    """Test that DQN models can be saved and loaded."""
    # Create and train a model
    env = gym.make("CartPole-v1")
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=0, 
        seed=42,
        learning_starts=50,  # Start learning earlier for testing
        buffer_size=1000     # Smaller buffer for faster testing
    )
    model.learn(total_timesteps=200)
    
    # Save the model
    model.save(temp_model_path)
    assert os.path.exists(temp_model_path), f"Model was not saved to {temp_model_path}"
    
    # Load the model
    loaded_model = DQN.load(temp_model_path, env=env)
    assert loaded_model is not None
    assert loaded_model.policy is not None
    assert loaded_model.env is not None


def test_dqn_prediction():
    """Test that DQN can make predictions after training."""
    # Create and train a model
    env = gym.make("CartPole-v1")
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=0, 
        seed=42,
        learning_starts=50,  # Start learning earlier for testing
        buffer_size=1000     # Smaller buffer for faster testing
    )
    model.learn(total_timesteps=200)
    
    # Get an observation
    obs, _ = env.reset(seed=42)
    
    # Make a prediction
    action, _states = model.predict(obs, deterministic=True)
    
    # Check that the action is valid
    assert action in [0, 1]  # CartPole has two actions


def test_dqn_evaluation():
    """Test that DQN can be evaluated."""
    # Create and train a model
    train_env = gym.make("CartPole-v1")
    model = DQN(
        "MlpPolicy", 
        train_env, 
        verbose=0, 
        seed=42,
        learning_starts=50,  # Start learning earlier for testing
        buffer_size=1000     # Smaller buffer for faster testing
    )
    model.learn(total_timesteps=200)
    
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
