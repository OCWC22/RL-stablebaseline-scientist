# tests/test_tune_a2c.py
import os
import pytest
import optuna
import numpy as np
import torch
import gymnasium as gym
import json
from unittest.mock import patch, MagicMock, mock_open

from scripts.tune_a2c import sample_a2c_params, objective


@pytest.fixture
def mock_args():
    """Create mock arguments for testing."""
    class Args:
        n_timesteps = 100  # Very small for testing
        n_evaluations = 1
        seed = 42
        verbose = 0
        output_dir = "./test_tuning_results"
    
    # Create output directory if it doesn't exist
    os.makedirs("./test_tuning_results", exist_ok=True)
    
    yield Args()
    
    # Cleanup: remove test directory if it exists
    import shutil
    if os.path.exists("./test_tuning_results"):
        shutil.rmtree("./test_tuning_results")


def test_sample_a2c_params():
    """Test that hyperparameters can be sampled correctly."""
    # Create a trial
    study = optuna.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    
    # Sample hyperparameters
    params = sample_a2c_params(trial)
    
    # Check that all expected parameters are present
    assert "learning_rate" in params
    assert "n_steps" in params
    assert "gamma" in params
    assert "gae_lambda" in params
    assert "ent_coef" in params
    assert "vf_coef" in params
    assert "max_grad_norm" in params
    assert "use_rms_prop" in params
    assert "normalize_advantage" in params
    assert "use_gae" in params
    assert "policy_kwargs" in params
    
    # Check that policy_kwargs contains expected keys
    assert "net_arch" in params["policy_kwargs"]
    assert "activation_fn" in params["policy_kwargs"]
    assert "ortho_init" in params["policy_kwargs"]


@pytest.mark.parametrize("net_arch_size,activation,use_gae", [
    ("small", "tanh", True),
    ("medium", "relu", False),
    ("large", "tanh", True),
])
def test_sample_a2c_params_variations(net_arch_size, activation, use_gae):
    """Test different variations of hyperparameters."""
    # Create a study and trial
    study = optuna.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    
    # Mock suggest_categorical to return our test values
    original_suggest_categorical = trial.suggest_categorical
    
    def mock_suggest_categorical(name, choices):
        if name == "net_arch_size":
            return net_arch_size
        elif name == "activation_fn":
            return activation
        elif name == "use_gae":
            return use_gae
        else:
            return original_suggest_categorical(name, choices)
    
    trial.suggest_categorical = mock_suggest_categorical
    
    # Sample hyperparameters
    params = sample_a2c_params(trial)
    
    # Check that the network architecture matches the expected size
    if net_arch_size == "small":
        assert params["policy_kwargs"]["net_arch"] == [dict(pi=[64], vf=[64])]
    elif net_arch_size == "medium":
        assert params["policy_kwargs"]["net_arch"] == [dict(pi=[64, 64], vf=[64, 64])]
    else:  # large
        assert params["policy_kwargs"]["net_arch"] == [dict(pi=[128, 64], vf=[128, 64])]
    
    # Check that the activation function matches the expected type
    if activation == "tanh":
        assert params["policy_kwargs"]["activation_fn"] == torch.nn.Tanh
    else:  # relu
        assert params["policy_kwargs"]["activation_fn"] == torch.nn.ReLU
    
    # Check that use_gae is set correctly
    assert params["use_gae"] == use_gae


@patch("scripts.tune_a2c.make_cartpole_vec_env")
@patch("scripts.tune_a2c.make_eval_env")
@patch("scripts.tune_a2c.A2C")
@patch("scripts.tune_a2c.EvalCallback")
@patch("scripts.tune_a2c.Monitor")
@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_objective_function(mock_json_dump, mock_file, mock_monitor, mock_eval_callback, mock_a2c, mock_make_eval_env, mock_make_cartpole_vec_env, mock_args):
    """Test that the objective function works correctly."""
    # Mock the A2C model and evaluation callback
    mock_model = mock_a2c.return_value
    mock_model.learn.return_value = None
    
    mock_callback = mock_eval_callback.return_value
    mock_callback.best_mean_reward = 450.0  # Good reward for CartPole-v1
    
    # Mock the monitor to return itself
    mock_monitor.return_value = mock_monitor
    
    # Create a trial
    study = optuna.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    
    # Run the objective function
    reward = objective(trial, mock_args)
    
    # Check that the reward matches the expected value
    assert reward == 450.0
    
    # Check that the model was trained with the correct parameters
    mock_a2c.assert_called_once()
    mock_model.learn.assert_called_once()
    
    # Check that the evaluation callback was created with the correct parameters
    mock_eval_callback.assert_called_once()


@patch("scripts.tune_a2c.make_cartpole_vec_env")
@patch("scripts.tune_a2c.make_eval_env")
@patch("scripts.tune_a2c.A2C")
@patch("scripts.tune_a2c.EvalCallback")
@patch("scripts.tune_a2c.Monitor")
@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_objective_function_exception(mock_json_dump, mock_file, mock_monitor, mock_eval_callback, mock_a2c, mock_make_eval_env, mock_make_cartpole_vec_env, mock_args):
    """Test that the objective function handles exceptions correctly."""
    # Mock the A2C model to raise an exception during training
    mock_model = mock_a2c.return_value
    mock_model.learn.side_effect = Exception("Test exception")
    
    # Mock the monitor to return itself
    mock_monitor.return_value = mock_monitor
    
    # Create a trial
    study = optuna.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    
    # Run the objective function
    reward = objective(trial, mock_args)
    
    # Check that the reward is -inf when an exception occurs
    assert reward == float("-inf")
    
    # Check that the model was created but learn was not completed
    mock_a2c.assert_called_once()
    mock_model.learn.assert_called_once()
