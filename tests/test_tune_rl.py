# tests/test_tune_rl.py
import os
import sys
import pytest
import argparse
from unittest.mock import patch, MagicMock

from scripts.tune_rl import prepare_args, get_default_timesteps, get_default_study_name, main


def test_get_default_timesteps():
    """Test that default timesteps are returned correctly for each algorithm."""
    assert get_default_timesteps("ppo") == 50000
    assert get_default_timesteps("a2c") == 50000
    assert get_default_timesteps("dqn") == 100000


def test_get_default_study_name():
    """Test that default study names are returned correctly for each algorithm."""
    assert get_default_study_name("ppo") == "ppo_cartpole"
    assert get_default_study_name("a2c") == "a2c_cartpole"
    assert get_default_study_name("dqn") == "dqn_cartpole"


def test_prepare_args():
    """Test that arguments are prepared correctly with defaults."""
    # Create a proper argparse.Namespace object
    args = argparse.Namespace(
        algorithm="ppo",
        n_trials=10,
        n_startup_trials=5,
        n_evaluations=3,
        n_timesteps=None,  # Should be set to default
        study_name=None,  # Should be set to default
        storage=None,
        output_dir="test_output",
        seed=42,
        verbose=1
    )
    
    modified_args = prepare_args(args)
    
    # Check that defaults were applied correctly
    assert modified_args.n_timesteps == 50000
    assert modified_args.study_name == "ppo_cartpole"
    assert modified_args.output_dir == os.path.join("test_output", "ppo")
    
    # Check that provided values are preserved
    assert modified_args.n_trials == 10
    assert modified_args.seed == 42


@patch("scripts.tune_rl.run_ppo_tuning")
@patch("scripts.tune_rl.run_a2c_tuning")
@patch("scripts.tune_rl.run_dqn_tuning")
@patch("scripts.tune_rl.parse_args")
def test_main_ppo(mock_parse_args, mock_run_dqn, mock_run_a2c, mock_run_ppo):
    """Test that the main function calls the correct tuning function for PPO."""
    # Create a proper argparse.Namespace object
    mock_args = argparse.Namespace(
        algorithm="ppo",
        n_trials=10,
        n_startup_trials=5,
        n_evaluations=3,
        n_timesteps=1000,
        study_name="test_study",
        storage=None,
        output_dir="test_output",
        seed=42,
        verbose=1
    )
    
    mock_parse_args.return_value = mock_args
    
    # Save original sys.argv
    original_argv = sys.argv
    
    try:
        # Run the main function
        main()
        
        # Check that the correct tuning function was called
        mock_run_ppo.assert_called_once()
        mock_run_a2c.assert_not_called()
        mock_run_dqn.assert_not_called()
        
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


@patch("scripts.tune_rl.run_ppo_tuning")
@patch("scripts.tune_rl.run_a2c_tuning")
@patch("scripts.tune_rl.run_dqn_tuning")
@patch("scripts.tune_rl.parse_args")
def test_main_a2c(mock_parse_args, mock_run_dqn, mock_run_a2c, mock_run_ppo):
    """Test that the main function calls the correct tuning function for A2C."""
    # Create a proper argparse.Namespace object
    mock_args = argparse.Namespace(
        algorithm="a2c",
        n_trials=10,
        n_startup_trials=5,
        n_evaluations=3,
        n_timesteps=1000,
        study_name="test_study",
        storage=None,
        output_dir="test_output",
        seed=42,
        verbose=1
    )
    
    mock_parse_args.return_value = mock_args
    
    # Save original sys.argv
    original_argv = sys.argv
    
    try:
        # Run the main function
        main()
        
        # Check that the correct tuning function was called
        mock_run_ppo.assert_not_called()
        mock_run_a2c.assert_called_once()
        mock_run_dqn.assert_not_called()
        
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


@patch("scripts.tune_rl.run_ppo_tuning")
@patch("scripts.tune_rl.run_a2c_tuning")
@patch("scripts.tune_rl.run_dqn_tuning")
@patch("scripts.tune_rl.parse_args")
def test_main_dqn(mock_parse_args, mock_run_dqn, mock_run_a2c, mock_run_ppo):
    """Test that the main function calls the correct tuning function for DQN."""
    # Create a proper argparse.Namespace object
    mock_args = argparse.Namespace(
        algorithm="dqn",
        n_trials=10,
        n_startup_trials=5,
        n_evaluations=3,
        n_timesteps=1000,
        study_name="test_study",
        storage=None,
        output_dir="test_output",
        seed=42,
        verbose=1
    )
    
    mock_parse_args.return_value = mock_args
    
    # Save original sys.argv
    original_argv = sys.argv
    
    try:
        # Run the main function
        main()
        
        # Check that the correct tuning function was called
        mock_run_ppo.assert_not_called()
        mock_run_a2c.assert_not_called()
        mock_run_dqn.assert_called_once()
        
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
