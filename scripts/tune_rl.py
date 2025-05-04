#!/usr/bin/env python3
"""
Unified command-line interface for hyperparameter tuning of RL algorithms.

This script provides a single entry point for tuning PPO, A2C, and DQN algorithms
on the CartPole-v1 environment using Optuna.

Usage:
    python tune_rl.py --algorithm ppo --n-trials 100
    python tune_rl.py --algorithm a2c --n-trials 50 --study-name my_a2c_study
    python tune_rl.py --algorithm dqn --n-trials 100 --storage sqlite:///tuning.db
"""

import os
import argparse
import importlib
from typing import Dict, Any, Optional

# Import tuning modules
from scripts.tune_ppo import main as run_ppo_tuning
from scripts.tune_a2c import main as run_a2c_tuning
from scripts.tune_dqn import main as run_dqn_tuning


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified interface for RL algorithm hyperparameter tuning")
    
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["ppo", "a2c", "dqn"],
        help="RL algorithm to tune (ppo, a2c, or dqn)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter search",
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=10,
        help="Number of startup trials before pruning begins",
    )
    parser.add_argument(
        "--n-evaluations",
        type=int,
        default=5,
        help="Number of evaluations per trial",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=None,
        help="Number of timesteps per trial (defaults to algorithm-specific value if not specified)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name of the Optuna study (defaults to algorithm name if not specified)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for Optuna storage (SQLite or MySQL)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tuning_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", 
        type=int, 
        default=1, 
        help="Verbosity level (0, 1, or 2)"
    )
    
    return parser.parse_args()


def get_default_timesteps(algorithm: str) -> int:
    """Get default number of timesteps for each algorithm."""
    defaults = {
        "ppo": 50000,
        "a2c": 50000,
        "dqn": 100000,  # DQN typically needs more steps
    }
    return defaults[algorithm]


def get_default_study_name(algorithm: str) -> str:
    """Get default study name for each algorithm."""
    return f"{algorithm}_cartpole"


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    """Prepare arguments with defaults based on the selected algorithm."""
    # Create a copy of the args to modify
    modified_args = argparse.Namespace(**vars(args))
    
    # Set algorithm-specific defaults if not provided
    if modified_args.n_timesteps is None:
        modified_args.n_timesteps = get_default_timesteps(modified_args.algorithm)
        
    if modified_args.study_name is None:
        modified_args.study_name = get_default_study_name(modified_args.algorithm)
        
    # Create algorithm-specific output directory
    modified_args.output_dir = os.path.join(
        modified_args.output_dir, 
        modified_args.algorithm
    )
    
    return modified_args


def main() -> None:
    """Main function for the unified tuning interface."""
    # Parse and prepare arguments
    args = parse_args()
    modified_args = prepare_args(args)
    
    # Map algorithms to their tuning functions
    tuning_functions = {
        "ppo": run_ppo_tuning,
        "a2c": run_a2c_tuning,
        "dqn": run_dqn_tuning,
    }
    
    # Get the tuning function for the selected algorithm
    tuning_function = tuning_functions[args.algorithm]
    
    # Override sys.argv to pass the modified args to the algorithm-specific main function
    import sys
    original_argv = sys.argv
    
    try:
        # Convert namespace to command-line arguments
        algorithm_args = []
        for key, value in vars(modified_args).items():
            if key != "algorithm":  # Skip the algorithm argument
                if value is not None:  # Skip None values
                    algorithm_args.append(f"--{key.replace('_', '-')}")
                    algorithm_args.append(str(value))
        
        # Replace sys.argv with the algorithm-specific arguments
        sys.argv = [f"tune_{args.algorithm}.py"] + algorithm_args
        
        # Run the tuning function
        print(f"Starting hyperparameter tuning for {args.algorithm.upper()} algorithm")
        print(f"Using {modified_args.n_trials} trials with {modified_args.n_timesteps} timesteps each")
        print(f"Results will be saved to {modified_args.output_dir}")
        print("-" * 80)
        
        tuning_function()
        
        print("-" * 80)
        print(f"Tuning completed for {args.algorithm.upper()} algorithm")
        print(f"Results saved to {modified_args.output_dir}")
        
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()
