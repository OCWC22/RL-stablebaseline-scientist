#!/usr/bin/env python3
"""
Hyperparameter tuning script for PPO algorithm on CartPole-v1 environment.

This script uses Optuna to automatically search for the best hyperparameters
for the PPO algorithm on the CartPole-v1 environment.

References:
- PPO paper: https://arxiv.org/abs/1707.06347
- SB3 PPO documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- Optuna documentation: https://optuna.readthedocs.io/
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

# Import our environment utilities
from src.env_utils import make_cartpole_vec_env, make_eval_env


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tune PPO hyperparameters on CartPole-v1")
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
        default=50000,
        help="Number of timesteps per trial",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="ppo_cartpole",
        help="Name of the Optuna study",
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
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Verbosity level (0, 1, or 2)"
    )
    return parser.parse_args()


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters for PPO."""
    # Learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    
    # Discount factor
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    
    # GAE lambda parameter
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    
    # Clipping parameter
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    
    # Number of steps to collect per environment before updating
    n_steps = trial.suggest_int("n_steps", 32, 2048, log=True)
    
    # Number of epochs for optimization
    n_epochs = trial.suggest_int("n_epochs", 3, 30)
    
    # Minibatch size
    batch_size = trial.suggest_int("batch_size", 32, 512, log=True)
    
    # Entropy coefficient
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
    
    # Value function coefficient
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0)
    
    # Max gradient norm
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 0.7)
    
    # Network architecture
    net_arch_size = trial.suggest_categorical("net_arch_size", ["small", "medium", "large"])
    if net_arch_size == "small":
        net_arch = [dict(pi=[64], vf=[64])]
    elif net_arch_size == "medium":
        net_arch = [dict(pi=[64, 64], vf=[64, 64])]
    else:  # large
        net_arch = [dict(pi=[128, 64], vf=[128, 64])]
    
    # Activation function
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    if activation_fn_name == "tanh":
        activation_fn = torch.nn.Tanh
    else:  # relu
        activation_fn = torch.nn.ReLU
    
    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
        ),
    }


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Objective function for Optuna optimization."""
    # Sample hyperparameters
    hyperparams = sample_ppo_params(trial)
    
    # Create output directory for this trial
    os.makedirs(args.output_dir, exist_ok=True)
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Create vectorized training environment
    env = make_cartpole_vec_env(n_envs=4, seed=args.seed)
    
    # Create separate environment for evaluation
    eval_env = make_eval_env(seed=args.seed + 1000)
    eval_env = Monitor(eval_env)
    
    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=trial_dir,
        log_path=trial_dir,
        eval_freq=max(args.n_timesteps // (args.n_evaluations * 4), 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    
    # Create and train the model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=args.verbose,
        seed=args.seed,
        tensorboard_log=os.path.join(trial_dir, "tensorboard"),
        **hyperparams
    )
    
    try:
        model.learn(
            total_timesteps=args.n_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )
    except Exception as e:
        # Handle any exceptions during training
        print(f"Trial {trial.number} failed with error: {e}")
        return float("-inf")
    
    # Get the mean reward from the last evaluation
    mean_reward = eval_callback.best_mean_reward
    
    # Save the hyperparameters and results
    results = {
        "trial_number": trial.number,
        "mean_reward": mean_reward,
        **hyperparams
    }
    
    # Save results to a JSON file
    import json
    with open(os.path.join(trial_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # Save the best model
    if mean_reward > 450:  # CartPole-v1 is considered solved at 475
        model.save(os.path.join(trial_dir, "final_model"))
    
    return mean_reward


def main():
    """Main function for hyperparameter tuning."""
    args = parse_args()
    
    # Create the Optuna study
    sampler = TPESampler(n_startup_trials=args.n_startup_trials, seed=args.seed)
    pruner = MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    
    # Optimize the objective function
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=None,  # No timeout
        show_progress_bar=True,
    )
    
    # Print the best hyperparameters
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print(f"Best value: {study.best_value}")
    
    # Save the best hyperparameters to a file
    os.makedirs(args.output_dir, exist_ok=True)
    best_params_file = os.path.join(args.output_dir, f"{args.study_name}_best_params.json")
    
    import json
    with open(best_params_file, "w") as f:
        json.dump(study.best_params, f, indent=4)
    
    print(f"Best hyperparameters saved to {best_params_file}")
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Plot optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(args.output_dir, f"{args.study_name}_history.png"))
        
        # Plot parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(args.output_dir, f"{args.study_name}_importance.png"))
        
        print(f"Plots saved to {args.output_dir}")
    except ImportError:
        print("Plotting requires matplotlib and plotly. Install them to enable plotting.")


if __name__ == "__main__":
    main()
