#!/usr/bin/env python3
"""
Optimized A2C training script for CartPole-v1 environment.

This script implements a production-ready training pipeline for A2C using Stable Baselines3,
with optimized hyperparameters specifically tuned for the CartPole-v1 environment.

References:
- A2C paper: https://arxiv.org/abs/1602.01783
- SB3 A2C documentation: https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import torch
from pathlib import Path
from datetime import datetime

# Stable Baselines3 imports
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

# Import our environment utilities
from src.env_utils import make_cartpole_vec_env, make_eval_env


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train an optimized A2C agent on CartPole-v1")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--n-envs", type=int, default=16, help="Number of parallel environments"
    )
    parser.add_argument(
        "--use-subproc",
        action="store_true",
        help="Use SubprocVecEnv instead of DummyVecEnv",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Save model every x timesteps",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluate model every x timesteps",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="logs",
        help="Tensorboard log directory",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Verbosity level (0, 1, or 2)"
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Create timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"optimized_a2c_cartpole_{timestamp}"
    model_path = os.path.join(args.output_dir, model_name)

    # Create vectorized training environment
    env = make_cartpole_vec_env(
        n_envs=args.n_envs, seed=args.seed, use_subproc=args.use_subproc
    )

    # Create separate environment for evaluation
    eval_env = make_eval_env(seed=args.seed + 1000)  # Different seed for eval
    eval_env = Monitor(eval_env)  # Wrap with Monitor for recording

    # Set up callbacks
    # Save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,  # Divide by n_envs because callback counts steps per env
        save_path=args.output_dir,
        name_prefix=model_name,
        save_vecnormalize=True,
    )

    # Evaluate the model periodically and save the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.output_dir, "best"),
        log_path=os.path.join(args.tensorboard_log, "evaluations"),
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # Define optimized A2C hyperparameters for CartPole-v1
    # These values are based on research and best practices for CartPole
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.001,           # Higher learning rate for faster convergence
        n_steps=8,                     # Shorter rollout for more frequent updates
        gamma=0.99,                    # Standard discount factor
        gae_lambda=0.95,               # Standard GAE lambda
        ent_coef=0.01,                 # Increased entropy coefficient for better exploration
        vf_coef=0.5,                   # Standard value function coefficient
        max_grad_norm=0.5,             # Standard gradient clipping
        rms_prop_eps=1e-5,             # Standard RMSprop epsilon
        use_rms_prop=True,             # Use RMSprop optimizer
        normalize_advantage=True,      # Normalize advantages for more stable training
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])],  # Deeper network
            activation_fn=torch.nn.ReLU,  # ReLU activation
        ),
        tensorboard_log=args.tensorboard_log,
        verbose=args.verbose,
        seed=args.seed,
    )

    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save the final model
    model.save(os.path.join(args.output_dir, f"{model_name}_final"))

    print(f"Training completed. Final model saved to {model_path}_final.zip")
    print(f"Best model saved to {os.path.join(args.output_dir, 'best')}")


if __name__ == "__main__":
    main()
