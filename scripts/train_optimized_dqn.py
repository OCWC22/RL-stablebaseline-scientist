#!/usr/bin/env python3
"""
Optimized DQN training script for CartPole-v1 environment.

This script implements a production-ready training pipeline for DQN using Stable Baselines3,
with optimized hyperparameters specifically tuned for the CartPole-v1 environment.

References:
- DQN paper: https://arxiv.org/abs/1312.5602
- Rainbow DQN paper: https://arxiv.org/abs/1710.02298
- SB3 DQN documentation: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import torch
from pathlib import Path
from datetime import datetime

# Stable Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Import our environment utilities
from src.env_utils import make_eval_env


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train an optimized DQN agent on CartPole-v1")
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
    model_name = f"optimized_dqn_cartpole_{timestamp}"
    model_path = os.path.join(args.output_dir, model_name)

    # Create training environment
    env = make_eval_env(seed=args.seed)

    # Create separate environment for evaluation
    eval_env = make_eval_env(seed=args.seed + 1000)  # Different seed for eval
    eval_env = Monitor(eval_env)  # Wrap with Monitor for recording

    # Set up callbacks
    # Save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.output_dir,
        name_prefix=model_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Evaluate the model periodically and save the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.output_dir, "best"),
        log_path=os.path.join(args.tensorboard_log, "evaluations"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # Define optimized DQN hyperparameters for CartPole-v1
    # These values are based on research and best practices for CartPole
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,            # Higher learning rate for faster convergence
        buffer_size=50000,             # Moderate buffer size
        learning_starts=1000,          # Start learning earlier
        batch_size=128,                # Larger batch size for stable updates
        gamma=0.99,                    # Standard discount factor
        train_freq=4,                  # Update every 4 steps
        gradient_steps=1,              # One gradient step per update
        target_update_interval=500,    # More frequent target network updates
        exploration_fraction=0.2,      # Faster exploration decay
        exploration_initial_eps=1.0,   # Start with full exploration
        exploration_final_eps=0.05,    # Higher final exploration for better stability
        policy_kwargs=dict(
            net_arch=[256, 256],       # Deeper network
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
