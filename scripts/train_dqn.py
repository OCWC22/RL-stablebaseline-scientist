#!/usr/bin/env python3
"""
Training script for Deep Q-Network (DQN) algorithm on CartPole-v1 environment.

This script implements a production-ready training pipeline for DQN using Stable Baselines3,
following best practices from the SB3 documentation and research benchmarks.

References:
- DQN paper: https://arxiv.org/abs/1312.5602
- Double DQN paper: https://arxiv.org/abs/1509.06461
- SB3 DQN documentation: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
- Hyperparameters from deep_research.md and RL Baselines3 Zoo
"""

import os
import argparse
import numpy as np
import gymnasium as gym
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
    parser = argparse.ArgumentParser(description="Train a DQN agent on CartPole-v1")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
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
    model_name = f"dqn_cartpole_{timestamp}"
    model_path = os.path.join(args.output_dir, model_name)

    # Create training environment
    # Note: DQN doesn't use vectorized environments by default
    env = gym.make("CartPole-v1")
    env = Monitor(env)  # Wrap with Monitor for recording
    env.reset(seed=args.seed)

    # Create separate environment for evaluation
    eval_env = make_eval_env(seed=args.seed + 1000)  # Different seed for eval

    # Set up callbacks
    # Save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.output_dir,
        name_prefix=model_name,
        save_replay_buffer=True,  # DQN uses a replay buffer
        save_vecnormalize=False,  # Not using VecNormalize with DQN
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

    # Define DQN hyperparameters
    # These values are based on the tuned hyperparameters from deep_research.md
    # and RL Baselines3 Zoo for CartPole-v1
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=2.3e-3,       # Tuned learning rate from deep_research.md
        buffer_size=100000,          # Replay buffer size
        learning_starts=1000,        # How many steps before starting to learn
        batch_size=64,               # Minibatch size
        tau=1.0,                     # Soft update coefficient (1.0 = hard update)
        gamma=0.99,                  # Discount factor
        train_freq=4,                # Update the model every 4 steps
        gradient_steps=1,            # How many gradient steps to do after each rollout
        target_update_interval=10,   # Update the target network every 10 gradient steps
        exploration_fraction=0.1,    # Fraction of total timesteps for exploration schedule
        exploration_initial_eps=1.0,  # Initial exploration rate
        exploration_final_eps=0.05,   # Final exploration rate
        max_grad_norm=10,            # Max gradient norm for gradient clipping
        tensorboard_log=args.tensorboard_log,
        policy_kwargs=dict(
            net_arch=[64, 64],  # Network architecture
            activation_fn=torch.nn.ReLU,  # ReLU activation is standard for DQN
        ),
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
    # Add torch import here to avoid issues if not used in the script
    import torch
    main()
