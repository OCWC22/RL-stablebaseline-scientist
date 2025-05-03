#!/usr/bin/env python3
"""
Evaluation script for RL agents trained on CartPole-v1 environment.

This script implements a production-ready evaluation pipeline for RL agents trained
with Stable Baselines3, following best practices from the SB3 documentation and research benchmarks.

References:
- SB3 Evaluation documentation: https://stable-baselines3.readthedocs.io/en/master/guide/evaluation.html
- Evaluation best practices from deep_research.md
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from pathlib import Path
from typing import Optional, Union

# Stable Baselines3 imports
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Import our environment utilities
from src.env_utils import make_eval_env


# Dictionary mapping algorithm names to their classes
ALGO_MAP = {
    "a2c": A2C,
    "ppo": PPO,
    "dqn": DQN,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent on CartPole-v1")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=list(ALGO_MAP.keys()),
        help="RL algorithm used to train the agent",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model file (.zip)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions for evaluation",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Verbosity level (0, 1, or 2)"
    )
    return parser.parse_args()


def evaluate_agent(
    model_path: str,
    algo: str,
    seed: int = 42,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    verbose: int = 1,
) -> tuple[float, float]:
    """Evaluate a trained agent on CartPole-v1.

    Args:
        model_path: Path to the saved model file (.zip)
        algo: Algorithm name ("a2c", "ppo", or "dqn")
        seed: Random seed for reproducibility
        n_eval_episodes: Number of episodes for evaluation
        deterministic: Whether to use deterministic actions
        render: Whether to render the environment
        verbose: Verbosity level

    Returns:
        Tuple of (mean_reward, std_reward)
    """
    # Create evaluation environment
    env = make_eval_env(seed=seed)

    # Load the trained model
    model_class = ALGO_MAP[algo.lower()]
    model = model_class.load(model_path, env=env)

    if verbose > 0:
        print(f"Evaluating {algo.upper()} model from {model_path}")
        print(f"Running {n_eval_episodes} evaluation episodes...")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
        return_episode_rewards=False,
        warn=verbose > 0,
    )

    if verbose > 0:
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Success rate: {(mean_reward >= 475):.2%}")

    env.close()
    return mean_reward, std_reward


def main():
    """Main evaluation function."""
    args = parse_args()

    # Evaluate the agent
    mean_reward, std_reward = evaluate_agent(
        model_path=args.model_path,
        algo=args.algo,
        seed=args.seed,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=args.deterministic,
        render=args.render,
        verbose=args.verbose,
    )

    # Print results
    print(f"\nEvaluation Results:")
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.n_eval_episodes}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # CartPole-v1 is considered solved when the average reward is at least 475 over 100 consecutive episodes
    if mean_reward >= 475:
        print(f"\nSuccess! The agent has solved the CartPole-v1 environment.")
    else:
        print(f"\nThe agent has not yet solved the CartPole-v1 environment.")
        print(f"CartPole-v1 is considered solved when the average reward is at least 475 over 100 consecutive episodes.")


if __name__ == "__main__":
    main()
