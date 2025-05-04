#!/usr/bin/env python3

import argparse
import os
import time
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--tensorboard-log", type=str, default="./tensorboard_logs/ppo_optimized/", 
                        help="Tensorboard log directory")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0: no output, 1: INFO)")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of episodes for evaluation")
    parser.add_argument("--save-path", type=str, default="./models/ppo_optimized_cartpole.zip", 
                        help="Path to save the model")
    return vars(parser.parse_args())


def main() -> None:
    """Train and evaluate an optimized PPO agent on CartPole-v1."""
    # Parse arguments
    args = parse_args()
    
    # Create log and model directories if they don't exist
    os.makedirs(os.path.dirname(args["tensorboard_log"]), exist_ok=True)
    os.makedirs(os.path.dirname(args["save_path"]), exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    
    # Create vectorized environment
    def make_env():
        env = gym.make(args["env"])
        env.reset(seed=args["seed"])
        return env
    
    env = DummyVecEnv([make_env for _ in range(args["n_envs"])])
    
    # Create optimized PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,            # Slightly increased learning rate
        n_steps=128,                   # Increased steps per update for better exploration
        batch_size=256,                # Larger batch size for more stable updates
        n_epochs=10,                   # More optimization epochs per update
        gamma=0.99,                    # Standard discount factor
        gae_lambda=0.95,               # Standard GAE lambda
        clip_range=0.2,                # Standard clip range
        clip_range_vf=None,            # No value function clipping
        normalize_advantage=True,      # Normalize advantages
        ent_coef=0.01,                 # Increased entropy coefficient for better exploration
        vf_coef=0.5,                   # Standard value function coefficient
        max_grad_norm=0.5,             # Standard gradient clipping
        use_sde=False,                 # No stochastic dynamics extraction
        sde_sample_freq=-1,            # Not using SDE
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])],  # Deeper network
            activation_fn=torch.nn.ReLU,  # ReLU activation
        ),
        tensorboard_log=args["tensorboard_log"],
        verbose=args["verbose"],
        seed=args["seed"],
    )
    
    # Train the agent
    start_time = time.time()
    model.learn(total_timesteps=args["total_timesteps"])
    total_time = time.time() - start_time
    
    # Save the trained model
    model.save(args["save_path"])
    print(f"Model saved to {args['save_path']}")
    
    # Evaluate the trained agent
    print("\nEvaluating the trained agent...")
    eval_env = gym.make(args["env"])
    eval_env.reset(seed=args["seed"] + 100)  # Different seed for evaluation
    
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args["eval_episodes"], deterministic=True
    )
    
    print(f"\nTraining time: {total_time:.2f} seconds")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Close environments
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
