#!/usr/bin/env python3

import argparse
import os
import time
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch

# Import the dummy components
from src.components.networks import DummyPolicyValueNetwork
from src.components.world_model import DummyWorldModel
from src.components.curiosity import DummyCuriosityModule
from src.components.buffer import DummyRolloutBuffer

# Import environment utilities
from src.env_utils import make_cartpole_vec_env, make_eval_env


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=10000, 
                        help="Total timesteps for training")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0: no output, 1: INFO)")
    parser.add_argument("--eval-episodes", type=int, default=10, 
                        help="Number of episodes for evaluation")
    
    # Algorithm-specific parameters from pseudocode
    parser.add_argument("--n-real-steps", type=int, default=2048, 
                        help="Environment steps per outer loop")
    parser.add_argument("--planning-horizon", type=int, default=5, 
                        help="Steps per imagined rollout")
    parser.add_argument("--min-planning-rollouts", type=int, default=10, 
                        help="Minimum number of planning rollouts")
    parser.add_argument("--max-planning-rollouts", type=int, default=100, 
                        help="Maximum number of planning rollouts")
    parser.add_argument("--confidence-thresh", type=float, default=0.05, 
                        help="Confidence threshold for adaptive imagination")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of PPO epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch size")
    parser.add_argument("--beta-init", type=float, default=0.2, 
                        help="Initial curiosity weight")
    parser.add_argument("--beta-adapt-rate", type=float, default=0.01, 
                        help="Adaptation rate for curiosity weight")
    
    return vars(parser.parse_args())


def interpolate(value, low, high, out_low, out_high):
    """Linear interpolation helper function.
    
    Maps 'value' from range [low, high] to range [out_low, out_high].
    """
    # Ensure value is within bounds
    value = max(low, min(high, value))
    # Calculate normalized position in the original range
    normalized = (value - low) / (high - low) if high > low else 0
    # Map to the output range
    return out_low + normalized * (out_high - out_low)


def main() -> None:
    """Implement the Model-Based Planning with Adaptive Imagination algorithm."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    
    # Create environment
    env = make_cartpole_vec_env(n_envs=1, seed=args["seed"])
    eval_env = make_eval_env(env_id=args["env"], seed=args["seed"] + 100)
    
    # Get environment spaces
    observation_space = env.observation_space
    action_space = env.action_space
    
    print("\n=== 1. INITIALIZATION ===\n")
    
    # Initialize policy network πθ (actor) and value network Vθ (critic)
    policy_value_net = DummyPolicyValueNetwork(observation_space, action_space)
    
    # Initialize world-model ŵφ (predicts next state, reward, done)
    world_model = DummyWorldModel(observation_space, action_space)
    
    # Initialize curiosity module Cψ (e.g. RND)
    curiosity = DummyCuriosityModule(observation_space, action_space, beta_init=args["beta_init"])
    
    # Initialize RolloutBuffer B (stores both real and imagined transitions)
    buffer = DummyRolloutBuffer(buffer_size=args["n_real_steps"] * 2)  # Extra space for imagined data
    
    # Initialize optimizers (in a real implementation)
    # optimiser_actor_critic = ...
    # optimiser_world_model = ...
    # optimiser_curiosity = ...
    
    # Track statistics
    total_steps = 0
    start_time = time.time()
    
    # Main loop
    print("\n=== 2. MAIN LOOP ===\n")
    
    while total_steps < args["total_timesteps"]:
        print(f"\n--- Iteration at step {total_steps}/{args['total_timesteps']} ---\n")
        
        # 2.1 COLLECT REAL EXPERIENCE (on-policy)
        print("\n=== 2.1 COLLECTING REAL EXPERIENCE ===\n")
        buffer.reset()
        obs = env.reset()[0]  # Reset returns (obs, info) tuple in Gym 0.26+
        
        for step in range(args["n_real_steps"]):
            # Get action, value, logp from policy
            action, value, logp = policy_value_net(obs)
            
            # Step the environment
            next_obs, reward, done, info = env.step([action])
            next_obs = next_obs[0]  # Unwrap from vectorized env
            ext_reward = reward[0]
            done = done[0]
            
            # Calculate intrinsic reward (optional)
            intr_reward = curiosity.beta * curiosity.intrinsic_reward(obs, action, next_obs)
            
            # Add to buffer
            buffer.add(
                obs=obs,
                action=action,
                reward=ext_reward + intr_reward,
                value=value,
                logp=logp,
                ext_reward=ext_reward,
                intr_reward=intr_reward,
                done=done,
                is_real=True
            )
            
            # Update observation
            if done:
                obs = env.reset()[0]
            else:
                obs = next_obs
            
            total_steps += 1
            if total_steps % 500 == 0:
                print(f"Collected {total_steps} real steps so far")
        
        # 2.2 UPDATE WORLD MODEL & COMPUTE CONFIDENCE
        print("\n=== 2.2 UPDATING WORLD MODEL ===\n")
        # In a real implementation, we would sample mini-batches from the real part of the buffer
        # For the dummy version, just call update with all real data
        loss_model = world_model.update(
            states=[],  # These would be filled with real data
            actions=[],
            next_states=[],
            rewards=[],
            dones=[]
        )
        
        # Calculate confidence as per pseudocode
        confidence = world_model.get_confidence()
        
        # 2.3 ADAPTIVE IMAGINATION (PLANNING)
        print("\n=== 2.3 ADAPTIVE IMAGINATION ===\n")
        # Determine number of rollouts based on confidence
        num_rollouts = int(interpolate(
            confidence,
            low=args["confidence_thresh"],
            high=1.0,
            out_low=args["min_planning_rollouts"],
            out_high=args["max_planning_rollouts"]
        ))
        
        print(f"Performing {num_rollouts} imagined rollouts")
        
        for rollout in range(num_rollouts):
            # Sample a recent state from the buffer
            sim_state = buffer.sample_recent_state()
            
            for h in range(args["planning_horizon"]):
                # Get action from policy
                a_sim, v_sim, logp_sim = policy_value_net(sim_state)
                
                # Predict next state, reward, done using world model
                s_next, r_hat, d_hat = world_model.predict(sim_state, a_sim)
                
                # Calculate intrinsic reward
                intr_sim = curiosity.beta * curiosity.intrinsic_reward(sim_state, a_sim, s_next)
                
                # Add to buffer
                buffer.add(
                    obs=sim_state,
                    action=a_sim,
                    reward=r_hat + intr_sim,
                    value=v_sim,
                    logp=logp_sim,
                    ext_reward=r_hat,
                    intr_reward=intr_sim,
                    done=d_hat,
                    is_real=False
                )
                
                # Update simulation state
                sim_state = s_next
                
                # Break if done
                if d_hat:
                    break
        
        # 2.4 ADVANTAGE & RETURN COMPUTATION
        print("\n=== 2.4 COMPUTING ADVANTAGES & RETURNS ===\n")
        # Get value of last observation
        last_value = policy_value_net(obs)[1]  # [1] is the value
        
        # Compute returns and advantages
        buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=args["gamma"],
            gae_lambda=args["gae_lambda"]
        )
        
        # 2.5 PPO-STYLE POLICY / VALUE UPDATE
        print("\n=== 2.5 UPDATING POLICY & VALUE FUNCTION ===\n")
        for epoch in range(args["n_epochs"]):
            print(f"PPO update epoch {epoch+1}/{args['n_epochs']}")
            
            for batch in buffer.iterate(batch_size=args["batch_size"]):
                # In a real implementation, this would update the policy and value function
                # For the dummy version, just print a message
                print("  Processing batch for PPO update")
                
                # This would compute the policy loss, value loss, entropy loss, etc.
                # and update the networks using the optimizers
                policy_loss = 0.1 * np.random.random()
                value_loss = 0.2 * np.random.random()
                entropy_loss = 0.05 * np.random.random()
                
                print(f"  Losses - Policy: {policy_loss:.4f}, Value: {value_loss:.4f}, "
                      f"Entropy: {entropy_loss:.4f}")
        
        # 2.6 CURIOSITY MODULE UPDATE
        print("\n=== 2.6 UPDATING CURIOSITY MODULE ===\n")
        if curiosity.beta > 0:
            curiosity.update(buffer.real_and_imagined())
        
        # 2.7 ADAPT β (CURIOSITY WEIGHT)
        print("\n=== 2.7 ADAPTING CURIOSITY WEIGHT ===\n")
        # Calculate moving average and trend of external rewards
        # In a real implementation, this would compute the actual trend
        # For the dummy version, just use a random trend
        avg_ext = np.mean(buffer.real_ext_rewards()) if buffer.real_ext_rewards() else 0
        trend = np.random.uniform(-0.02, 0.02)  # Random trend between -0.02 and 0.02
        
        # Adapt beta based on trend
        curiosity.adapt_beta(trend)
        
        # 2.8 LOGGING
        print("\n=== 2.8 LOGGING ===\n")
        print({
            "confidence": confidence,
            "imagined_rollouts": num_rollouts,
            "beta": curiosity.beta,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "model_loss": loss_model,
            "mean_real_reward": avg_ext
        })
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    
    # Evaluate the final policy
    print("\nEvaluating the final policy...")
    total_reward = 0
    episodes = 0
    
    obs, _ = eval_env.reset()
    done = False
    
    while episodes < args["eval_episodes"]:
        action, _, _ = policy_value_net(obs)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        
        if done:
            obs, _ = eval_env.reset()
            episodes += 1
    
    mean_reward = total_reward / args["eval_episodes"]
    print(f"Mean evaluation reward: {mean_reward:.2f} over {args['eval_episodes']} episodes")
    
    # Close environments
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
