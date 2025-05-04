# Reinforcement Learning Results Summary (CartPole-v1)

This document summarizes the training and evaluation results for PPO, A2C, and DQN algorithms implemented using Stable Baselines3 on the CartPole-v1 environment.

## Algorithm Performance

| Algorithm | Training Script | Evaluation Mean Reward | Notes |
|---|---|---|---|
| PPO (Baseline) | `scripts/train_ppo.py` | ~200 (Baseline) | Initial implementation, not optimized. |
| PPO (Optimized) | `scripts/train_optimized_ppo.py` | 500.00 ± 0.00 | Achieved perfect score after hyperparameter tuning. |
| A2C (Optimized) | `scripts/train_optimized_a2c.py` | 500.00 ± 0.00 | Achieved perfect score after hyperparameter tuning. |
| DQN (Optimized) | `scripts/train_optimized_dqn.py` | 424.65 ± 97.26 | Significant improvement over baseline, but slightly less stable than A2C. |

*Note: Baseline PPO results are typical initial runs without tuning. Optimized results reflect runs after hyperparameter adjustments.* 

## Training Scripts

- **PPO (Baseline):** [scripts/train_ppo.py](scripts/train_ppo.py) (Baseline implementation)
- **A2C (Baseline):** [scripts/train_a2c.py](scripts/train_a2c.py) (Baseline implementation)
- **DQN (Baseline):** [scripts/train_dqn.py](scripts/train_dqn.py) (Baseline implementation)
- **PPO (Optimized):** [scripts/train_optimized_ppo.py](scripts/train_optimized_ppo.py)
- **A2C (Optimized):** [scripts/train_optimized_a2c.py](scripts/train_optimized_a2c.py)
- **DQN (Optimized):** [scripts/train_optimized_dqn.py](scripts/train_optimized_dqn.py)

## Scaling with GPU

Stable Baselines3 automatically utilizes a GPU if a compatible one (CUDA-enabled Nvidia GPU) and the necessary drivers/toolkits (CUDA, cuDNN) are detected along with the correct PyTorch version (`torch` with CUDA support).

**To leverage GPU acceleration:**

1.  **Hardware:** Ensure you have a CUDA-compatible Nvidia GPU.
2.  **Drivers & Toolkit:** Install the appropriate Nvidia drivers and CUDA toolkit version compatible with your desired PyTorch version.
3.  **PyTorch Installation:** Install PyTorch with CUDA support. You can find the correct command on the [PyTorch website](https://pytorch.org/get-started/locally/). Example: `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (Replace `cu118` with your CUDA version if different).
4.  **SB3:** No specific changes are needed in the SB3 code. It will automatically detect and use the GPU if PyTorch is configured correctly (`model.device` will show `cuda:0`).

Using a GPU can significantly speed up training, especially for larger networks and more complex environments, as it accelerates the parallel computations involved in neural network forward and backward passes.

## Future Work

- Implement custom callbacks for detailed logging/monitoring.
- Extend experiments to more complex Gymnasium environments.
- Conduct more extensive hyperparameter searches (e.g., using Optuna) for all algorithms.
- Investigate advanced techniques like model-based RL or hierarchical RL.
