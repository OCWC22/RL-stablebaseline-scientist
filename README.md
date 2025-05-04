# AI Co-Scientist RL Benchmarking (Stable Baselines3)

This project implements and benchmarks standard Reinforcement Learning algorithms (PPO, A2C, DQN) on the CartPole-v1 environment using the Stable Baselines3 library. The goal is to establish reliable and well-tested implementations as a foundation. Additionally, the project now includes a skeleton implementation of a Model-Based PPO algorithm with adaptive imagination.

Based on the plan outlined in `prd.md`.

## Project Structure

```
/
├── src/                # Core logic (e.g., environment utilities)
│   ├── __init__.py
│   ├── env_utils.py
│   └── components/     # Components for Model-Based PPO
│       ├── buffer.py   # Rollout buffer for real and imagined transitions
│       ├── curiosity.py # Curiosity module for intrinsic rewards
│       ├── networks.py # Policy and value networks
│       └── world_model.py # World model for predicting next states
├── tests/              # Pytest unit and integration tests
│   ├── __init__.py
│   └── test_env_utils.py
│   └── ... (test_ppo_training.py, etc.)
├── scripts/            # Training and evaluation scripts
│   ├── __init__.py
│   ├── train_ppo.py
│   ├── train_a2c.py
│   ├── train_dqn.py
│   ├── train_mbppo_skeleton.py # Model-Based PPO skeleton implementation
│   ├── evaluate_agent.py
│   ├── tune_rl.py
│   ├── tune_ppo.py
│   ├── tune_a2c.py
│   └── tune_dqn.py
├── models/             # Saved model checkpoints (.zip)
├── logs/               # TensorBoard logs
├── tuning_results/     # Hyperparameter tuning results
├── requirements.txt    # Project dependencies
├── prd.md              # Project plan and details
├── pseudocode.md       # Pseudocode for Model-Based PPO
├── coding_updates_1.md # Log of code changes
├── setup_and_test_guide.md # Detailed setup and testing instructions
└── README.md           # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd RL-stablebaseline-scientist
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv-sb3
    source .venv-sb3/bin/activate  # On Windows use `.venv-sb3\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    python -m pip install -r requirements.txt
    ```

    For detailed setup instructions, see [setup_and_test_guide.md](setup_and_test_guide.md).

## Usage

### Training Agents

Use the scripts in the `scripts/` directory to train agents. Models will be saved to `models/` and logs to `logs/`.

*   **Train PPO:**
    ```bash
    python scripts/train_ppo.py --total-timesteps 100000
    ```
*   **Train A2C:**
    ```bash
    python scripts/train_a2c.py --total-timesteps 100000
    ```
*   **Train DQN:**
    ```bash
    python scripts/train_dqn.py --total-timesteps 100000
    ```

    *Note: You can adjust `--total-timesteps` and other parameters defined in the scripts (e.g., `--seed`, `--n-envs`).*

### Evaluating Agents

Use the `evaluate_agent.py` script to evaluate a saved model.

```bash
python scripts/evaluate_agent.py --algo <algo_name> --model-path <path_to_model.zip>
```

*   **Example (evaluating a saved PPO model):**
    ```bash
    python scripts/evaluate_agent.py --algo ppo --model-path models/best_ppo_cartpole.zip --n-eval-episodes 50
    ```

### Monitoring Training (TensorBoard)

Launch TensorBoard to view training progress:

```bash
tensorboard --logdir logs/
```
Navigate to `http://localhost:6006/` in your browser.

### Model-Based PPO with Adaptive Imagination

The project includes a skeleton implementation of a Model-Based PPO algorithm with adaptive imagination, based on the pseudocode in `pseudocode.md`. This implementation demonstrates the structure and flow of the algorithm without implementing the actual neural networks.

```bash
# Run the Model-Based PPO skeleton implementation
python scripts/train_mbppo_skeleton.py --total-timesteps 5000
```

#### What is Model-Based PPO?

Unlike standard PPO (which is model-free), Model-Based PPO combines:

1. **A world model** that learns to predict the next state, reward, and done flag based on the current state and action
2. **A policy/value network** (similar to standard PPO) that decides which actions to take
3. **Adaptive imagination** that generates synthetic experience using the world model

This approach can be more sample-efficient than standard model-free methods because it leverages the world model to generate additional training data without requiring real environment interactions.

The key innovation in this implementation is the adaptive nature of the imagination process - the number of imagined rollouts scales with the world model's confidence. As the model becomes more accurate, it's trusted for more planning steps.

See `model_based_rl_explained.md` for a comprehensive explanation of model-based reinforcement learning and this specific algorithm.

The skeleton implementation includes:
- Policy/value networks (dummy implementation)
- World model for predicting next states (dummy implementation)
- Curiosity module for intrinsic rewards (dummy implementation)
- Rollout buffer for storing both real and imagined transitions
- Adaptive imagination based on world model confidence

This serves as a foundation for implementing the full algorithm with actual neural networks and optimization logic.

## Hyperparameter Tuning

The project includes automated hyperparameter tuning capabilities using Optuna for all three algorithms (PPO, A2C, and DQN). This allows you to find optimal hyperparameters for your specific environment and requirements.

### Unified Tuning Interface

Use the unified tuning interface to optimize any of the implemented algorithms:

```bash
# Tune PPO with 50 trials
python -m scripts.tune_rl --algorithm ppo --n-trials 50

# Tune A2C with custom study name and storage
python -m scripts.tune_rl --algorithm a2c --study-name my_a2c_study --storage sqlite:///tuning.db

# Tune DQN with more timesteps
python -m scripts.tune_rl --algorithm dqn --n-timesteps 200000
```

### Algorithm-Specific Tuning

You can also use the algorithm-specific tuning scripts directly:

```bash
# Tune PPO
python -m scripts.tune_ppo --n-trials 50

# Tune A2C
python -m scripts.tune_a2c --n-trials 50

# Tune DQN
python -m scripts.tune_dqn --n-trials 50
```

### Tuning Results

Results are saved in the `tuning_results` directory (or a custom directory if specified) with the following structure:

- Best hyperparameters as JSON files
- Visualization of optimization history and parameter importance
- Individual trial results and trained models

### Using Tuned Hyperparameters

To use the best hyperparameters found during tuning in your training scripts:

```python
import json

# Load the best hyperparameters
with open("tuning_results/ppo/ppo_cartpole_best_params.json", "r") as f:
    best_params = json.load(f)

# Create the model with the best hyperparameters
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, **best_params, verbose=1)
model.learn(total_timesteps=100000)
```

## CartPole Benchmark

This project includes a comprehensive benchmark of PPO, A2C, and DQN algorithms on the CartPole-v1 environment. The CartPole environment is considered solved when the agent achieves an average reward of 475 or more over 100 consecutive episodes.

### Running the CartPole Benchmark

To benchmark all three algorithms on CartPole-v1:

```bash
# Activate the virtual environment
source .venv-sb3/bin/activate

# Set the Python path to include the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Train PPO on CartPole (best performance)
python scripts/train_ppo.py --total-timesteps 100000 --n-envs 4 --save-freq 10000 --eval-freq 10000

# Train A2C on CartPole (good performance)
python scripts/train_a2c.py --total-timesteps 100000 --n-envs 4 --save-freq 10000 --eval-freq 10000

# Train DQN on CartPole (requires tuning)
python scripts/train_dqn.py --total-timesteps 100000 --save-freq 10000 --eval-freq 10000
```

### Evaluating Benchmark Results

After training, evaluate each algorithm's performance:

```bash
# Evaluate PPO
python scripts/evaluate_agent.py --algo ppo --model-path models/ppo_cartpole_*_final.zip --n-eval-episodes 20

# Evaluate A2C
python scripts/evaluate_agent.py --algo a2c --model-path models/a2c_cartpole_*_final.zip --n-eval-episodes 20

# Evaluate DQN
python scripts/evaluate_agent.py --algo dqn --model-path models/dqn_cartpole_*_final.zip --n-eval-episodes 20
```

### Benchmark Results

Our benchmark testing shows:

| Algorithm | Average Reward | Standard Deviation | Solved? |
|-----------|----------------|-------------------|---------|
| PPO       | 500.00         | 0.00              | Yes  |
| A2C       | ~435.00        | ~64.00            | No   |
| DQN       | ~10.00         | ~1.00             | No   |

**Notes:**
- PPO consistently solves the environment with default parameters
- A2C comes close to solving the environment and may solve it with more training
- DQN requires hyperparameter tuning to solve the environment effectively

### Improving Performance

For algorithms that don't solve the environment with default parameters, use the hyperparameter tuning scripts:

```bash
# Tune DQN hyperparameters
python scripts/tune_dqn.py --n-trials 50 --n-timesteps 200000

# Train DQN with tuned hyperparameters
# (Load the best parameters from the JSON file produced by tuning)
```

## Testing

Run the test suite using pytest:

```bash
python -m pytest tests/
```

For verbose output:

```bash
python -m pytest tests/ -v
```

To run specific test categories:

```bash
# Environment utilities
python -m pytest tests/test_env_utils.py -v

# Algorithm-specific tests
python -m pytest tests/test_ppo_training.py -v
python -m pytest tests/test_a2c_training.py -v
python -m pytest tests/test_dqn_training.py -v

# Evaluation tests
python -m pytest tests/test_evaluate_agent.py -v
```

For a detailed breakdown of all tests and common troubleshooting, see [setup_and_test_guide.md](setup_and_test_guide.md).