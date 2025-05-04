# AI Co-Scientist RL Benchmarking (Stable Baselines3)

This project implements and benchmarks standard Reinforcement Learning algorithms (PPO, A2C, DQN) on the CartPole-v1 environment using the Stable Baselines3 library. The goal is to establish reliable and well-tested implementations as a foundation.

Based on the plan outlined in `prd.md`.

## Project Structure

```
/
├── src/                # Core logic (e.g., environment utilities)
│   ├── __init__.py
│   └── env_utils.py
├── tests/              # Pytest unit and integration tests
│   ├── __init__.py
│   └── test_env_utils.py
│   └── ... (test_ppo_training.py, etc.)
├── scripts/            # Training and evaluation scripts
│   ├── __init__.py
│   └── train_ppo.py
│   └── train_a2c.py
│   └── train_dqn.py
│   └── evaluate_agent.py
│   └── tune_rl.py
│   └── tune_ppo.py
│   └── tune_a2c.py
│   └── tune_dqn.py
├── models/             # Saved model checkpoints (.zip)
├── logs/               # TensorBoard logs
├── tuning_results/     # Hyperparameter tuning results
├── requirements.txt    # Project dependencies
├── prd.md              # Project plan and details
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