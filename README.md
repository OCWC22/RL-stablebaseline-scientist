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
├── models/             # Saved model checkpoints (.zip)
├── logs/               # TensorBoard logs
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