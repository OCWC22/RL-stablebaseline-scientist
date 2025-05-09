# Environment Setup and Test Guide

This document provides step-by-step instructions for setting up the environment and running tests for the RL-stablebaseline-scientist project.

## Environment Setup

### 1. Create and Activate Virtual Environment

```bash
# Create a new virtual environment
python3 -m venv .venv-sb3

# Activate the virtual environment
source .venv-sb3/bin/activate
```

### 2. Install Dependencies

```bash
# Install all required packages with exact versions
python -m pip install -r requirements.txt

# Upgrade pip (optional but recommended)
python -m pip install --upgrade pip
```

## Running Tests

### Run All Tests

```bash
# Run all tests with verbose output
python -m pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Run only environment utility tests
python -m pytest tests/test_env_utils.py -v

# Run only PPO tests
python -m pytest tests/test_ppo_training.py -v

# Run only A2C tests
python -m pytest tests/test_a2c_training.py -v

# Run only DQN tests
python -m pytest tests/test_dqn_training.py -v

# Run only evaluation tests
python -m pytest tests/test_evaluate_agent.py -v
```

## Test Breakdown

### Environment Utility Tests (`test_env_utils.py`)

1. **`test_make_eval_env`**: Verifies that a single evaluation environment can be created correctly.
   - Checks if the environment is properly wrapped with Monitor
   - Validates observation space

2. **`test_make_cartpole_vec_env`**: Tests vectorized environment creation with different configurations.
   - Tests with 1 environment using DummyVecEnv
   - Tests with 4 environments using DummyVecEnv
   - Tests with 4 environments using SubprocVecEnv
   - Validates observation/action spaces and step/reset methods

### PPO Training Tests (`test_ppo_training.py`)

1. **`test_ppo_instantiation`**: Verifies that a PPO model can be instantiated with our environment.
   - Checks if model, policy, and environment are properly initialized

2. **`test_ppo_short_training`**: Tests that PPO can be trained for a small number of steps.
   - Trains for 500 timesteps and verifies the model updates

3. **`test_ppo_save_load`**: Tests that PPO models can be saved and loaded.
   - Saves a trained model to disk
   - Loads the model and verifies it's intact

4. **`test_ppo_prediction`**: Tests that PPO can make predictions after training.
   - Trains a model
   - Gets an observation and makes a prediction
   - Verifies the action has the correct shape and values

5. **`test_ppo_evaluation`**: Tests that PPO can be evaluated.
   - Trains a model
   - Evaluates it on a separate environment
   - Checks that evaluation returns valid results

### A2C Training Tests (`test_a2c_training.py`)

1. **`test_a2c_instantiation`**: Verifies that an A2C model can be instantiated with our environment.

2. **`test_a2c_short_training`**: Tests that A2C can be trained for a small number of steps.

3. **`test_a2c_save_load`**: Tests that A2C models can be saved and loaded.

4. **`test_a2c_prediction`**: Tests that A2C can make predictions after training.

5. **`test_a2c_evaluation`**: Tests that A2C can be evaluated.

### DQN Training Tests (`test_dqn_training.py`)

1. **`test_dqn_instantiation`**: Verifies that a DQN model can be instantiated with our environment.

2. **`test_dqn_short_training`**: Tests that DQN can be trained for a small number of steps.
   - Uses a smaller buffer and earlier learning start for faster testing

3. **`test_dqn_save_load`**: Tests that DQN models can be saved and loaded.

4. **`test_dqn_prediction`**: Tests that DQN can make predictions after training.

5. **`test_dqn_evaluation`**: Tests that DQN can be evaluated.

### Evaluation Tests (`test_evaluate_agent.py`)

1. **`test_evaluate_agent_function`**: Tests that the evaluate_agent function works correctly.
   - Creates a temporarily trained PPO model
   - Evaluates it using the evaluate_agent function
   - Verifies the evaluation returns valid results

2. **`test_evaluate_agent_with_different_algos`**: Tests algorithm type validation.
   - Verifies that models can be loaded with the correct algorithm type
   - Checks model type integrity

## Hyperparameter Tuning

The project includes comprehensive hyperparameter tuning capabilities for all three implemented algorithms (PPO, A2C, and DQN) using Optuna, a powerful hyperparameter optimization framework.

### Setting Up for Tuning

To use the hyperparameter tuning capabilities, you'll need to install the additional dependencies:

```bash
# Activate the virtual environment
source .venv-sb3/bin/activate

# Install tuning dependencies
python -m pip install optuna plotly kaleido
```

### Using the Unified Tuning Interface

The project provides a unified command-line interface for tuning any of the implemented algorithms:

```bash
# Tune PPO with default settings
python -m scripts.tune_rl --algorithm ppo

# Tune A2C with fewer trials for faster results
python -m scripts.tune_rl --algorithm a2c --n-trials 20

# Tune DQN with more timesteps for better convergence
python -m scripts.tune_rl --algorithm dqn --n-timesteps 200000
```

### Algorithm-Specific Tuning

You can also use the algorithm-specific tuning scripts directly:

```bash
# Tune PPO with specific parameters
python -m scripts.tune_ppo --n-trials 50 --output-dir custom_tuning_results

# Tune A2C with database storage for persistence
python -m scripts.tune_a2c --storage sqlite:///tuning.db --study-name a2c_study

# Tune DQN with specific seed for reproducibility
python -m scripts.tune_dqn --seed 12345
```

### Understanding Tuning Results

Tuning results are saved in the `tuning_results` directory (or a custom directory if specified) with the following structure:

- `<algorithm>_cartpole_best_params.json`: The best hyperparameters found during tuning
- `<algorithm>_history.png`: Visualization of the optimization history
- `<algorithm>_importance.png`: Visualization of parameter importance
- Individual trial directories with:
  - Trial-specific results
  - Trained models for successful trials
  - TensorBoard logs for each trial

### Using Tuned Hyperparameters

To use the best hyperparameters found during tuning in your training scripts:

1. Locate the best parameters JSON file in the tuning results directory
2. Load the parameters in your training script:

```python
import json

# Load the best hyperparameters
with open("tuning_results/ppo/ppo_cartpole_best_params.json", "r") as f:
    best_params = json.load(f)

# Create the model with the best hyperparameters
model = PPO(
    policy="MlpPolicy",
    env=env,
    **best_params,
    verbose=1
)
```

### Testing Tuning Scripts

The project includes tests for all tuning scripts to ensure they work correctly:

```bash
# Test all tuning scripts
python -m pytest tests/test_tune_ppo.py tests/test_tune_a2c.py tests/test_tune_dqn.py tests/test_tune_rl.py -v

# Test specific tuning script
python -m pytest tests/test_tune_rl.py -v
```

## Common Test Failures and Solutions

1. **PPO Timestep Mismatch**: PPO processes data in batches, so the actual number of timesteps may be slightly higher than requested (rounded up to complete the last batch).
   - Solution: Use `>=` instead of `==` when checking timesteps.

2. **Algorithm Type Loading**: Stable Baselines3 is more flexible than expected when loading models with different algorithm classes.
   - Solution: Directly verify model types rather than expecting exceptions.

3. **Environment Wrapper Issues**: If tests fail with errors about missing attributes like `get_episode_rewards`, check that environments are properly wrapped with Monitor.
   - Solution: Ensure all environments used for evaluation are wrapped with Monitor.

4. **Vectorized Environment Return Types**: In newer versions of Stable Baselines3, the `step()` method of vectorized environments returns `infos` as a tuple of dictionaries rather than a list.
   - Solution: Use `isinstance(infos, (list, tuple))` for compatibility across versions.
