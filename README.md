# AI Co-Scientist RL Benchmarking (Stable Baselines3)

This project implements and benchmarks standard Reinforcement Learning algorithms (PPO, A2C, DQN) on the CartPole-v1 environment using the Stable Baselines3 library. The goal is to establish reliable and well-tested implementations as a foundation. Additionally, the project includes a skeleton implementation of a Model-Based PPO algorithm with adaptive imagination, which will be developed into a full implementation.

## Algorithm Comparison

### Performance Visualization

```
Reward
500 |                                   ********** PPO & A2C (Final Performance)
    |                                  /
400 |                                 /
    |                                /
300 |                               /
    |                              /
200 |                             /
    |          A2C (Local)        /
100 |          Initial *          /
    |                  \         /
 50 |                   \       /                  ********* DQN (Local Final)
    |                    \     /
 25 |                     \___/ <-- Colab Initial Performance (~17-24)                              
 20 | ***************************************************** MB-PPO Skeleton
    |                                                        (Remains at ~20)
  0 +------------------------------------------------------------
      Start                    Training Steps                  End
```

### Runtime and Performance Metrics

| Algorithm | Implementation | Runtime | Initial Performance | Final Performance | Improvement Factor |
|-----------|----------------|---------|---------------------|-------------------|--------------------|  
| PPO | Optimized (Local) | ~17 sec | 9.10 reward | 500.00 reward | 55x |
| PPO | Unoptimized (Local) | ~25 sec | 8.40 reward | 450.00 reward | 54x |
| PPO | Optimized (Colab) | ~8 sec | 24.10 reward | 500.00 reward | 21x |
| A2C | Optimized (Local) | ~20 sec | 126.60 reward | 500.00 reward | 4x |
| A2C | Unoptimized (Local) | ~30 sec | 15.20 reward | 425.00 reward | 28x |
| A2C | Optimized (Colab) | ~10 sec | 17.60 reward | Testing | Increasing |
| DQN | Optimized (Local) | ~16 sec | 9.50 reward | 40.50 reward | 4.3x |
| DQN | Unoptimized (Local) | ~22 sec | 9.20 reward | 20.30 reward | 2.2x |
| DQN | Optimized (Colab) | ~9 sec | 16.40 reward | Testing | Increasing |
| MB-PPO | Dummy (Local) | ~30 sec | ~20.00 reward | ~20.00 reward | 1x (no change) |

### Optimization Impact

Optimization had significant effects on algorithm performance:

- **PPO**: Optimization improved final reward by ~11% and reduced runtime by ~32%
- **A2C**: Optimization improved final reward by ~18% and reduced runtime by ~33%
- **DQN**: Optimization had the biggest impact, doubling the final reward and reducing runtime by ~27%

These improvements highlight the importance of hyperparameter tuning and implementation optimization in reinforcement learning.

## Key Takeaways

1. **Standard RL Algorithms Work Effectively** - PPO and A2C both achieve perfect performance on CartPole-v1
2. **Framework Validation** - The robust performance of optimized algorithms confirms our testing setup
3. **MB-PPO Skeleton Verification** - Our skeleton implementation maintains the expected random-policy baseline performance
4. **Cross-Environment Consistency** - Both local and Colab implementations show the expected learning patterns

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
│   ├── test_env_utils.py
│   ├── test_ppo_training.py
│   ├── test_a2c_training.py
│   └── test_dqn_training.py
├── scripts/            # Training and evaluation scripts
│   ├── train_ppo.py
│   ├── train_a2c.py
│   ├── train_dqn.py
│   ├── train_mb_ppo.py # Model-Based PPO (skeleton implementation)
│   ├── evaluate_agent.py
│   └── tune_rl.py      # Hyperparameter tuning script
├── models/             # Saved model files
├── logs/               # Training logs and TensorBoard files
├── notebook_a2c_cells.py   # A2C implementation for Colab notebook
├── notebook_dqn_cells.py   # DQN implementation for Colab notebook
├── ppo_test.py             # PPO test script
├── a2c_test.py             # A2C test script
├── dqn_test.py             # DQN test script
├── mb_ppo_test.py          # Model-Based PPO test script
├── streamlit_app.py       # Interactive dashboard for algorithm comparison
├── algorithm_comparison.md # Detailed algorithm comparison document
├── model_based_rl_explained.md # Explanation of Model-Based RL approach
├── notebook_modification_instructions.md # Instructions for notebook modifications
├── project_presentation.md # Comprehensive project presentation document
├── coding_updates_1.md     # Log of code changes and updates
├── prd.md                  # Project plan and requirements
├── deep_research.md        # Research on SB3 implementation details
├── deep_research_2.md      # Additional research on Model-Based RL
└── starting.md             # Initial project setup and TDD guidelines
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
python scripts/train_mb_ppo.py --total-timesteps 5000
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

## Running the Interactive Dashboard

This project includes an interactive Streamlit dashboard for visualizing and comparing algorithm performance metrics:

```bash
# Install Streamlit if not already installed
pip install streamlit

# Run the dashboard
streamlit run streamlit_app.py
```

The dashboard provides:
- Performance comparison across algorithms and environments
- Optimization impact analysis
- Learning visualization
- Model-Based RL architecture explanation

The dashboard directly integrates with the project's markdown documentation files to ensure consistency between visualizations and documentation.

## Running the Algorithms

### Local Environment Setup

```bash
# Install dependencies
pip install gymnasium stable-baselines3 numpy torch matplotlib

# Set Python path to include the project root
export PYTHONPATH=$PYTHONPATH:/path/to/RL-stablebaseline-scientist
```

### Running Standard Algorithms

```bash
# Run PPO
python ppo_test.py

# Run A2C
python a2c_test.py

# Run DQN
python dqn_test.py

# Run Model-Based PPO skeleton
python mb_ppo_test.py
```

### Notebook Implementation

To modify the Colab notebook (`Copy_of_1_getting_started.ipynb`) for different algorithms:

1. Follow instructions in `notebook_modification_instructions.md`
2. Use code cells from `notebook_a2c_cells.py` or `notebook_dqn_cells.py`

## Documentation

- **algorithm_comparison.md**: Detailed comparison of algorithm performance
- **model_based_rl_explained.md**: Comprehensive explanation of Model-Based RL approach
- **project_presentation.md**: Complete project overview for presentation purposes

## Future Work

1. Implement learning neural networks for MB-PPO components
2. Evaluate sample efficiency compared to model-free approaches
3. Add advanced exploration mechanisms via the curiosity module
4. Test on more complex environments beyond CartPole-v1

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

## Acknowledgments

This project uses [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for the standard RL algorithm implementations.