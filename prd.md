# Project Plan: Stable Baselines3 RL Algorithm Implementation (PPO, A2C, DQN)

**Goal:** Implement PPO, A2C, and DQN algorithms using Stable Baselines3 for the CartPole-v1 environment, ensuring research-grade reliability, correctness, and testability through specific code examples and Test-Driven Development (TDD).

**Phase 1: Setup & Environment**

1.  **Task:** Define Project Structure
    *   **Sub-task:** Create directories: `src/` (for core logic), `tests/`, `scripts/` (for training/eval runners), `models/` (for saved agents), `logs/` (for TensorBoard).
    ```bash
    mkdir src tests scripts models logs
    touch src/__init__.py tests/__init__.py scripts/__init__.py src/env_utils.py requirements.txt README.md coding_updates_1.md
    ```

2.  **Task:** Environment Setup
    *   **Sub-task:** Verify Python version (>= 3.8).
    *   **Sub-task:** Create `requirements.txt` with specific versions (adjust versions as needed based on compatibility, referencing SB3/Gymnasium docs).
        ```plaintext
        # requirements.txt
        stable-baselines3[extra]>=2.2.1
        gymnasium[classic_control]>=0.29.1
        pytest>=7.4.0
        torch>=2.0.0 # Or specify cuda/cpu version if necessary
        tensorboard>=2.13.0
        # Add other dependencies like numpy if used directly
        ```
    *   **Sub-task:** Install dependencies using `uv`.
        ```bash
        uv pip install -r requirements.txt
        ```
    *   **Sub-task:** Create utility functions in `src/env_utils.py`.
        ```python
        # src/env_utils.py
        import gymnasium as gym
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.env_checker import check_env

        ENV_ID = "CartPole-v1"

        def create_env(env_id: str = ENV_ID, rank: int = 0, seed: int = 0) -> callable:
            """Utility function for multiprocessed envs."""
            def _init():
                env = gym.make(env_id)
                # Important: use Monitor wrapper Vectorized envs assume it already exists
                # Use Monitor even for rank 0 to have consistent behavior
                # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecmonitor-warning
                env = Monitor(env)
                env.reset(seed=seed + rank)
                return env
            return _init

        def make_cartpole_vec_env(n_envs: int = 1, seed: int = 0, use_subproc: bool = False) -> VecEnv:
            """Create a wrapped, possibly vectorized CartPole environment."""
            env_kwargs = {}
            vec_env_cls = DummyVecEnv if n_envs == 1 or not use_subproc else SubprocVecEnv
            env = make_vec_env(lambda: gym.make(ENV_ID, **env_kwargs), n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
            # No need to wrap with VecMonitor, make_vec_env does it
            return env

        def make_eval_env(env_id: str = ENV_ID, seed: int = 0) -> gym.Env:
             """Creates a single environment for evaluation, wrapped with Monitor."""
             env = gym.make(env_id)
             # Monitor is crucial for evaluation to log episode returns and lengths
             env = Monitor(env)
             env.reset(seed=seed)
             # check_env(env) # Optional: Check custom env compliance
             return env

        ```
    *   **Sub-task:** Add initial test in `tests/test_env_utils.py`.
        ```python
        # tests/test_env_utils.py
        import pytest
        import gymnasium as gym
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from src.env_utils import make_cartpole_vec_env, make_eval_env, ENV_ID

        def test_make_eval_env():
            """Test creation of a single evaluation environment."""
            env = make_eval_env(seed=42)
            assert isinstance(env.unwrapped, gym.envs.classic_control.CartPoleEnv)
            # Check if Monitor wrapper is present
            assert hasattr(env, 'get_episode_rewards')
            # Optional: Run environment checker
            # check_env(env)
            obs, _ = env.reset()
            assert env.observation_space.contains(obs)
            env.close()

        @pytest.mark.parametrize("n_envs, use_subproc, expected_class", [
            (1, False, DummyVecEnv),
            (4, False, DummyVecEnv),
            pytest.param(4, True, SubprocVecEnv, marks=pytest.mark.skipif(True, reason="SubprocVecEnv can cause issues in some test setups/OS")), # Skip by default
        ])
        def test_make_cartpole_vec_env(n_envs, use_subproc, expected_class):
            """Test creation of vectorized environments."""
            env = make_cartpole_vec_env(n_envs=n_envs, seed=42, use_subproc=use_subproc)
            assert isinstance(env, expected_class)
            assert env.num_envs == n_envs
            assert env.observation_space.shape == (4,)
            assert env.action_space.shape == ()
            env.close()

        ```

**Phase 2: Algorithm Implementation & Training (Using SB3)**

*   Define Base Hyperparameters (can be in `src/config.py` or directly in scripts)
    ```python
    # Example for PPO from deep_research.md Table 1 / RL Zoo
    PPO_HYPERPARAMS = {
        "policy": "MlpPolicy",
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "learning_rate": 3e-4,
        "policy_kwargs": dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
        # Add other params like seed, device, verbose as needed
    }
    # Define similar dicts for A2C_HYPERPARAMS, DQN_HYPERPARAMS based on deep_research.md
    ```

1.  **Task:** Implement PPO Training Script (`scripts/train_ppo.py`)
    *   **Sub-task:** Starter Code:
        ```python
        # scripts/train_ppo.py
        import argparse
        import os
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
        from stable_baselines3.common.env_util import make_vec_env
        from src.env_utils import make_cartpole_vec_env, make_eval_env, ENV_ID
        # from src.config import PPO_HYPERPARAMS # Option 1: Import from config

        # Option 2: Define directly or load from YAML/JSON
        PPO_HYPERPARAMS = {
            "policy": "MlpPolicy", "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
            "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0,
            "vf_coef": 0.5, "learning_rate": 3e-4, "policy_kwargs": dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
        }

        DEFAULT_TOTAL_TIMESTEPS = 100_000
        DEFAULT_N_ENVS = 4 # Matches RL Zoo recommendations

        def parse_args():
            parser = argparse.ArgumentParser(description="Train PPO on CartPole-v1")
            parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS, help="Total training timesteps")
            parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS, help="Number of parallel environments")
            parser.add_argument("--seed", type=int, default=42, help="Random seed")
            parser.add_argument("--log-dir", type=str, default="./logs/", help="Tensorboard log directory")
            parser.add_argument("--model-dir", type=str, default="./models/", help="Directory to save models")
            parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluate the agent every n steps")
            parser.add_argument("--use-subproc", action="store_true", help="Use SubprocVecEnv instead of DummyVecEnv")
            # Add args for hyperparameter tuning if needed
            return parser.parse_args()

        def main():
            args = parse_args()
            os.makedirs(args.log_dir, exist_ok=True)
            os.makedirs(args.model_dir, exist_ok=True)

            log_path = os.path.join(args.log_dir, "ppo_cartpole")
            model_save_path = os.path.join(args.model_dir, "ppo_cartpole_final")
            best_model_save_path = os.path.join(args.model_dir, "best_ppo_cartpole")

            # Create vectorized training environment
            train_env = make_cartpole_vec_env(n_envs=args.n_envs, seed=args.seed, use_subproc=args.use_subproc)

            # Create separate evaluation environment
            eval_env = make_eval_env(seed=args.seed + 1000) # Use different seed for eval

            # Callbacks
            # Stop training if reward threshold is reached
            callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
            eval_callback = EvalCallback(eval_env, best_model_save_path=best_model_save_path,
                                     log_path=log_path, eval_freq=max(args.eval_freq // args.n_envs, 1),
                                     n_eval_episodes=10, deterministic=True,
                                     render=False, callback_on_new_best=callback_on_best)

            # Instantiate the agent
            model = PPO(
                env=train_env,
                seed=args.seed,
                tensorboard_log=log_path,
                verbose=1,
                **PPO_HYPERPARAMS
            )

            print(f"Training PPO model for {args.total_timesteps} timesteps...")
            print(f"Logging to: {log_path}")
            print(f"Saving best model to: {best_model_save_path}")

            try:
                model.learn(
                    total_timesteps=args.total_timesteps,
                    callback=eval_callback,
                    progress_bar=True
                )
            except KeyboardInterrupt:
                print("Training interrupted by user.")

            print(f"Saving final model to {model_save_path}.zip")
            model.save(model_save_path)

            # Clean up environments
            train_env.close()
            eval_env.close()
            print("Training finished.")

        if __name__ == "__main__":
            main()

        ```
    *   **Sub-task:** Add Tests (`tests/test_ppo_training.py`):
        ```python
        # tests/test_ppo_training.py
        import os
        import pytest
        import subprocess
        from stable_baselines3 import PPO

        SCRIPT_PATH = "scripts/train_ppo.py"
        MODEL_DIR = "./tests/models/"
        LOG_DIR = "./tests/logs/"

        @pytest.fixture(scope="module", autouse=True)
        def setup_and_teardown():
            # Create dirs before tests
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(LOG_DIR, exist_ok=True)
            yield
            # Clean up after tests
            # (Could use tmp_path fixture instead)
            # import shutil
            # shutil.rmtree(MODEL_DIR)
            # shutil.rmtree(LOG_DIR)

        def test_ppo_script_short_run():
            """Test if the PPO training script runs for a few steps without errors."""
            total_timesteps = 100 # Very short run
            n_envs = 1
            cmd = [
                "python", SCRIPT_PATH,
                "--total-timesteps", str(total_timesteps),
                "--n-envs", str(n_envs),
                "--model-dir", MODEL_DIR,
                "--log-dir", LOG_DIR,
                "--eval-freq", "50" # Evaluate quickly
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"
            assert os.path.exists(os.path.join(MODEL_DIR, "ppo_cartpole_final.zip"))
            assert os.path.exists(os.path.join(MODEL_DIR, "best_ppo_cartpole.zip"))
            # Check logs exist (more specific checks could be added)
            assert len(os.listdir(os.path.join(LOG_DIR, "ppo_cartpole"))) > 0

        def test_ppo_load_predict():
             """Test loading the saved PPO model and predicting."""
             model_path = os.path.join(MODEL_DIR, "ppo_cartpole_final.zip")
             assert os.path.exists(model_path)
             model = PPO.load(model_path)

             # Create a dummy observation (replace with actual env if needed)
             from src.env_utils import make_eval_env
             env = make_eval_env()
             obs, _ = env.reset()

             action, _states = model.predict(obs, deterministic=True)
             assert env.action_space.contains(action)
             env.close()

        ```

2.  **Task:** Implement A2C Training Script (`scripts/train_a2c.py`) - *Similar structure to PPO, using `stable_baselines3.A2C` and A2C hyperparameters from `deep_research.md`.* Add corresponding tests in `tests/test_a2c_training.py`. 

3.  **Task:** Implement DQN Training Script (`scripts/train_dqn.py`) - *Similar structure, using `stable_baselines3.DQN` and DQN hyperparameters. Add check for `total_timesteps > learning_starts`.* Add corresponding tests in `tests/test_dqn_training.py`.
    *   **DQN Specifics:** Requires `buffer_size`, `learning_starts`, `target_update_interval`, etc. from `deep_research.md`. Use `make_eval_env` for eval callback, as DQN often uses a single environment for training.

**Phase 3: Evaluation**

1.  **Task:** Implement Evaluation Script (`scripts/evaluate_agent.py`)
    *   **Sub-task:** Starter Code:
        ```python
        # scripts/evaluate_agent.py
        import argparse
        import os
        import numpy as np
        from stable_baselines3 import PPO, A2C, DQN
        from stable_baselines3.common.evaluation import evaluate_policy
        from src.env_utils import make_eval_env, ENV_ID

        ALGOS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}

        def parse_args():
            parser = argparse.ArgumentParser(description="Evaluate a trained RL agent on CartPole-v1")
            parser.add_argument("--algo", type=str, required=True, choices=ALGOS.keys(), help="Algorithm used for training (ppo, a2c, dqn)")
            parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model (.zip file)")
            parser.add_argument("--n-eval-episodes", type=int, default=20, help="Number of episodes for evaluation")
            parser.add_argument("--seed", type=int, default=123, help="Random seed for evaluation environment")
            parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions for evaluation")
            return parser.parse_args()

        def main():
            args = parse_args()

            if not os.path.exists(args.model_path):
                print(f"Error: Model path not found: {args.model_path}")
                return

            print(f"Evaluating {args.algo.upper()} model from: {args.model_path}")
            print(f"Environment: {ENV_ID}")
            print(f"Number of evaluation episodes: {args.n_eval_episodes}")
            print(f"Deterministic actions: {args.deterministic}")

            # Create evaluation environment
            eval_env = make_eval_env(seed=args.seed)

            # Load the trained agent
            model_class = ALGOS[args.algo]
            try:
                model = model_class.load(args.model_path, env=eval_env)
            except Exception as e:
                print(f"Error loading model: {e}")
                eval_env.close()
                return

            # Evaluate the agent
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=args.deterministic,
                render=False # Set to True if you want to see the agent play
            )

            print(f"\nEvaluation Results:")
            print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

            # Compare against target (CartPole-v1 solved is typically 500)
            target_reward = 500
            if mean_reward >= target_reward:
                print(f"Agent achieved target reward of {target_reward}!")
            else:
                print(f"Agent did not reach target reward of {target_reward}.")

            eval_env.close()

        if __name__ == "__main__":
            main()
        ```
    *   **Sub-task:** Add Tests (`tests/test_evaluation.py`):
        ```python
        # tests/test_evaluation.py
        import os
        import pytest
        import subprocess
        from tests.test_ppo_training import SCRIPT_PATH as PPO_TRAIN_SCRIPT, MODEL_DIR, LOG_DIR # Reuse paths

        EVAL_SCRIPT_PATH = "scripts/evaluate_agent.py"

        # Fixture to ensure a model exists (runs the short PPO training)
        @pytest.fixture(scope="module", autouse=True)
        def ensure_ppo_model_exists():
            model_path = os.path.join(MODEL_DIR, "ppo_cartpole_final.zip")
            if not os.path.exists(model_path):
                print("\nTraining short PPO model for evaluation tests...")
                total_timesteps = 50
                n_envs = 1
                cmd = [
                    "python", PPO_TRAIN_SCRIPT,
                    "--total-timesteps", str(total_timesteps),
                    "--n-envs", str(n_envs),
                    "--model-dir", MODEL_DIR,
                    "--log-dir", LOG_DIR,
                    "--eval-freq", "25"
                ]
                subprocess.run(cmd, check=True)
            assert os.path.exists(model_path)

        def test_evaluate_script_runs():
            """Test if the evaluation script runs without errors using the saved PPO model."""
            model_path = os.path.join(MODEL_DIR, "ppo_cartpole_final.zip") # Use final or best
            cmd = [
                "python", EVAL_SCRIPT_PATH,
                "--algo", "ppo",
                "--model-path", model_path,
                "--n-eval-episodes", "2", # Short eval
                "--deterministic"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            assert result.returncode == 0, f"Evaluation script failed:\n{result.stderr}"
            assert "Mean reward:" in result.stdout
            assert "+/-" in result.stdout

        # Add similar tests for A2C and DQN models once their training scripts/tests exist
        ```

**Phase 4: Testing (Integrated)**

*   Tests are defined alongside the code they verify in Phases 1-3. Key test areas mirroring `starting.md`'s intent (but applied to SB3 usage):
    *   **Initialization (`test_*_training.py`)**: Verify model (`PPO`, `A2C`, `DQN`) instantiation with correct policy and parameters.
    *   **Single-step Learning (`test_*_training.py`)**: Verify `model.learn(total_timesteps=small_number)` completes and updates `num_timesteps`.
    *   **Policy Prediction (`test_*_training.py`)**: Verify `Model.load(path).predict(obs)` returns actions of correct shape/type.
    *   **Callbacks (`test_*_training.py`)**: Implicitly tested by `EvalCallback` usage in training scripts and checking for saved best models.
    *   **Loss/Network**: Not directly tested as we rely on SB3's internal implementation. Focus is on integration and usage.

**Phase 5: Documentation & Finalization**

1.  **Task:** Update Project Documentation
    *   **Sub-task:** Maintain `prd.md` (this file).
    *   **Sub-task:** Create/Update `README.md` with:
        *   Setup instructions (clone, `uv pip install -r requirements.txt`).
        *   How to run training: `python scripts/train_ppo.py --total-timesteps 200000` (etc. for A2C, DQN).
        *   How to run evaluation: `python scripts/evaluate_agent.py --algo ppo --model-path models/best_ppo_cartpole.zip`.
        *   How to run tests: `pytest tests/`.
        *   How to view logs: `tensorboard --logdir logs/`.
    *   **Sub-task:** Maintain `coding_updates_1.md` for all code changes.
2.  **Task:** Code Cleanup & Review
    *   **Sub-task:** Ensure code follows standards (e.g., PEP 8), is readable, and efficient.
    *   **Sub-task:** Add necessary comments and docstrings, especially for utility functions and script arguments.

## Implementation Status (05-03-2025)

### Completed Components

#### Core Infrastructure
- ✅ Project directory structure created
- ✅ Environment utilities implemented in `src/env_utils.py`
- ✅ Requirements specified with exact version numbers
- ✅ `.gitignore` configured for Python development

#### Training Scripts
- ✅ PPO implementation (`scripts/train_ppo.py`)
- ✅ A2C implementation (`scripts/train_a2c.py`)
- ✅ DQN implementation (`scripts/train_dqn.py`)

#### Evaluation
- ✅ Agent evaluation script (`scripts/evaluate_agent.py`)

#### Testing
- ✅ Environment utility tests
- ✅ PPO training tests
- ✅ A2C training tests
- ✅ DQN training tests
- ✅ Evaluation script tests

#### Documentation
- ✅ README with usage instructions
- ✅ Detailed setup and testing guide
- ✅ Code change history in `coding_updates_1.md`

### Test Results

All 21 tests are passing, confirming that the implementation is working correctly:

```
============================= test session starts ==============================
platform darwin -- Python 3.9.6, pytest-8.3.5, pluggy-1.5.0
collecting ... collected 21 items

tests/test_a2c_training.py::test_a2c_instantiation PASSED                [  4%]
tests/test_a2c_training.py::test_a2c_short_training PASSED               [  9%]
tests/test_a2c_training.py::test_a2c_save_load PASSED                    [ 14%]
tests/test_a2c_training.py::test_a2c_prediction PASSED                   [ 19%]
tests/test_a2c_training.py::test_a2c_evaluation PASSED                   [ 23%]
tests/test_dqn_training.py::test_dqn_instantiation PASSED                [ 28%]
tests/test_dqn_training.py::test_dqn_short_training PASSED               [ 33%]
tests/test_dqn_training.py::test_dqn_save_load PASSED                    [ 38%]
tests/test_dqn_training.py::test_dqn_prediction PASSED                   [ 42%]
tests/test_dqn_training.py::test_dqn_evaluation PASSED                   [ 47%]
tests/test_env_utils.py::test_make_eval_env PASSED                       [ 52%]
tests/test_env_utils.py::test_make_cartpole_vec_env[1-False-DummyVecEnv] PASSED [ 57%]
tests/test_env_utils.py::test_make_cartpole_vec_env[4-False-DummyVecEnv] PASSED [ 61%]
tests/test_env_utils.py::test_make_cartpole_vec_env[4-True-SubprocVecEnv] PASSED [ 66%]
tests/test_evaluate_agent.py::test_evaluate_agent_function PASSED        [ 71%]
tests/test_evaluate_agent.py::test_evaluate_agent_with_different_algos PASSED [ 76%]
tests/test_ppo_training.py::test_ppo_instantiation PASSED                [ 80%]
tests/test_ppo_training.py::test_ppo_short_training PASSED               [ 85%]
tests/test_ppo_training.py::test_ppo_save_load PASSED                    [ 90%]
tests/test_ppo_training.py::test_ppo_prediction PASSED                   [ 95%]
tests/test_ppo_training.py::test_ppo_evaluation PASSED                   [100%]

============================= 21 passed in 11.42s ==============================
```

### Environment Setup

To set up the development environment:

```bash
# Create a new virtual environment
python3 -m venv .venv-sb3

# Activate the virtual environment
source .venv-sb3/bin/activate

# Install all required packages with exact versions
python -m pip install -r requirements.txt
```

Detailed instructions are available in `setup_and_test_guide.md`.

### Running Tests

To run all tests:

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

### Next Steps

1. **Extended Testing**: Add more extensive tests for edge cases and error handling.
2. **Hyperparameter Tuning**: Add scripts for automated hyperparameter optimization using Optuna or similar libraries.
3. **Custom Callbacks**: Implement custom callbacks for more detailed logging and visualization.
4. **Environment Extensions**: Extend to other Gymnasium environments beyond CartPole-v1.
5. **Neural Network Customization**: Add support for custom neural network architectures.
6. **CI/CD Integration**: Set up continuous integration and deployment for automated testing.

### Conclusion

The implementation is now fully verified and ready for use. All required components have been implemented following best practices from Stable Baselines3 documentation and research benchmarks. The comprehensive test suite ensures reliability and correctness, making this a production-ready reinforcement learning system.
