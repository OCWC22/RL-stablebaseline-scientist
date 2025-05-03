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
    # Check if Monitor wrapper is present (it should be added by make_eval_env)
    assert hasattr(env, 'get_episode_rewards')
    # Optional: Run environment checker
    # check_env(env) # Can be slow, uncomment if needed
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    env.close()

@pytest.mark.parametrize("n_envs, use_subproc, expected_class", [
    (1, False, DummyVecEnv), # n_envs=1 should always use DummyVecEnv
    (4, False, DummyVecEnv), # n_envs>1, use_subproc=False -> DummyVecEnv
    # pytest.param(4, True, SubprocVecEnv, marks=pytest.mark.skipif(os.name == 'nt', reason="SubprocVecEnv can be problematic on Windows in tests")), # Conditional skip
    (4, True, SubprocVecEnv), # n_envs>1, use_subproc=True -> SubprocVecEnv (assuming not Windows or issues resolved)
])
def test_make_cartpole_vec_env(n_envs, use_subproc, expected_class):
    """Test creation of vectorized environments for training."""
    # Added check to skip SubprocVecEnv test on Windows if necessary, or manage potential issues
    import os
    if use_subproc and expected_class == SubprocVecEnv and os.name == 'nt':
         pytest.skip("SubprocVecEnv test skipped on Windows due to potential issues")

    env = make_cartpole_vec_env(n_envs=n_envs, seed=42, use_subproc=use_subproc)
    assert isinstance(env, expected_class)
    assert env.num_envs == n_envs
    # Check observation and action spaces (CartPole specific)
    assert env.observation_space.shape == (4,)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 2

    # Test reset and step
    obs = env.reset()
    assert obs.shape == (n_envs, 4)
    action = env.action_space.sample()
    # Ensure action is compatible; for VecEnv, actions need to be an array matching n_envs
    # However, env.action_space.sample() might return a single action if not handled correctly by the VecEnv wrapper
    # SB3 models handle this internally, but for direct testing:
    actions = [env.action_space.sample() for _ in range(n_envs)]
    obs, rewards, dones, infos = env.step(actions)

    assert obs.shape == (n_envs, 4)
    assert rewards.shape == (n_envs,)
    assert dones.shape == (n_envs,)
    # infos is a list of dicts, one per env
    assert isinstance(infos, list)
    assert len(infos) == n_envs

    env.close()

