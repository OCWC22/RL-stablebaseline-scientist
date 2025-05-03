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
    # Use DummyVecEnv for n_envs=1 or when subproc is False
    # Use SubprocVecEnv for n_envs > 1 and use_subproc is True
    vec_env_cls = SubprocVecEnv if n_envs > 1 and use_subproc else DummyVecEnv

    # The lambda function ensures each process gets a different environment instance
    # Monitor wrapper is automatically added by make_vec_env unless vec_env_cls is specified
    # Since we specify vec_env_cls, Monitor might not be added automatically.
    # However, create_env already includes Monitor, so it should be fine.
    # Let's rely on create_env which includes Monitor
    env = make_vec_env(env_id=ENV_ID, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls, monitor_dir=None)

    return env

def make_eval_env(env_id: str = ENV_ID, seed: int = 0) -> gym.Env:
     """Creates a single environment for evaluation, wrapped with Monitor."""
     env = gym.make(env_id)
     # Monitor is crucial for evaluation to log episode returns and lengths
     env = Monitor(env)
     env.reset(seed=seed)
     # check_env(env) # Optional: Check custom env compliance
     return env