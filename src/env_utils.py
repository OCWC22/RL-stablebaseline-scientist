import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.env_checker import check_env # Keep commented out for now

ENV_ID = "CartPole-v1"

def create_env(env_id: str = ENV_ID, rank: int = 0, seed: int = 0) -> callable:
    """Utility function for multiprocessed envs, used by make_vec_env.

    :param env_id: the environment ID
    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    :return: a function that creates the environment
    """
    def _init():
        env = gym.make(env_id)
        # Important: use Monitor wrapper even for single-process training
        # A vectorized environment wraps things automatically in DummyVecEnv or SubprocVecEnv
        # which take care of multiple instances. For evaluation, we need Monitor manually.
        # make_vec_env automatically wraps the envs in Monitor unless monitor_dir is None.
        # We will let make_vec_env handle Monitor wrapping for training.
        env.reset(seed=seed + rank)
        return Monitor(env) # Return Monitor-wrapped env for consistency in creation process
    return _init

def make_cartpole_vec_env(n_envs: int = 1, seed: int = 0, use_subproc: bool = False) -> VecEnv:
    """Create a wrapped, possibly vectorized CartPole environment for training.

    :param n_envs: Number of parallel environments
    :param seed: The initial seed for RNG
    :param use_subproc: Whether to use SubprocVecEnv or DummyVecEnv
    :return: The vectorized environment
    """
    env_kwargs = {}
    # Use DummyVecEnv for n_envs=1 or when subproc is False
    # Use SubprocVecEnv for n_envs > 1 and use_subproc is True
    vec_env_cls = SubprocVecEnv if n_envs > 1 and use_subproc else DummyVecEnv

    # make_vec_env uses the create_env function internally and handles Monitor wrapping
    # if monitor_dir is not None (default is None).
    # For SB3, logging is usually handled by Monitor wrapper automatically added by make_vec_env
    # or explicitly via callbacks / VecMonitor.
    # Let's simplify and let make_vec_env handle its default wrapping.
    # We ensure Monitor is part of the base creation in `create_env` just in case.
    env = make_vec_env(
        env_id=ENV_ID,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
        env_kwargs=env_kwargs,
        # monitor_dir=None # Let make_vec_env use its default Monitor behavior if needed
    )
    return env

def make_eval_env(env_id: str = ENV_ID, seed: int = 0) -> gym.Env:
     """Creates a single environment for evaluation, ensuring it's wrapped with Monitor.

     :param env_id: The environment ID
     :param seed: The initial seed for RNG
     :return: The evaluation environment
     """
     # Use the same creation logic but ensure Monitor is there
     env = gym.make(env_id)
     # Monitor is essential for evaluation to track episode rewards/lengths
     env = Monitor(env)
     env.reset(seed=seed)
     # Optional: check compliance, useful for custom envs
     # from stable_baselines3.common.env_checker import check_env
     # check_env(env)
     return env