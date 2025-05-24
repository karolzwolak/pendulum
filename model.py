from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from cartpole import CartPoleSimulation
from env import Env


def create_model(env):
    return PPO("MlpPolicy", env, verbose=1, device="cpu")


def load_model(model_path, env):
    return PPO.load(model_path, env, device="cpu")


def make_parallel_envs(num_envs=4):
    """Create multiple parallel environments for faster sampling"""
    return make_vec_env(
        lambda: Env(CartPoleSimulation()), n_envs=num_envs, vec_env_cls=SubprocVecEnv
    )
