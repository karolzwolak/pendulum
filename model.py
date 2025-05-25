from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from cartpole import CartPoleSimulation
from simulation import MAX_STEPS
from env import Env

NUM_ENVS = 16


def create_model(env):
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        policy_kwargs=policy_kwargs,
        n_steps=MAX_STEPS,
        batch_size=NUM_ENVS * MAX_STEPS,
    )


def load_model(model_path, env):
    return PPO.load(model_path, env, device="cpu")


def make_parallel_envs(num_envs=NUM_ENVS):
    """Create multiple parallel environments for faster sampling"""
    return make_vec_env(
        lambda: Env(CartPoleSimulation()), n_envs=num_envs, vec_env_cls=SubprocVecEnv
    )
