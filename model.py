from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from cartpole import CartPoleSimulation
from simulation import MAX_STEPS
import torch as th
from env import Env

NUM_ENVS = 32


def linear_schedule(initial_value):
    def func(progress):
        return initial_value * progress

    return func


def create_model(env):
    policy_kwargs = dict(net_arch=[16, 8], activation_fn=th.nn.Tanh)
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        policy_kwargs=policy_kwargs,
        n_steps=MAX_STEPS,
        batch_size=NUM_ENVS * MAX_STEPS,
        n_epochs=20,
        learning_rate=linear_schedule(5e-3),
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.01,
    )


def save_model(model, env, path="models/cartpole"):
    """Save the model to the specified path."""
    model.save(path)
    env.save(f"{path}_vec_normalize.pkl")
    print(f"Model saved to {path}")


def make_parallel_envs(num_envs=NUM_ENVS):
    """Create multiple parallel environments for faster sampling"""
    return make_vec_env(
        lambda: Env(CartPoleSimulation()),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
    )


def make_parallel_normalized_envs(num_envs=NUM_ENVS):
    return VecNormalize(make_parallel_envs(num_envs), norm_obs=True, norm_reward=True)


def load_model_for_training(vec_env=make_parallel_envs, path="models/cartpole"):
    eval_env = VecNormalize.load(f"{path}_vec_normalize.pkl", vec_env())

    eval_env.training = False
    eval_env.norm_reward = False

    return eval_env, PPO.load(path, env=eval_env, device="cpu")


def load_model(path, sim=CartPoleSimulation):
    sim = sim()
    vec_env = DummyVecEnv([lambda: Env(sim)])
    eval_env = VecNormalize.load(f"{path}_vec_normalize.pkl", vec_env)

    eval_env.training = False
    eval_env.norm_reward = False

    return sim, eval_env, PPO.load(f"{path}.zip", env=eval_env, device="cpu")
