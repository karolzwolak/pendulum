from collections import deque

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from tqdm import tqdm

from cartpole import CartPoleSimulation
from env import Env
from simulation import MAX_STEPS

NUM_ENVS = 16


class CurriculumLearningCallback(BaseCallback):
    def __init__(
        self,
        stop_traning_reward_threshold: float = 80,
        progress_curriculum_theshold: float = 80,
        curriculum_steps: int = 10,
        n_episodes: int = 2,
        verbose=0,
    ):
        super().__init__(verbose)
        self.stop_traning_reward_threshold = stop_traning_reward_threshold
        self.progress_curriculum_reward_threshold = progress_curriculum_theshold
        self.n_episodes = n_episodes
        self.episode_rewards = deque(maxlen=n_episodes)
        self.curriculum_steps = curriculum_steps
        self.curriculum_step = 0

    def _on_training_start(self) -> None:
        self.training_env.env_method("progress_curriculum", 0)
        self.model._current_progress_remaining = 1.0
        self.pbar = tqdm(
            total=self.curriculum_steps,
            desc="Curriculum Progress",
            dynamic_ncols=True,
        )

    def _on_training_end(self) -> None:
        self.pbar.close()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is not None and infos is not None:
            for done, info in zip(dones, infos):
                if done and "episode" in info:
                    ep_reward = info["episode"]["r"]
                    self.episode_rewards.append(ep_reward)

        if len(self.episode_rewards) < self.n_episodes:
            return True

        mean_reward = sum(self.episode_rewards) / self.n_episodes

        finished_curriculum = self.curriculum_step >= self.curriculum_steps
        if (
            not finished_curriculum
            and mean_reward >= self.progress_curriculum_reward_threshold
        ):
            self.curriculum_step += 1
            t = self.curriculum_step / self.curriculum_steps

            self.training_env.env_method("progress_curriculum", t)
            self.episode_rewards.clear()

            self._current_progress_remaining = 1.0 - t
            self.pbar.n = self.curriculum_step
            self.pbar.refresh()

        if finished_curriculum and mean_reward >= self.stop_traning_reward_threshold:
            print(
                f"Threshold reached: mean reward = {mean_reward:.2f} ≥ {self.stop_traning_reward_threshold}"
                ", stopping training."
            )
            return False
        return True


def linear_schedule(initial_value):
    def func(progress):
        return initial_value * progress

    return func


def create_model(env):
    policy_kwargs = dict(net_arch=[64, 32], activation_fn=th.nn.LeakyReLU)
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
