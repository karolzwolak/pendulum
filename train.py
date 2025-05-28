from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

from model import create_model, make_parallel_normalized_envs, save_model


class CurriculumLearningCallback(BaseCallback):
    def __init__(
        self,
        stop_traning_reward_threshold: float = 70,
        progress_curriculum_theshold: float = 70,
        curriculum_steps: int = 100,
        n_episodes: int = 10,
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
            print(f"Progressing curriculum with t={t}")
            self.episode_rewards.clear()

        if finished_curriculum and mean_reward >= self.stop_traning_reward_threshold:
            print(
                f"Threshold reached: mean reward = {mean_reward:.2f} ≥ {self.stop_traning_reward_threshold}"
                ", stopping training."
            )
            return False
        return True


def main():
    # Use parallel environments
    env = make_parallel_normalized_envs()

    model = create_model(env)

    try:
        model.learn(
            total_timesteps=10_000_000,
            progress_bar=True,
            callback=CurriculumLearningCallback(),
        )
    except KeyboardInterrupt:
        if input("Save model? (y/n): ").lower() == "y":
            save_model(model, env)
        return
    save_model(model, env)


if __name__ == "__main__":
    main()
