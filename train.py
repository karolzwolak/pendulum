from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

from model import create_model, make_parallel_normalized_envs, save_model


class StopTrainingCallback(BaseCallback):
    def __init__(self, reward_threshold: float, n_episodes: int = 10, verbose=0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.n_episodes = n_episodes
        self.episode_rewards = deque(maxlen=n_episodes)

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

        if mean_reward >= self.reward_threshold:
            print(
                f"Threshold reached: mean reward = {mean_reward:.2f} ≥ {self.reward_threshold}"
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
            total_timesteps=1_000_000,
            progress_bar=True,
            callback=StopTrainingCallback(reward_threshold=80),
        )
    except KeyboardInterrupt:
        if input("Save model? (y/n): ").lower() == "y":
            save_model(model, env)
        return
    save_model(model, env)


if __name__ == "__main__":
    main()
