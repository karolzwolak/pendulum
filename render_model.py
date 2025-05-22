from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from cartpole import CartPoleSimulation
from model import Env
from renderer import Renderer


class RenderableEnv(Renderer):
    def __init__(self, sim):
        super().__init__(sim)
        self.env = Env(sim)
        check_env(self.env, warn=True)
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        self.model.load("models/cartpole")

    def update(self):
        action, _ = self.model.predict(self.sim.state(), deterministic=True)
        _, _, done, _, _ = self.env.step(action)
        if done:
            self.sim.reset()


if __name__ == "__main__":
    sim = CartPoleSimulation()
    renderer = RenderableEnv(sim)

    renderer.run()
