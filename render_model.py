from stable_baselines3.common.env_checker import check_env
from cartpole import CartPoleSimulation
from env import Env
from model import load_model
from renderer import Renderer


class RenderableEnv(Renderer):
    def __init__(self, sim):
        super().__init__(sim)
        self.env = Env(sim)
        check_env(self.env, warn=True)
        self.model = load_model("models/cartpole", self.env)

    def update(self):
        action, _ = self.model.predict(self.env.state(), deterministic=True)
        print(f"Action: {action[0]:.2f}")
        print(f"reward: {self.env.sim.compute_reward():.2f}")
        self.env.step(action)
        if self.env.sim.is_done():
            self.env.reset()


if __name__ == "__main__":
    sim = CartPoleSimulation()
    renderer = RenderableEnv(sim)

    renderer.run()
