import sys
from model import load_model
from renderer import Renderer


class RenderableEnv(Renderer):
    def __init__(self, load_model=load_model, path="models/cartpole"):
        sim, self.env, self.model = load_model(path=path)
        super().__init__(sim)
        self.obs = self.env.reset()
        self.total_reward = 0

    def update(self):
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, reward, done, _ = self.env.step(action)
        reward = reward[0]
        print(f"reward: {reward:.2f} action: {action[0][0]:.2f}")
        self.total_reward += reward
        if done:
            self.obs = self.env.reset()
            print("========== Episode finished ==========")
            print(f"Total reward: {self.total_reward:.2f}")
            print("======================================")
            self.total_reward = 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) > 0:
        renderer = RenderableEnv(path=argv[0])
    else:
        renderer = RenderableEnv()

    renderer.run()
