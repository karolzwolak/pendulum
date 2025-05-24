from renderer import Renderer
from cartpole import CartPoleSimulation


class ManualRenderer(Renderer):
    def __init__(self, sim):
        super().__init__(sim)

    def update(self):
        super().update()
        print("curr tick reward: ", self.sim.compute_reward())
        if self.sim.is_done():
            self.sim.reset()


def main():
    renderer = ManualRenderer(CartPoleSimulation())
    renderer.run()


if __name__ == "__main__":
    main()
