from renderer import Renderer
from cartpole import CartPoleSimulation


class ManualRenderer(Renderer):
    def __init__(self, sim):
        super().__init__(sim)

    def update(self):
        super().update()
        print("curr tick reward: ", self.sim.compute_reward())


def main():
    renderer = ManualRenderer(CartPoleSimulation())
    renderer.run()


if __name__ == "__main__":
    main()
