import sys
import pygame
from renderer import Renderer
from cartpole import CartPoleSimulation


class ManualRenderer(Renderer):
    def __init__(self, sim):
        super().__init__(sim)

    def handle_input(self):
        super().handle_input()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            self.sim.apply_force(-1)
        if keys[pygame.K_RIGHT]:
            self.sim.apply_force(1)
        if keys[pygame.K_r]:
            self.sim.reset()


def main():
    renderer = ManualRenderer(CartPoleSimulation())
    renderer.run()


if __name__ == "__main__":
    main()
