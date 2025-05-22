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

        dir = 0
        if keys[pygame.K_a]:
            dir = -1
        if keys[pygame.K_d]:
            dir = 1

        self.sim.manually_move(dir)


def main():
    renderer = ManualRenderer(CartPoleSimulation())
    renderer.run()


if __name__ == "__main__":
    main()
