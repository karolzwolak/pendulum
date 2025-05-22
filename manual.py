import sys
import pygame
from renderer import Renderer
from cartpole import CartPoleSimulation


class ManualRenderer(Renderer):
    def __init__(self, sim_class):
        super().__init__(sim_class)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()

        dir = 0
        if keys[pygame.K_a]:
            dir = -1
        if keys[pygame.K_d]:
            dir = 1

        self.sim.manually_move(dir)


def main():
    renderer = ManualRenderer(CartPoleSimulation)
    renderer.loop()


if __name__ == "__main__":
    main()
