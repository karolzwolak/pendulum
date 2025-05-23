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
        if keys[pygame.K_w]:
            self.sim.reset(angle=3.14)
        if keys[pygame.K_s]:
            self.sim.reset(angle=0)
        if keys[pygame.K_d]:
            self.sim.reset(angle=3.14 * 1.5)
        if keys[pygame.K_a]:
            self.sim.reset(angle=3.14 * 0.5)
        if keys[pygame.K_e]:
            self.sim.reset(angle=-3.14 * 0.75)
        if keys[pygame.K_q]:
            self.sim.reset(angle=3.14 * 0.75)
        if keys[pygame.K_t]:
            self.sim.reset(self.sim.angle())


def main():
    renderer = ManualRenderer(CartPoleSimulation())
    renderer.run()


if __name__ == "__main__":
    main()
