import pymunk.pygame_util
import pymunk
import sys
import pygame


class Renderer:
    def __init__(self, crate_sim_fn):
        pygame.init()
        info = pygame.display.Info()
        pygame.display.set_caption("Pendulum Simulation")
        self.screen = pygame.display.set_mode((info.current_w, info.current_h))
        pygame.display.toggle_fullscreen()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()
        self.sim = crate_sim_fn(self.draw_options)

    def clear(self):
        self.screen.fill((255, 255, 255))

    def update(self):
        self.sim.step()

    def draw(self):
        self.clear()
        self.sim.draw(self.draw_options)
        pygame.display.flip()

    def loop(self):
        while True:
            self.handle_input()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.update()
            self.draw()
            self.clock.tick(self.sim.frequency)
