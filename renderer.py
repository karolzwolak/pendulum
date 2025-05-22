from simulation import WORLD_SIZE
import pymunk.pygame_util
import pymunk
import sys
import pygame


class Renderer:
    def __init__(self, sim):
        pygame.init()
        info = pygame.display.Info()
        pygame.display.set_caption("Pendulum Simulation")
        self.screen = pygame.display.set_mode((info.current_w, info.current_h))
        pygame.display.toggle_fullscreen()

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        scale_x = info.current_w / (WORLD_SIZE * 2)
        scale_y = info.current_h / (WORLD_SIZE * 2)
        scale = min(scale_x, scale_y)
        offset_x = info.current_w / 2
        offset_y = info.current_h / 2
        transform = pymunk.Transform(a=scale, d=scale, tx=offset_x, ty=offset_y)
        self.draw_options.transform = transform

        self.clock = pygame.time.Clock()
        self.sim = sim

    def clear(self):
        self.screen.fill((255, 255, 255))

    def update(self):
        self.sim.step()

    def draw(self):
        self.clear()
        self.sim.draw(self.draw_options)
        pygame.display.flip()

    def clock_tick(self):
        self.clock.tick(self.sim.frequency)

    def exit(self):
        pygame.quit()
        sys.exit()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.exit()

    def render(self):
        self.handle_input()
        self.update()
        self.draw()
        self.clock_tick()

    def run(self):
        while True:
            self.render()
