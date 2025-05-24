from simulation import WORLD_SIZE
import pymunk.pygame_util
import pymunk
import sys
import pygame


class Renderer:
    def __init__(self, sim):
        pygame.init()
        pygame.display.set_caption("Pendulum Simulation")
        screen_size = (1000, 1000)
        self.screen = pygame.display.set_mode(
            (screen_size[0], screen_size[1]), pygame.RESIZABLE
        )

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        scale_x = screen_size[0] / (WORLD_SIZE * 2)
        scale_y = screen_size[1] / (WORLD_SIZE * 2)
        scale = min(scale_x, scale_y)
        offset_x = screen_size[0] / 2
        offset_y = screen_size[1] / 2
        transform = pymunk.Transform(
            a=scale, d=scale, tx=offset_x, ty=offset_y)
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
        if keys[pygame.K_a]:
            self.sim.reset(angle=3.14 * 1.5)
        if keys[pygame.K_d]:
            self.sim.reset(angle=3.14 * 0.5)
        if keys[pygame.K_e]:
            self.sim.reset(angle=3.14 * 0.75)
        if keys[pygame.K_q]:
            self.sim.reset(angle=-3.14 * 0.75)
        if keys[pygame.K_t]:
            self.sim.reset(self.sim.angle())

    def render(self):
        self.handle_input()
        self.update()
        self.draw()
        self.clock_tick()

    def run(self):
        while True:
            self.render()
