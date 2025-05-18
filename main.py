import pygame
import sys

PENDULUM_MASS_COLOR = (255, 0, 0)
PENDULUM_BASE_COLOR = (0, 0, 255)


class Pendulum:
    def __init__(self, length, angle, mass, base_position, mass_position):
        self.length = length
        self.angle = angle
        self.mass = mass
        self.base_position = base_position
        self.mass_position = mass_position

    def move_base(self, dx, dy):
        self.base_position = (self.base_position[0] + dx, self.base_position[1] + dy)

    def draw(self, screen):
        pygame.draw.circle(
            screen,
            PENDULUM_BASE_COLOR,
            (int(self.base_position[0]), int(self.base_position[1])),
            self.mass,
        )
        pygame.draw.circle(
            screen,
            PENDULUM_MASS_COLOR,
            (int(self.mass_position[0]), int(self.mass_position[1])),
            self.mass,
        )


class Renderer:
    def __init__(self, screen, pendulum):
        self.screen = screen
        self.pendulum = pendulum

    def clear(self):
        self.screen.fill((255, 255, 255))

    def update(self):
        pass

    def draw(self):
        self.clear()
        self.pendulum.draw(self.screen)
        pygame.display.flip()

    def loop(self):
        pygame.init()
        clock = pygame.time.Clock()
        _ = clock.tick(60)
        while True:
            self.handle_input()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.draw()

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.pendulum.move_base(0, -1)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.pendulum.move_base(0, 1)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.pendulum.move_base(-1, 0)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.pendulum.move_base(1, 0)
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()


def main():
    pygame.init()
    play_surface = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Pendulum Simulation")
    pendulum = Pendulum(100, 45, 10, (400, 300), (400, 100))
    renderer = Renderer(play_surface, pendulum)
    renderer.loop()


if __name__ == "__main__":
    main()
