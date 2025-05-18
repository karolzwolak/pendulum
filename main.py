import pygame
import sys
import pymunk
import pymunk.pygame_util

PENDULUM_MASS_COLOR = (255, 0, 0)
PENDULUM_BASE_COLOR = (0, 0, 255)
GRAVITY = (0, 981)  # pixels per second squared


class Pendulum:
    def __init__(self, length, angle, mass, base_position, space):
        self.length = length
        self.mass = mass
        self.space = space

        # Create pivot point (dynamic body for movement)
        self.pivot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.pivot_body.position = base_position

        # Create pendulum bob (dynamic body)
        self.bob_body = pymunk.Body()
        self.bob_body.position = (
            base_position[0] + length * pymunk.Vec2d(1, 0).rotated(angle).x,
            base_position[1] + length * pymunk.Vec2d(1, 0).rotated(angle).y,
        )

        # Create shapes
        self.pivot_shape = pymunk.Circle(self.pivot_body, mass)
        self.bob_shape = pymunk.Circle(self.bob_body, mass)
        self.bob_shape.mass = mass

        # Create joint
        self.joint = pymunk.PinJoint(
            self.pivot_body, self.bob_body, (0, 0), (0, 0))

        # Add to space
        self.space.add(self.pivot_body, self.pivot_shape)
        self.space.add(self.bob_body, self.bob_shape, self.joint)

    def draw(self, screen):
        pygame.draw.circle(
            screen,
            PENDULUM_BASE_COLOR,
            (int(self.pivot_body.position.x), int(self.pivot_body.position.y)),
            self.mass,
        )
        pygame.draw.circle(
            screen,
            PENDULUM_MASS_COLOR,
            (int(self.bob_body.position.x), int(self.bob_body.position.y)),
            self.mass,
        )


class Renderer:
    def __init__(self, screen, pendulum):
        self.screen = screen
        self.pendulum = pendulum
        self.space = pendulum.space
        self.space.gravity = GRAVITY
        self.draw_options = pymunk.pygame_util.DrawOptions(screen)
        self.clock = pygame.time.Clock()

    def clear(self):
        self.screen.fill((255, 255, 255))

    def update(self):
        dt = 1.0 / 60.0
        self.space.step(dt)

    def draw(self):
        self.clear()
        # Draw physics objects
        self.space.debug_draw(self.draw_options)
        # Also draw our custom pendulum graphics
        self.pendulum.draw(self.screen)
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
            self.clock.tick(60)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()

        # Handle left/right movement
        move_speed = 200  # pixels per second
        velocity = 0
        if keys[pygame.K_a]:
            velocity = -move_speed
        if keys[pygame.K_d]:
            velocity = move_speed

        self.pendulum.pivot_body.velocity = (velocity, 0)


def main():
    pygame.init()
    play_surface = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Pendulum Simulation")

    # Create physics space
    space = pymunk.Space()

    pendulum = Pendulum(200, 3.14 / 4, 10, (400, 200), space)
    renderer = Renderer(play_surface, pendulum)
    renderer.loop()


if __name__ == "__main__":
    main()
