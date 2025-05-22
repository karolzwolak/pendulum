import pymunk.pygame_util
import pymunk
import sys
from gymnasium import spaces
import pygame
import numpy as np
import gymnasium as gym
from copy import deepcopy

WIDTH = 800
HEIGHT = 600
PENDULUM_MASS_COLOR = (255, 0, 0)
PENDULUM_BASE_COLOR = (0, 0, 255)
GRAVITY = (0, 981)  # pixels per second squared


class Pendulum:
    def __init__(self, length, angle, mass, cart_mass, base_position, space):
        self.length = length
        self.mass = mass
        self.space = space

        self.pivot_body = pymunk.Body()
        self.pivot_body.mass = cart_mass
        self.pivot_body.moment = float("inf")  # Prevent it from rotating
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

    def step(self, force):
        self.pivot_body.apply_force_at_local_point((force, 0), (0, 0))

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


class CartPoleSimulation:
    def __init__(self, pendulum):
        self.initial_pendulum = deepcopy(pendulum)
        self.pendulum = pendulum
        self.gravity = GRAVITY
        self.dt = 1.0 / 60.0
        self.obs_size = 4  # [angle, angular_velocity, cart_x, cart_velocity_x]
        self.max_steps = 1000
        self.steps = 0

    def angle(self):
        return self.pendulum.bob_body.angle

    def angular_velocity(self):
        return self.pendulum.bob_body.angular_velocity

    def cart_x(self):
        world_x = self.pendulum.pivot_body.position.x
        local_x = world_x - WIDTH / 2
        normized_x = local_x / (WIDTH / 2)
        return normized_x

    def cart_velocity_x(self):
        return self.pendulum.pivot_body.velocity.x

    def state(self):
        return np.array(
            [
                self.angle(),
                self.angular_velocity(),
                self.cart_x(),
                self.cart_velocity_x(),
            ],
            dtype=np.float32,
        )

    def reset(self):
        self.pendulum = deepcopy(self.initial_pendulum)
        return self.state()

    def compute_reward(self):
        upright_bonus = np.cos(self.angle())  # 1 when angle = 0 (upright)
        position_penalty = -abs(self.cart_x()) * 0.1
        velocity_penalty = -0.01 * (
            abs(self.angular_velocity()) + abs(self.cart_velocity_x())
        )
        return upright_bonus + position_penalty + velocity_penalty + 1.0

    def is_done(self):
        return abs(self.cart_x()) > 1.0 or self.steps >= self.max_steps

    def step(self, force):
        self.steps += 1
        self.pendulum.step(force)

        return self.state(), self.compute_reward(), self.is_done()


class Env(gym.Env):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(sim.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )  # One float: direction & magnitude

    def reset(self, seed=None, options=None):
        obs = self.sim.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        # Action is a float in a 1D array
        force = float(action[0])
        obs, reward, done = self.sim.step(force)
        return np.array(obs, dtype=np.float32), reward, done, False, {}


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
    play_surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pendulum Simulation")

    # Create physics space
    space = pymunk.Space()

    pendulum = Pendulum(200, 3.14 / 4, 10, 10, (400, 200), space)
    renderer = Renderer(play_surface, pendulum)
    renderer.loop()


if __name__ == "__main__":
    main()
