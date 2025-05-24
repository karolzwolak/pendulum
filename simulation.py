import pymunk

WORLD_SIZE = 100


class Simulation:
    def __init__(
        self,
        cart_mass=1,
        cart_radius=2,
        gravity=(0, 981),
        frequency=60,
        max_steps=1000,
    ):
        self.space = pymunk.Space()

        self.cart_body = pymunk.Body(cart_mass)
        self.cart_body.moment = float("inf")  # Prevent it from rotating
        self.cart_body.position = (0, 0)

        self.cart_shape = pymunk.Circle(self.cart_body, cart_radius)

        self.space.add(self.cart_body, self.cart_shape)

        self.gravity = gravity
        self.space.gravity = self.gravity
        self.frequency = frequency
        self.dt = 1.0 / frequency
        self.max_steps = max_steps
        self.steps = 0

    def space(self):
        raise NotImplementedError()

    def state(self):
        raise NotImplementedError()

    def compute_reward(self):
        raise NotImplementedError()

    def cart_x(self):
        return self.cart_body.position.x

    def cart_velocity_x(self):
        return self.cart_body.velocity.x

    def apply_force(self, force, mult=WORLD_SIZE * 10):
        self.cart_body.apply_force_at_local_point((force * mult, 0), (0, 0))

    def reset(self):
        self.steps = 0
        self.cart_body.position = (0, 0)
        self.cart_body.velocity = (0, 0)

    def is_done(self):
        return abs(self.cart_x()) > WORLD_SIZE or self.steps >= self.max_steps

    def step(self, force=0):
        self.steps += 1
        self.apply_force(force)
        self.space.step(self.dt)
        self.cart_body.velocity = (self.cart_body.velocity.x, 0)

        return self.state(), self.compute_reward(), self.is_done()

    def draw(self, draw_options):
        self.space.debug_draw(draw_options)

    def manually_move(self, direction, speed=WORLD_SIZE / 5):
        self.cart_body.velocity = (direction * speed, 0)
