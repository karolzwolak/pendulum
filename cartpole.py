import numpy as np
from simulation import Simulation
import pymunk

STARTING_ANGLE = 3.14 / 4


class CartPoleSimulation(Simulation):
    def __init__(
        self,
        length=15,
        angle=STARTING_ANGLE,
        weigth_mass=1,
        cart_mass=1,
    ):
        super().__init__(cart_mass=cart_mass)
        self.obs_size = 4  # [angle, angular_velocity, cart_x, cart_velocity_x]

        self.length = length

        # Create pendulum bob (dynamic body)
        self.bob_body = pymunk.Body()
        self.bob_body.position = (
            self.cart_body.position.x + length * pymunk.Vec2d(1, 0).rotated(angle).x,
            self.cart_body.position.y + length * pymunk.Vec2d(1, 0).rotated(angle).y,
        )

        self.bob_shape = pymunk.Circle(self.bob_body, weigth_mass)
        self.bob_shape.mass = weigth_mass

        self.joint = pymunk.PinJoint(self.cart_body, self.bob_body, (0, 0), (0, 0))

        self.space.add(self.bob_body, self.bob_shape, self.joint)

    def angle(self):
        return self.bob_body.angle

    def angular_velocity(self):
        return self.bob_body.angular_velocity

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

    def compute_reward(self):
        upright_bonus = np.cos(self.angle())  # 1 when angle = 0 (upright)
        position_penalty = -abs(self.cart_x()) * 0.1
        velocity_penalty = -0.01 * (
            abs(self.angular_velocity()) + abs(self.cart_velocity_x())
        )
        return upright_bonus + position_penalty + velocity_penalty + 1.0
