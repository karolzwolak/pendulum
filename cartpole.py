import numpy as np
import math
from satellite_joint import SatelliteJoint
from simulation import Simulation
import pymunk

import simulation


class CartPoleSimulation(Simulation):
    def __init__(
        self,
        arm_length=15,
        weigth_mass=0.1,
        cart_mass=1,
        initial_angle=math.pi,  # top
    ):
        super().__init__(cart_mass=cart_mass)
        self.obs_size = 4  # [angle, angular_velocity, cart_x, cart_velocity_x]
        self.initial_angle = initial_angle

        # Create pendulum bob (dynamic body)
        self.bob_body = pymunk.Body()

        self.bob_shape = pymunk.Circle(self.bob_body, weigth_mass)
        self.bob_shape.mass = weigth_mass

        self.joint = SatelliteJoint(
            self.cart_body, self.bob_body, arm_length, initial_angle
        )

        self.space.add(self.bob_body, self.bob_shape)
        self.joint.add_to_space(self.space)
        self.reset()

    def reset(self, angle=None):
        if angle is None:
            angle = self.initial_angle
        super().reset()
        self.joint.reset(angle)

    def angle(self):
        return self.joint.relative_angle()

    def angular_velocity(self):
        return self.joint.relative_angular_velocity()

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
        angle = self.angle()
        upright_bonus = -np.cos(angle)  # 1 when angle = 0 (upright)
        position_penalty = -abs(self.cart_x()) / simulation.WORLD_SIZE
        velocity_penalty = -0.01 * (
            abs(self.angular_velocity()) + abs(self.cart_velocity_x())
        )
        upright_bonus *= 10
        return upright_bonus + position_penalty + velocity_penalty + 1.0
