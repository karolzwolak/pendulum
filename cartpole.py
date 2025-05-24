import numpy as np
from satellite_joint import SatelliteJoint
from simulation import Simulation
import pymunk

import simulation


class CartPoleSimulation(Simulation):
    def __init__(
        self,
        arm_length=15,
        satellite_mass=0.1,
        satellite_radius=2,
        cart_mass=1,
        initial_angle=0,  # bottom
    ):
        super().__init__(cart_mass=cart_mass)
        self.obs_size = 4  # [angle, angular_velocity, cart_x, cart_velocity_x]
        self.initial_angle = initial_angle

        self.satellite_body = pymunk.Body(
            satellite_mass,
            pymunk.moment_for_circle(satellite_mass, 0, satellite_radius),
        )
        self.satellite_shape = pymunk.Circle(self.satellite_body, satellite_radius)

        self.joint = SatelliteJoint(
            self.cart_body, self.satellite_body, arm_length, initial_angle
        )

        self.space.add(self.satellite_body, self.satellite_shape)
        self.joint.add_to_space(self.space)
        self.reset()

    def step(self, force=0):
        self.joint.step()
        return super().step(force)

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

    @staticmethod
    def shaped_upright_reward(upright: float) -> float:
        """
        Maps upright ∈ [-1, 1] to a reward ∈ [0, 10].
        Quadratic shape: 10 when upright, 0 when hanging.
        """
        # Map -1..1 → 0..1, then square for curvature
        normalized = (upright + 1) / 2  # -1 → 0, 0 → 0.5, 1 → 1
        reward = 10 * (normalized**2)  # steeper near upright
        return reward

    def compute_reward(self):
        if self.is_out_of_bounds():
            return -10 * self.max_steps
        upright_bonus = self.shaped_upright_reward(self.joint.upright())
        position_penalty = -abs(self.cart_x()) / simulation.WORLD_SIZE
        velocity_penalty = -0.004 * (
            abs(self.angular_velocity()) + abs(self.cart_velocity_x())
        )
        return upright_bonus + position_penalty + velocity_penalty
