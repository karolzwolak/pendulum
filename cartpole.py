import numpy as np
from satellite_joint import SatelliteJoint
from simulation import Simulation
import pymunk
import math

import simulation


class CartPoleSimulation(Simulation):
    def __init__(
        self,
        arm_length=20,
        satellite_mass=0.1,
        satellite_radius=2,
        cart_mass=1,
        initial_angle=0,
        max_steps=simulation.MAX_STEPS,
    ):
        super().__init__(cart_mass=cart_mass, max_steps=max_steps)
        self.obs_size = 4  # [angle, angular_velocity, cart_x, cart_velocity_x]
        self.initial_angle = initial_angle

        self.mid_body = pymunk.Body(
            satellite_mass,
            pymunk.moment_for_circle(satellite_mass, 0, satellite_radius),
        )
        self.mid_shape = pymunk.Circle(self.mid_body, satellite_radius)

        self.mid_joint = SatelliteJoint(
            self.cart_body, self.mid_body, arm_length, initial_angle
        )

        self.space.add(self.mid_body, self.mid_shape)
        self.mid_joint.add_to_space(self.space)

        self.tip_body = pymunk.Body(
            satellite_mass,
            pymunk.moment_for_circle(satellite_mass, 0, satellite_radius),
        )
        self.tip_shape = pymunk.Circle(self.tip_body, satellite_radius)

        self.tip_joint = SatelliteJoint(
            self.mid_body, self.tip_body, arm_length, initial_angle
        )

        self.space.add(self.tip_body, self.tip_shape)
        self.tip_joint.add_to_space(self.space)

        if not math.isinf(self.max_steps):
            # scale so the ideal total reward is 100
            self.max_step_reward = self.reward(1, 0)
            self.max_step_reward *= self.max_steps / 100
        else:
            self.max_step_reward = 1

        self.total_reward = 0

        self.initial_angle = initial_angle
        self.reset()

    def step(self, force=0):
        self.mid_joint.step()
        self.tip_joint.step()
        res = super().step(force)
        self.total_reward += self.compute_reward()
        return res

    def reset(self, angle=None):
        if angle is None:
            angle = self.initial_angle
        super().reset()
        self.mid_joint.reset(angle)
        self.tip_joint.reset(angle)
        self.total_reward = 0

    def angle(self):
        return self.mid_joint.relative_angle()

    def angular_velocity(self):
        return self.mid_joint.relative_angular_velocity()

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

    @staticmethod
    def reward(upright, cart_x, penalty_threshold=0.7):
        upright_bonus = CartPoleSimulation.shaped_upright_reward(upright)

        # we apply a penalty only if the pendulum is upright
        # we don't want to discourage exploration
        if upright_bonus < penalty_threshold:
            return upright_bonus

        position_penalty = -4 * ((abs(cart_x) / simulation.WORLD_SIZE) ** 2)

        return upright_bonus + position_penalty

    def compute_reward(self):
        upright = self.mid_joint.upright()
        reward = self.reward(
            upright,
            self.cart_x(),
        )
        return reward / self.max_step_reward
