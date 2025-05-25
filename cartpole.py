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
        # how often to randomize angle (in episodes)
        randomize_angle_frequency=0,
        max_steps=simulation.MAX_STEPS,
    ):
        super().__init__(cart_mass=cart_mass, max_steps=max_steps)
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

        if not math.isinf(self.max_steps):
            # scale so the ideal total reward is 100
            self.max_step_reward = self.reward(1, 0, 0, 0)
            self.max_step_reward *= self.max_steps / 100
        else:
            self.max_step_reward = 1

        self.total_reward = 0

        self.initial_angle = initial_angle
        self.angle_range = None
        self.randomize_angle_frequency = randomize_angle_frequency
        self.randomize_angle_counter = 1
        self.reset()

    def step(self, force=0):
        self.joint.step()
        res = super().step(force)
        self.total_reward += self.compute_reward()
        return res

    def update_angle_range(self):
        if self.randomize_angle_counter < self.randomize_angle_frequency:
            self.randomize_angle_counter += 1
            return
        # do not update if the model is performing poorly
        if self.total_reward <= 0:
            return
        performance = self.total_reward / self.max_steps  # (0, 1]
        # better performance means lower baseline (harder)
        baseline = 1 - performance
        baseline *= np.pi
        print("baseline upright:", -np.cos(baseline))
        range_diff = np.random.uniform(0.1, 0.5) / 2
        self.angle_range = (baseline - range_diff, baseline + range_diff)
        self.randomize_angle_counter = 1
        self.reroll_angle()
        print("updating random angle range to", self.angle_range)

    def reroll_angle(self):
        new_angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        rand_sign = np.random.choice([-1, 1])
        new_angle *= rand_sign
        self.initial_angle = new_angle

    def reset(self, angle=None):
        if self.randomize_angle_frequency > 0:
            self.update_angle_range()
        if angle is None:
            angle = self.initial_angle
        super().reset()
        self.joint.reset(angle)
        self.total_reward = 0

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

    @staticmethod
    def reward(upright, angular_velocity, cart_x, cart_velocity_x):
        """
        Computes the reward based on the upright position, angular velocity,
        cart position, and cart velocity.
        """
        upright_bonus = CartPoleSimulation.shaped_upright_reward(upright)
        position_penalty = -4 * ((abs(cart_x) / simulation.WORLD_SIZE) ** 2)
        velocity_penalty = -0.005 * abs(angular_velocity) - 0.002 * abs(cart_velocity_x)

        return upright_bonus + position_penalty + velocity_penalty

    def compute_reward(self):
        """Computes the reward for the current state.
        Scale to max reward of 1.
        Additional penalty for going out-of-bounds.
        """
        upright = self.joint.upright()
        reward = self.reward(
            upright,
            self.angular_velocity(),
            self.cart_x(),
            self.cart_velocity_x(),
        )
        return reward / self.max_step_reward
