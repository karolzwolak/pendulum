import numpy as np
from satellite_joint import SatelliteJoint
from simulation import Simulation
import pymunk
import math

import simulation


class CartPoleSimulation(Simulation):
    def __init__(
        self,
        arm_length=25,
        satellite_mass=0.0001,
        cart_mass=1,
        max_steps=simulation.MAX_STEPS,
        gravity_curriculum=(1, 981),
        damping_curriculum=(0.1, 0.8),
        initial_upright_upright=(-0.99, -0.99),
        initial_curriculum_progress=1,
    ):
        super().__init__(cart_mass=cart_mass, max_steps=max_steps)

        self.mid_body = pymunk.Body(
            satellite_mass,
            float("inf"),
        )

        self.mid_joint = SatelliteJoint(self.cart_body, self.mid_body, arm_length, 0)

        self.space.add(self.mid_body)
        self.mid_joint.add_to_space(self.space)

        self.tip_body = pymunk.Body(
            satellite_mass,
            float("inf"),
        )

        self.tip_joint = SatelliteJoint(self.mid_body, self.tip_body, arm_length, 0)

        self.space.add(self.tip_body)
        self.tip_joint.add_to_space(self.space)

        if not math.isinf(self.max_steps):
            # scale so the ideal total reward is 100
            self.max_step_reward = self.reward(1, 1, 0)
            self.max_step_reward *= self.max_steps / 100
        else:
            self.max_step_reward = 1

        self.total_reward = 0

        self.gravity_curriculum = gravity_curriculum
        self.damping_curriculum = damping_curriculum
        self.initial_upright_curriculum = initial_upright_upright
        self.progress_curriculum(initial_curriculum_progress)

        self.obs_size = np.size(self.state())
        self.reset()

    @staticmethod
    def interpolate_curriculum(start, end, t):
        """
        Interpolates between start and end values based on t (0 to 1).
        """
        # linear interpolation
        return start + (end - start) * t

    def progress_curriculum(self, t):
        gravity = self.interpolate_curriculum(
            self.gravity_curriculum[0], self.gravity_curriculum[1], t
        )
        self.space.gravity = (0, gravity)
        damping = self.interpolate_curriculum(
            self.damping_curriculum[0], self.damping_curriculum[1], t
        )
        self.space.damping = damping
        self.initial_upright = self.interpolate_curriculum(
            self.initial_upright_curriculum[0],
            self.initial_upright_curriculum[1],
            t,
        )

    def step(self, force=0):
        self.mid_joint.step()
        self.tip_joint.step()
        res = super().step(force)
        self.total_reward += self.compute_reward()
        return res

    def reset(self, angle=None):
        if angle is None:
            sign = 1 if np.random.rand() < 0.5 else -1
            angle = sign * np.arccos(-self.initial_upright)
        super().reset()
        self.mid_joint.reset(angle)
        self.tip_joint.reset(angle)
        self.total_reward = 0

    def state(self):
        mid_angle = self.mid_joint.relative_angle()
        tip_angle = self.tip_joint.relative_angle()
        mid_angular_velocity = self.mid_joint.relative_angular_velocity()
        tip_angular_velocity = self.tip_joint.relative_angular_velocity()
        return np.array(
            [
                self.cart_x() / simulation.WORLD_SIZE,
                self.cart_velocity_x() / simulation.WORLD_SIZE,
                np.sin(mid_angle),
                np.cos(mid_angle),
                np.sin(tip_angle),
                np.cos(tip_angle),
                mid_angular_velocity,
                tip_angular_velocity,
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
        reward = 10 * (normalized**3)  # steeper near upright
        return reward

    @staticmethod
    def reward(upright, cart_x, penalty_threshold=0.9):
        upright_bonus = CartPoleSimulation.shaped_upright_reward(upright)

        # we apply a penalty only if the pendulum is upright
        # we don't want to discourage exploration
        if upright < penalty_threshold:
            return upright_bonus

        position_penalty = -4 * ((abs(cart_x) / simulation.WORLD_SIZE) ** 2)

        return upright_bonus + position_penalty

    def upright(self):
        return -self.tip_body.position.y / 2 / self.tip_joint.length

    def compute_reward(self):
        upright_reward = self.reward(
            self.upright(),
            self.cart_x(),
        )
        # penalty for cart velocity to encourage stability
        velocity_penalty = -0.1 * abs(self.cart_velocity_x()) / simulation.WORLD_SIZE
        total_reward = upright_reward + velocity_penalty
        return total_reward / self.max_step_reward
