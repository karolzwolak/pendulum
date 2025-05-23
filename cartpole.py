import numpy as np
import math
from simulation import Simulation
import pymunk

import simulation


class CartPoleSimulation(Simulation):
    def __init__(
        self,
        arm_length=15,
        weigth_mass=0.1,
        cart_mass=1,
    ):
        super().__init__(cart_mass=cart_mass)
        self.obs_size = 4  # [angle, angular_velocity, cart_x, cart_velocity_x]

        self.arm_length = arm_length

        # Create pendulum bob (dynamic body)
        self.bob_body = pymunk.Body()
        self.reset()

        self.arm_length = arm_length
        self.bob_shape = pymunk.Circle(self.bob_body, weigth_mass)
        self.bob_shape.mass = weigth_mass

        self.joint = pymunk.PinJoint(self.cart_body, self.bob_body, (0, 0), (0, 0))

        self.space.add(self.bob_body, self.bob_shape, self.joint)

    def reset(self, angle=0):
        super().reset()
        self.bob_body.position = (
            self.cart_body.position
            + self.arm_length * pymunk.Vec2d(0, 1).rotated(angle)
        )
        self.bob_body.velocity = (0, 0)

    def angle(self):
        # Convert local anchors to world coordinates
        world_anchor_a = self.cart_body.local_to_world(self.joint.anchor_a)
        world_anchor_b = self.bob_body.local_to_world(self.joint.anchor_b)

        # Vector from A to B
        dx = world_anchor_b.x - world_anchor_a.x
        dy = world_anchor_b.y - world_anchor_a.y

        angle = math.atan2(dy, dx)  # Angle in radians
        # 0 is at the bottem, pi is at the top
        angle = angle - math.pi / 2  # Shift to 0 at the top
        return angle

    def angular_velocity(self):
        # World-space positions
        pivot_pos = self.cart_body.position
        bob_pos = self.bob_body.position

        # Relative vector from pivot to bob
        r = bob_pos - pivot_pos

        # Bob's linear velocity at its center
        v = self.bob_body.velocity

        # Angular velocity = (r x v) / |r|²
        angular_velocity = (r.x * v.y - r.y * v.x) / r.get_length_sqrd()
        return angular_velocity

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
