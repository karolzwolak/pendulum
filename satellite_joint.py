import pymunk
from pymunk.vec2d import Vec2d
import numpy as np


class SatelliteJoint:
    def __init__(
        self,
        primary: pymunk.Body,
        satellite: pymunk.Body,
        length: float,
        initial_angle: float = 0.0,
        anchor_point_primary=(0, 0),
        anchor_point_satellite=(0, 0),
    ):
        """
        Initialize the satellite joint.

        Parameters:
            primary: pymunk.Body - The fixed or main body.
            satellite: pymunk.Body - The orbiting or attached body.
            length: float - Distance between primary and satellite.
            initial_angle: float - Initial angle (radians), 0 is directly below.
            anchor_point_primary: tuple - Anchor on primary body.
            anchor_point_satellite: tuple - Anchor on satellite body.
        """
        self.primary = primary
        self.satellite = satellite
        self.length = length

        self.reset(initial_angle)
        # Create the pin joint (enforces distance constraint)
        self.joint = pymunk.PinJoint(
            primary, satellite, anchor_point_primary, anchor_point_satellite
        )

    def add_to_space(self, space: pymunk.Space):
        space.add(self.joint)

    def remove_from_space(self, space: pymunk.Space):
        space.remove(self.joint)

    def relative_angle(self) -> float:
        """
        Get the angle from primary to satellite, where:
            0   => directly below
            π   => directly above
            π/2 => right
            3π/2 => left
        Returns:
            Angle in radians ∈ [0, 2π).
        """
        delta = self.satellite.position - self.primary.position
        angle = np.atan2(delta.x, delta.y) % (2 * np.pi)
        return angle

    def relative_angular_velocity(self) -> float:
        """
        Calculate the angular velocity of the satellite relative to the primary.
        Returns:
            Angular velocity in radians per second.
        """
        r = self.satellite.position - self.primary.position
        v = self.satellite.velocity - self.primary.velocity

        # Angular velocity = (r x v) / |r|^2
        r_squared = r.get_length_sqrd()
        if r_squared == 0:
            return 0.0
        return r.cross(v) / r_squared

    def reset(self, angle: float = 0.0):
        """
        Reposition the satellite at the specified angle.
        0 = directly below, π = directly above.
        """
        self.satellite.position = self.primary.position + Vec2d(0, self.length).rotated(
            -angle
        )

        # Zero velocity for clean reset
        self.satellite.velocity = Vec2d(0, 0)
        self.satellite.angular_velocity = 0

    def upright(self):
        return -np.cos(self.relative_angle())
