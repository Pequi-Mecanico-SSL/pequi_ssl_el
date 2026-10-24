"""Wheel geometry of the 4-omniwheel base, shared by kinematics and odometry.

Conventions (REP 103): x forward, y left, angular.z positive = CCW from top.
Positive wheel speed = clockwise seen from outside the robot.
Wheel angle = position around the robot, CCW from the front.
"""
import math

# motor order: 0 front-right, 1 front-left, 2 back-right, 3 back-left
DEFAULT_WHEEL_ANGLES_DEG = [-60.0, 60.0, -135.0, 135.0]
DEFAULT_ROBOT_RADIUS = 0.077    # m, center to wheel contact
DEFAULT_WHEEL_RADIUS = 0.0245   # m


def wheel_matrix(angles_deg, robot_radius, wheel_radius):
    """Rows mapping body velocity (vx, vy, w) to wheel speed (rad/s).

    wheel drive direction = tangent (-sin, cos) at its position angle
    """
    return [[-math.sin(math.radians(a)) / wheel_radius,
             math.cos(math.radians(a)) / wheel_radius,
             robot_radius / wheel_radius] for a in angles_deg]
