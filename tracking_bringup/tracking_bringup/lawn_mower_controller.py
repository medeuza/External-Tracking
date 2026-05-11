"""
Lawn-mower trajectory controller.

Drives a single robot through a configurable sequence of waypoints in
the world frame, using visual pose feedback (e.g. /apriltag_pose_<id>)
and publishing velocity commands to a configurable /cmd_vel topic.

The lawn-mower pattern is computed from four parameters:
    num_lanes      Number of straight passes (e.g. 3 for a snake with 3 sweeps)
    lane_length    Length of each pass, meters
    lane_spacing   Sideways distance between consecutive passes, meters
    primary_axis   "x" (passes along x, step along y)  or
                   "y" (passes along y, step along x)

For each waypoint the controller alternates DRIVE (go to waypoint) and
TURN (rotate to face the next waypoint). Both phases use simple
proportional control with deadbands.

Each robot is one instance of this node, configured with its own
pose_topic, cmd_vel_topic, and start_xy.
"""

import math
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from tf_transformations import euler_from_quaternion


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_pose(msg: PoseStamped) -> float:
    q = msg.pose.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return wrap_to_pi(yaw)


class LawnMowerController(Node):
    def __init__(self):
        super().__init__("lawn_mower_controller")

        # Topics — set per robot.
        self.declare_parameter("pose_topic", "/apriltag_pose_0")
        self.declare_parameter("cmd_vel_topic", "/model/turtlebot3_burger_apriltag_000/cmd_vel")

        # Robot start pose in world frame. Used as the (0,0) anchor of
        # the lawn-mower pattern, the actual robot position is read from
        # pose_topic.
        self.declare_parameter("start_x", -4.0)
        self.declare_parameter("start_y", 0.0)

        # Lawn-mower pattern parameters.
        self.declare_parameter("num_lanes", 3)
        self.declare_parameter("lane_length", 5.0)
        self.declare_parameter("lane_spacing", 1.0)
        self.declare_parameter("primary_axis", "x")  # "x" or "y"

        # Control gains and limits.
        self.declare_parameter("linear_speed", 0.12)
        self.declare_parameter("max_angular_speed", 0.2)
        self.declare_parameter("k_yaw", 1.5)
        self.declare_parameter("k_xtrack", 1.5)

        self.declare_parameter("distance_tolerance", 0.05)
        self.declare_parameter("turn_tolerance_rad", 0.05)

        self.declare_parameter("pose_timeout", 1.5)
        self.declare_parameter("control_rate_hz", 20.0)

        self.declare_parameter("log_every_n", 20)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)

        self.start_x = float(self.get_parameter("start_x").value)
        self.start_y = float(self.get_parameter("start_y").value)

        self.num_lanes = int(self.get_parameter("num_lanes").value)
        self.lane_length = float(self.get_parameter("lane_length").value)
        self.lane_spacing = float(self.get_parameter("lane_spacing").value)
        self.primary_axis = str(self.get_parameter("primary_axis").value).lower()

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.k_yaw = float(self.get_parameter("k_yaw").value)
        self.k_xtrack = float(self.get_parameter("k_xtrack").value)

        self.distance_tolerance = float(self.get_parameter("distance_tolerance").value)
        self.turn_tolerance_rad = float(self.get_parameter("turn_tolerance_rad").value)

        self.pose_timeout = float(self.get_parameter("pose_timeout").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.log_every_n = int(self.get_parameter("log_every_n").value)

        self.waypoints: List[Tuple[float, float]] = self._build_lawnmower()
        self.current_wp_idx = 0
        self.phase = "TURN"  # alternates "TURN" -> "DRIVE"

        self.cur_x: Optional[float] = None
        self.cur_y: Optional[float] = None
        self.cur_yaw: Optional[float] = None
        self.last_pose_time = None

        self.pose_sub = self.create_subscription(
            PoseStamped, self.pose_topic, self.pose_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self.control_step)
        self.tick = 0
        self.finished = False

        self.get_logger().info(f"Lawn-mower controller started")
        self.get_logger().info(f"Pose:    {self.pose_topic}")
        self.get_logger().info(f"Cmd_vel: {self.cmd_vel_topic}")
        self.get_logger().info(f"Start: ({self.start_x:.2f}, {self.start_y:.2f}), "
                               f"lanes={self.num_lanes}, len={self.lane_length}, "
                               f"spacing={self.lane_spacing}, axis={self.primary_axis}")
        self.get_logger().info(f"Waypoints ({len(self.waypoints)}):")
        for i, (wx, wy) in enumerate(self.waypoints):
            self.get_logger().info(f"  [{i}] ({wx:.2f}, {wy:.2f})")

    def _build_lawnmower(self) -> List[Tuple[float, float]]:
        """Generate the sequence of waypoint corners for the lawn-mower."""
        pts: List[Tuple[float, float]] = []
        for i in range(self.num_lanes):
            # End A and End B of the i-th lane, alternating direction.
            if i % 2 == 0:
                a_along = 0.0
                b_along = self.lane_length
            else:
                a_along = self.lane_length
                b_along = 0.0

            across = i * self.lane_spacing

            if self.primary_axis == "x":
                pa = (self.start_x + a_along, self.start_y + across)
                pb = (self.start_x + b_along, self.start_y + across)
            else:
                pa = (self.start_x + across, self.start_y + a_along)
                pb = (self.start_x + across, self.start_y + b_along)

            if i == 0:
                pts.append(pa)
            pts.append(pb)
            # After reaching B, step sideways to the next lane start.
            if i < self.num_lanes - 1:
                next_across = (i + 1) * self.lane_spacing
                if self.primary_axis == "x":
                    pts.append((pb[0], self.start_y + next_across))
                else:
                    pts.append((self.start_x + next_across, pb[1]))
        return pts

    def pose_callback(self, msg: PoseStamped):
        self.cur_x = float(msg.pose.position.x)
        self.cur_y = float(msg.pose.position.y)
        self.cur_yaw = yaw_from_pose(msg)
        self.last_pose_time = self.get_clock().now()

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _have_fresh_pose(self) -> bool:
        if self.cur_x is None or self.last_pose_time is None:
            return False
        age = (self.get_clock().now() - self.last_pose_time).nanoseconds * 1e-9
        return age <= self.pose_timeout

    def control_step(self):
        self.tick += 1

        if self.finished:
            self._stop()
            return

        if not self._have_fresh_pose():
            self._stop()
            if self.tick % max(1, self.log_every_n) == 1:
                self.get_logger().warn("No fresh pose, stopping")
            return

        if self.current_wp_idx >= len(self.waypoints):
            self._stop()
            self.finished = True
            self.get_logger().info("Lawn-mower complete!")
            return

        wx, wy = self.waypoints[self.current_wp_idx]
        dx = wx - self.cur_x
        dy = wy - self.cur_y
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw_err = wrap_to_pi(target_yaw - self.cur_yaw)

        cmd = Twist()

        if self.phase == "TURN":
            if abs(yaw_err) <= self.turn_tolerance_rad:
                self.phase = "DRIVE"
                if self.tick % max(1, self.log_every_n) == 1:
                    self.get_logger().info(
                        f"WP {self.current_wp_idx}: turn done, dist={dist:.3f}"
                    )
            else:
                w = max(-self.max_angular_speed,
                        min(self.max_angular_speed, self.k_yaw * yaw_err))
                cmd.angular.z = w
        else:  # DRIVE
            if dist <= self.distance_tolerance:
                self._stop()
                self.get_logger().info(
                    f"WP {self.current_wp_idx} reached: ({wx:.2f}, {wy:.2f})"
                )
                self.current_wp_idx += 1
                self.phase = "TURN"
                return
            cmd.linear.x = self.linear_speed
            # Small steering correction while driving.
            cmd.angular.z = max(-self.max_angular_speed,
                                min(self.max_angular_speed, self.k_yaw * yaw_err))

        self.cmd_pub.publish(cmd)

        if self.tick % max(1, self.log_every_n) == 1:
            self.get_logger().info(
                f"[{self.phase}] wp={self.current_wp_idx}/{len(self.waypoints)} "
                f"pos=({self.cur_x:.2f},{self.cur_y:.2f}) "
                f"target=({wx:.2f},{wy:.2f}) dist={dist:.3f} "
                f"yaw={self.cur_yaw:.2f} target_yaw={target_yaw:.2f} "
                f"v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = LawnMowerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())  # stop on exit
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()