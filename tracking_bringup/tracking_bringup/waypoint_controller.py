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


class WaypointController(Node):
    def __init__(self):
        super().__init__("waypoint_controller")

        self.declare_parameter("pose_topic", "/apriltag_pose_0")
        self.declare_parameter("cmd_vel_topic", "/model/turtlebot3_burger_apriltag_000/cmd_vel")
        self.declare_parameter("waypoints", [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])


        self.declare_parameter("num_loops", 1)

        self.declare_parameter("linear_speed", 0.15)
        self.declare_parameter("max_angular_speed", 0.25)
        self.declare_parameter("k_yaw", 1.5)

        self.declare_parameter("distance_tolerance", 0.08)
        self.declare_parameter("turn_tolerance_rad", 0.05)

        self.declare_parameter("pose_timeout", 1.5)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("log_every_n", 30)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        flat_wps = list(self.get_parameter("waypoints").value)
        self.num_loops = max(1, int(self.get_parameter("num_loops").value))

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.k_yaw = float(self.get_parameter("k_yaw").value)
        self.distance_tolerance = float(self.get_parameter("distance_tolerance").value)
        self.turn_tolerance_rad = float(self.get_parameter("turn_tolerance_rad").value)
        self.pose_timeout = float(self.get_parameter("pose_timeout").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.log_every_n = int(self.get_parameter("log_every_n").value)

        if len(flat_wps) < 4 or len(flat_wps) % 2 != 0:
            raise ValueError(
                f"waypoints must be a flat [x0,y0,x1,y1,...] with even length >= 4, "
                f"got {len(flat_wps)} values"
            )
        self.waypoints: List[Tuple[float, float]] = [
            (float(flat_wps[i]), float(flat_wps[i + 1]))
            for i in range(0, len(flat_wps), 2)
        ]

        self.current_wp_idx = 1
        self.current_loop = 1
        self.phase = "TURN"

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

        self.get_logger().info(f"Waypoint controller started")
        self.get_logger().info(f"Pose:    {self.pose_topic}")
        self.get_logger().info(f"Cmd_vel: {self.cmd_vel_topic}")
        self.get_logger().info(f"Loops to run: {self.num_loops}")
        self.get_logger().info(f"Waypoints ({len(self.waypoints)}):")
        for i, (wx, wy) in enumerate(self.waypoints):
            marker = " <-- START" if i == 0 else ""
            self.get_logger().info(f"  [{i}] ({wx:.2f}, {wy:.2f}){marker}")

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
            if self.current_loop < self.num_loops:
                self.current_loop += 1
                self.current_wp_idx = 1
                self.phase = "TURN"
                self.get_logger().info(
                    f"=== Loop {self.current_loop - 1}/{self.num_loops} complete. "
                    f"Starting loop {self.current_loop}/{self.num_loops} ==="
                )
            else:
                self._stop()
                self.finished = True
                self.get_logger().info(
                    f"All {self.num_loops} loop(s) complete! Trajectory finished."
                )
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
                self.get_logger().info(
                    f"L{self.current_loop} WP {self.current_wp_idx}: turn complete, "
                    f"driving toward ({wx:.2f}, {wy:.2f}), dist={dist:.3f}"
                )
            else:
                w = max(-self.max_angular_speed,
                        min(self.max_angular_speed, self.k_yaw * yaw_err))
                cmd.angular.z = w
        else:
            if dist <= self.distance_tolerance:
                self._stop()
                self.get_logger().info(
                    f"L{self.current_loop} WP {self.current_wp_idx} reached "
                    f"at ({wx:.2f}, {wy:.2f}). Final dist={dist:.3f}"
                )
                self.current_wp_idx += 1
                self.phase = "TURN"
                return
            cmd.linear.x = self.linear_speed
            cmd.angular.z = max(-self.max_angular_speed,
                                min(self.max_angular_speed, self.k_yaw * yaw_err))

        self.cmd_pub.publish(cmd)

        if self.tick % max(1, self.log_every_n) == 1:
            self.get_logger().info(
                f"[{self.phase}] L{self.current_loop}/{self.num_loops} "
                f"wp={self.current_wp_idx}/{len(self.waypoints) - 1} "
                f"pos=({self.cur_x:.2f},{self.cur_y:.2f}) "
                f"target=({wx:.2f},{wy:.2f}) dist={dist:.2f} "
                f"yaw_err={yaw_err:.2f} v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()