import math
from typing import Optional

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


class CircleController(Node):
    def __init__(self):
        super().__init__("circle_controller")

        self.declare_parameter("pose_topic", "/stag_pose_5")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

       
        self.declare_parameter("cx", 0.0)
        self.declare_parameter("cy", 0.0)
        self.declare_parameter("radius", 1.0)
        self.declare_parameter("direction", 1)


        self.declare_parameter("linear_speed", 0.12) 
        self.declare_parameter("max_angular_speed", 0.6)
        self.declare_parameter("k_yaw", 1.8) 

        self.declare_parameter("lookahead_arc", 0.20)

        self.declare_parameter("num_laps", 2.0)

        self.declare_parameter("pose_timeout", 1.5)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("log_every_n", 20)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)
        self.radius = float(self.get_parameter("radius").value)
        self.direction = int(self.get_parameter("direction").value)
        if self.direction not in (-1, 1):
            self.direction = 1
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.k_yaw = float(self.get_parameter("k_yaw").value)
        self.lookahead_arc = float(self.get_parameter("lookahead_arc").value)
        self.num_laps = float(self.get_parameter("num_laps").value)
        self.pose_timeout = float(self.get_parameter("pose_timeout").value)
        self.rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.log_every_n = int(self.get_parameter("log_every_n").value)


        self.cur_x: Optional[float] = None
        self.cur_y: Optional[float] = None
        self.cur_yaw: Optional[float] = None
        self.last_pose_time = None


        self.prev_theta: Optional[float] = None
        self.total_unwrap = 0.0
        self.finished = False


        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, self.pose_topic, self.pose_cb, 10
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self.step)
        self.tick = 0

        self.get_logger().info("Circle controller started")
        self.get_logger().info(f"  pose:    {self.pose_topic}")
        self.get_logger().info(f"  cmd_vel: {self.cmd_vel_topic}")
        self.get_logger().info(
            f"  circle: center=({self.cx:.2f}, {self.cy:.2f}), "
            f"radius={self.radius:.2f} m, dir={'+CCW' if self.direction>0 else '-CW'}"
        )
        self.get_logger().info(
            f"  motion: v={self.linear_speed:.2f} m/s, "
            f"w_max={self.max_angular_speed:.2f} rad/s, "
            f"lookahead={self.lookahead_arc:.2f} m"
        )
        self.get_logger().info(f"  stop after {self.num_laps:.2f} laps")

    def pose_cb(self, msg: PoseStamped):
        self.cur_x = float(msg.pose.position.x)
        self.cur_y = float(msg.pose.position.y)
        self.cur_yaw = yaw_from_pose(msg)
        self.last_pose_time = self.get_clock().now()

    def _have_fresh_pose(self) -> bool:
        if self.cur_x is None or self.last_pose_time is None:
            return False
        age = (self.get_clock().now() - self.last_pose_time).nanoseconds * 1e-9
        return age <= self.pose_timeout

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def step(self):
        self.tick += 1

        if self.finished:
            self._stop()
            return

        if not self._have_fresh_pose():
            self._stop()
            if self.tick % max(1, self.log_every_n) == 1:
                self.get_logger().warn("No fresh pose, stopping")
            return


        dx = self.cur_x - self.cx
        dy = self.cur_y - self.cy
        theta = math.atan2(dy, dx)

    
        if self.prev_theta is None:
            self.prev_theta = theta
        else:
            dtheta = wrap_to_pi(theta - self.prev_theta)
            self.total_unwrap += dtheta
            self.prev_theta = theta

 
        laps_done = self.direction * self.total_unwrap / (2.0 * math.pi)
        if laps_done >= self.num_laps:
            self._stop()
            self.finished = True
            self.get_logger().info(
                f"Done: {laps_done:.3f} laps. Holding zero cmd_vel."
            )
            return


        dtheta_ahead = (self.lookahead_arc / max(self.radius, 1e-3)) * self.direction
        theta_target = theta + dtheta_ahead
        tx = self.cx + self.radius * math.cos(theta_target)
        ty = self.cy + self.radius * math.sin(theta_target)

        target_heading = math.atan2(ty - self.cur_y, tx - self.cur_x)
        yaw_err = wrap_to_pi(target_heading - self.cur_yaw)

        cmd = Twist()
        cmd.linear.x = self.linear_speed
        cmd.angular.z = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, self.k_yaw * yaw_err),
        )
        self.cmd_pub.publish(cmd)

        if self.tick % max(1, self.log_every_n) == 1:
            r_cur = math.hypot(dx, dy)
            self.get_logger().info(
                f"laps={laps_done:.2f}/{self.num_laps:.2f} "
                f"pos=({self.cur_x:+.2f},{self.cur_y:+.2f}) "
                f"r={r_cur:.2f}/{self.radius:.2f} "
                f"theta={math.degrees(theta):+.0f}deg "
                f"yaw_err={math.degrees(yaw_err):+.0f}deg "
                f"v={cmd.linear.x:.2f} w={cmd.angular.z:+.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = CircleController()
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