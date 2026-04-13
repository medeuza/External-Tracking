import math
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def yaw_from_quat(q) -> float:
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return wrap_to_pi(yaw)


class PID:
    def __init__(self, kp: float, ki: float, kd: float, i_limit: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = abs(i_limit)

        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 1e-6:
            return self.kp * error

        self.integral += error * dt
        self.integral = clamp(self.integral, -self.i_limit, self.i_limit)

        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt

        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class SquarePosePIDController(Node):
    def __init__(self):
        super().__init__("square_pose_pid_controller")

        # ---------------- source ----------------
        self.declare_parameter("pose_source", "odom")  # odom or aruco
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("aruco_topic", "/aruco_pose")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        # ---------------- trajectory ----------------
        self.declare_parameter("side_length", 0.30)
        self.declare_parameter("num_sides", 4)
        self.declare_parameter("turn_angle_deg", 90.0)
        self.declare_parameter("control_rate", 20.0)

        # ---------------- straight motion ----------------
        self.declare_parameter("linear_speed", 0.08)
        self.declare_parameter("max_linear_speed", 0.10)
        self.declare_parameter("max_angular_speed", 1.0)

        self.declare_parameter("distance_tolerance", 0.01)
        self.declare_parameter("yaw_stop_threshold", 0.35)
        self.declare_parameter("slowdown_distance", 0.05)

        # ---------------- turn motion ----------------
        self.declare_parameter("turn_tolerance_deg", 2.0)

        # ---------------- pose validity ----------------
        self.declare_parameter("pose_timeout", 0.30)

        # ---------------- PID ----------------
        # straight: держим нужный yaw
        self.declare_parameter("drive_kp", 2.5)
        self.declare_parameter("drive_ki", 0.0)
        self.declare_parameter("drive_kd", 0.10)

        # turn: крутимся до нужного угла
        self.declare_parameter("turn_kp", 3.0)
        self.declare_parameter("turn_ki", 0.0)
        self.declare_parameter("turn_kd", 0.12)

        self.pose_source = str(self.get_parameter("pose_source").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.aruco_topic = str(self.get_parameter("aruco_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)

        self.side_length = float(self.get_parameter("side_length").value)
        self.num_sides = int(self.get_parameter("num_sides").value)
        self.turn_angle = math.radians(float(self.get_parameter("turn_angle_deg").value))
        self.control_rate = float(self.get_parameter("control_rate").value)

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)

        self.distance_tolerance = float(self.get_parameter("distance_tolerance").value)
        self.yaw_stop_threshold = float(self.get_parameter("yaw_stop_threshold").value)
        self.slowdown_distance = float(self.get_parameter("slowdown_distance").value)

        self.turn_tolerance = math.radians(float(self.get_parameter("turn_tolerance_deg").value))
        self.pose_timeout = float(self.get_parameter("pose_timeout").value)

        self.drive_pid = PID(
            float(self.get_parameter("drive_kp").value),
            float(self.get_parameter("drive_ki").value),
            float(self.get_parameter("drive_kd").value),
            i_limit=0.5,
        )

        self.turn_pid = PID(
            float(self.get_parameter("turn_kp").value),
            float(self.get_parameter("turn_ki").value),
            float(self.get_parameter("turn_kd").value),
            i_limit=0.5,
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.aruco_sub = self.create_subscription(PoseStamped, self.aruco_topic, self.aruco_callback, 10)

        self.timer = self.create_timer(1.0 / self.control_rate, self.control_callback)

        # current pose
        self.have_pose = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_pose_time = None

        # state machine
        self.started = False
        self.finished = False
        self.phase = "WAIT_START"  # WAIT_START, DRIVE, TURN, FINISHED

        self.side_idx = 0

        # current segment references
        self.segment_start_x = None
        self.segment_start_y = None
        self.segment_yaw_ref = None

        self.turn_start_yaw = None
        self.turn_target_yaw = None

        self.prev_control_time = self.get_clock().now()
        self.last_debug_time = self.get_clock().now()

        self.get_logger().info(
            f"Square controller started | source={self.pose_source}, side={self.side_length:.3f} m"
        )

    # ---------------- pose callbacks ----------------
    def odom_callback(self, msg: Odometry):
        if self.pose_source != "odom":
            return

        pose = msg.pose.pose
        self.x = float(pose.position.x)
        self.y = float(pose.position.y)
        self.yaw = yaw_from_quat(pose.orientation)
        self.have_pose = True
        self.last_pose_time = self.get_clock().now()

    def aruco_callback(self, msg: PoseStamped):
        if self.pose_source != "aruco":
            return

        self.x = float(msg.pose.position.x)
        self.y = float(msg.pose.position.y)

        q = msg.pose.orientation
        self.yaw = yaw_from_quat(q)

        self.have_pose = True
        self.last_pose_time = self.get_clock().now()

    # ---------------- helpers ----------------
    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def start_drive_phase(self):
        self.phase = "DRIVE"
        self.segment_start_x = self.x
        self.segment_start_y = self.y
        self.segment_yaw_ref = self.yaw
        self.drive_pid.reset()

        self.get_logger().info(
            f"Start DRIVE side={self.side_idx + 1}/{self.num_sides} | "
            f"start=({self.segment_start_x:.3f},{self.segment_start_y:.3f}) "
            f"yaw_ref={self.segment_yaw_ref:.3f}"
        )

    def start_turn_phase(self):
        self.phase = "TURN"
        self.turn_start_yaw = self.yaw
        self.turn_target_yaw = wrap_to_pi(self.turn_start_yaw + self.turn_angle)
        self.turn_pid.reset()

        self.get_logger().info(
            f"Start TURN side={self.side_idx + 1}/{self.num_sides} | "
            f"yaw_start={self.turn_start_yaw:.3f} yaw_target={self.turn_target_yaw:.3f}"
        )

    # ---------------- main control ----------------
    def control_callback(self):
        now = self.get_clock().now()
        dt = (now - self.prev_control_time).nanoseconds / 1e9
        self.prev_control_time = now

        if self.finished:
            self.stop_robot()
            return

        if not self.have_pose:
            self.stop_robot()
            return

        if self.last_pose_time is not None:
            pose_age = (now - self.last_pose_time).nanoseconds / 1e9
            if pose_age > self.pose_timeout:
                self.get_logger().warn(
                    f"Pose timeout: {pose_age:.3f} s -> stop"
                )
                self.stop_robot()
                return

        if not self.started:
            self.started = True
            self.side_idx = 0
            self.start_drive_phase()
            return

        cmd = Twist()

        if self.phase == "DRIVE":
            dx = self.x - self.segment_start_x
            dy = self.y - self.segment_start_y
            traveled = math.sqrt(dx * dx + dy * dy)

            remaining = max(0.0, self.side_length - traveled)
            yaw_err = wrap_to_pi(self.segment_yaw_ref - self.yaw)

            # reached current side end
            if remaining <= self.distance_tolerance:
                self.stop_robot()
                self.get_logger().info(
                    f"Finished side {self.side_idx + 1}/{self.num_sides} | "
                    f"traveled={traveled:.3f} m"
                )

                if self.side_idx >= self.num_sides - 1:
                    self.finished = True
                    self.phase = "FINISHED"
                    self.stop_robot()
                    self.get_logger().info("Square completed")
                    return

                self.start_turn_phase()
                return

            # slowdown near end of side
            v = self.linear_speed
            if remaining < self.slowdown_distance:
                v = self.linear_speed * (remaining / self.slowdown_distance)
                v = max(0.03, v)

            # if robot смотрит сильно не туда — не едем вперёд
            if abs(yaw_err) > self.yaw_stop_threshold:
                v = 0.0

            w = self.drive_pid.update(yaw_err, dt)

            cmd.linear.x = clamp(v, 0.0, self.max_linear_speed)
            cmd.angular.z = clamp(w, -self.max_angular_speed, self.max_angular_speed)
            self.cmd_pub.publish(cmd)

            dbg_dt = (now - self.last_debug_time).nanoseconds / 1e9
            if dbg_dt > 1.0:
                self.get_logger().info(
                    f"[DRIVE] side={self.side_idx + 1}/{self.num_sides} "
                    f"x={self.x:.3f} y={self.y:.3f} yaw={self.yaw:.3f} "
                    f"traveled={traveled:.3f} remaining={remaining:.3f} "
                    f"yaw_err={yaw_err:.3f} v={cmd.linear.x:.3f} w={cmd.angular.z:.3f}"
                )
                self.last_debug_time = now

        elif self.phase == "TURN":
            yaw_err = wrap_to_pi(self.turn_target_yaw - self.yaw)

            if abs(yaw_err) <= self.turn_tolerance:
                self.stop_robot()
                self.side_idx += 1
                self.get_logger().info(
                    f"Finished turn -> next side {self.side_idx + 1}/{self.num_sides}"
                )
                self.start_drive_phase()
                return

            w = self.turn_pid.update(yaw_err, dt)

            cmd.linear.x = 0.0
            cmd.angular.z = clamp(w, -self.max_angular_speed, self.max_angular_speed)
            self.cmd_pub.publish(cmd)

            dbg_dt = (now - self.last_debug_time).nanoseconds / 1e9
            if dbg_dt > 1.0:
                turned = wrap_to_pi(self.yaw - self.turn_start_yaw)
                self.get_logger().info(
                    f"[TURN] side={self.side_idx + 1}/{self.num_sides} "
                    f"yaw={self.yaw:.3f} turned={turned:.3f} "
                    f"target_delta={self.turn_angle:.3f} yaw_err={yaw_err:.3f} "
                    f"w={cmd.angular.z:.3f}"
                )
                self.last_debug_time = now

        else:
            self.stop_robot()


def main(args=None):
    rclpy.init(args=args)
    node = SquarePosePIDController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_robot()
        except Exception:
            pass

        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()