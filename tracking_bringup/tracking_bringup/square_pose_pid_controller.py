import math
from typing import Optional, Tuple

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
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_limit = abs(float(i_limit))
        self.integral = 0.0
        self.prev_error: Optional[float] = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 1e-6:
            return self.kp * error

        self.integral += error * dt
        self.integral = clamp(self.integral, -self.i_limit, self.i_limit)

        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative


class SquarePosePIDController(Node):
    def __init__(self):
        super().__init__("square_pose_pid_controller")

        self.declare_parameter("pose_source", "apriltag")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("visual_topic", "/apriltag_pose")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.declare_parameter("side_length", 0.50)
        self.declare_parameter("num_sides", 4)
        self.declare_parameter("turn_angle_deg", 90.0)
        self.declare_parameter("control_rate", 30.0)
        self.declare_parameter("startup_wait_sec", 0.5)

        self.declare_parameter("linear_speed", 0.045)
        self.declare_parameter("max_linear_speed", 0.065)
        self.declare_parameter("max_angular_speed", 0.25)
        self.declare_parameter("min_turn_speed", 0.12)

        self.declare_parameter("distance_tolerance", 0.003)
        self.declare_parameter("drive_settle_cycles", 6)
        self.declare_parameter("turn_tolerance_deg", 3.0)
        self.declare_parameter("turn_settle_cycles", 3)
        self.declare_parameter("yaw_stop_threshold", 0.9)
        self.declare_parameter("slowdown_distance", 0.10)
        self.declare_parameter("pose_timeout", 1.5)

        self.declare_parameter("cross_track_gain", 0.3)
        self.declare_parameter("max_heading_correction_deg", 18.0)
        self.declare_parameter("segment_progress_guard", 0.03)

        self.declare_parameter("tag_offset_x", -0.032)
        self.declare_parameter("tag_offset_y", 0.0)

        self.declare_parameter("drive_kp", 0.8)
        self.declare_parameter("drive_ki", 0.0)
        self.declare_parameter("drive_kd", 0.20)
        self.declare_parameter("turn_kp", 2.0)
        self.declare_parameter("turn_ki", 0.0)
        self.declare_parameter("turn_kd", 0.25)

        self.pose_source = str(self.get_parameter("pose_source").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.visual_topic = str(self.get_parameter("visual_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)

        self.side_length = float(self.get_parameter("side_length").value)
        self.num_sides = int(self.get_parameter("num_sides").value)
        self.turn_angle = math.radians(float(self.get_parameter("turn_angle_deg").value))
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.startup_wait_sec = float(self.get_parameter("startup_wait_sec").value)

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.min_turn_speed = float(self.get_parameter("min_turn_speed").value)

        self.distance_tolerance = float(self.get_parameter("distance_tolerance").value)
        self.drive_settle_cycles = int(self.get_parameter("drive_settle_cycles").value)
        self.turn_tolerance = math.radians(float(self.get_parameter("turn_tolerance_deg").value))
        self.turn_settle_cycles = int(self.get_parameter("turn_settle_cycles").value)
        self.yaw_stop_threshold = float(self.get_parameter("yaw_stop_threshold").value)
        self.slowdown_distance = float(self.get_parameter("slowdown_distance").value)
        self.pose_timeout = float(self.get_parameter("pose_timeout").value)

        self.cross_track_gain = float(self.get_parameter("cross_track_gain").value)
        self.max_heading_correction = math.radians(
            float(self.get_parameter("max_heading_correction_deg").value)
        )
        self.segment_progress_guard = float(self.get_parameter("segment_progress_guard").value)

        self.tag_offset_x = float(self.get_parameter("tag_offset_x").value)
        self.tag_offset_y = float(self.get_parameter("tag_offset_y").value)

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
        self.visual_sub = self.create_subscription(
            PoseStamped, self.visual_topic, self.visual_callback, 10
        )

        self.timer = self.create_timer(1.0 / self.control_rate, self.control_callback)

        self.have_pose = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_pose_time = None

        self.started = False
        self.finished = False
        self.phase = "WAIT_START"
        self.start_wall_time = self.get_clock().now()

        self.side_idx = 0
        self.segment_start: Optional[Tuple[float, float]] = None
        self.segment_yaw_ref: Optional[float] = None
        self.turn_target_yaw: Optional[float] = None
        self.turn_start_yaw: Optional[float] = None

        self.drive_settle_count = 0
        self.turn_settle_count = 0

        self.prev_control_time = self.get_clock().now()
        self.last_debug_time = self.get_clock().now()

        self.get_logger().info(
            f"Square controller started | source={self.pose_source}, "
            f"side={self.side_length:.3f}, sides={self.num_sides}"
        )
        self.get_logger().info(
            f"AprilTag offset compensation: tag_offset_x={self.tag_offset_x:.3f}, "
            f"tag_offset_y={self.tag_offset_y:.3f}"
        )

    def odom_callback(self, msg: Odometry):
        if self.pose_source != "odom":
            return

        pose = msg.pose.pose
        self.x = float(pose.position.x)
        self.y = float(pose.position.y)
        self.yaw = yaw_from_quat(pose.orientation)
        self.have_pose = True
        self.last_pose_time = self.get_clock().now()

    def visual_callback(self, msg: PoseStamped):
        if self.pose_source not in ["visual", "aruco", "apriltag", "stag"]:
            return

        tag_x = float(msg.pose.position.x)
        tag_y = float(msg.pose.position.y)
        yaw = yaw_from_quat(msg.pose.orientation)

        c = math.cos(yaw)
        s = math.sin(yaw)

        base_x = tag_x - (c * self.tag_offset_x - s * self.tag_offset_y)
        base_y = tag_y - (s * self.tag_offset_x + c * self.tag_offset_y)

        self.x = base_x
        self.y = base_y
        self.yaw = yaw

        self.have_pose = True
        self.last_pose_time = self.get_clock().now()

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def start_drive_phase(self):
        self.phase = "DRIVE"
        self.segment_start = (self.x, self.y)

        if self.segment_yaw_ref is None:
            self.segment_yaw_ref = self.yaw

        self.drive_settle_count = 0
        self.drive_pid.reset()

        self.get_logger().info(
            f"Start DRIVE side={self.side_idx + 1}/{self.num_sides} | "
            f"start=({self.x:.3f},{self.y:.3f}) yaw_ref={self.segment_yaw_ref:.3f}"
        )

    def start_turn_phase(self):
        self.phase = "TURN"
        self.turn_start_yaw = self.yaw
        self.turn_target_yaw = wrap_to_pi(self.segment_yaw_ref + self.turn_angle)
        self.turn_settle_count = 0
        self.turn_pid.reset()

        self.get_logger().info(
            f"Start TURN side={self.side_idx + 1}/{self.num_sides} | "
            f"yaw={self.yaw:.3f} -> target={self.turn_target_yaw:.3f}"
        )

    def segment_errors(self) -> Tuple[float, float]:
        assert self.segment_start is not None
        assert self.segment_yaw_ref is not None

        sx, sy = self.segment_start
        dx = self.x - sx
        dy = self.y - sy

        ux = math.cos(self.segment_yaw_ref)
        uy = math.sin(self.segment_yaw_ref)

        progress = dx * ux + dy * uy
        cross_track_error = -dx * uy + dy * ux

        return progress, cross_track_error

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
                self.get_logger().warn(f"Pose timeout: {pose_age:.3f}s -> stop")
                self.stop_robot()
                return

        if not self.started:
            since_start = (now - self.start_wall_time).nanoseconds / 1e9
            if since_start < self.startup_wait_sec:
                self.stop_robot()
                return

            self.started = True
            self.side_idx = 0
            self.segment_yaw_ref = self.yaw
            self.start_drive_phase()
            return

        cmd = Twist()

        if self.phase == "DRIVE":
            progress, cte = self.segment_errors()
            remaining = max(0.0, self.side_length - progress)

            if progress >= self.side_length:
                self.drive_settle_count += 1
            else:
                self.drive_settle_count = 0

            if self.drive_settle_count >= self.drive_settle_cycles:
                self.stop_robot()

                self.get_logger().info(
                    f"Finished side {self.side_idx + 1}/{self.num_sides} | "
                    f"progress={progress:.3f} cte={cte:.3f} "
                    f"settle={self.drive_settle_count}/{self.drive_settle_cycles}"
                )

                if self.side_idx >= self.num_sides - 1:
                    self.finished = True
                    self.phase = "FINISHED"
                    self.stop_robot()
                    self.get_logger().info("Square completed")
                    return

                self.start_turn_phase()
                return

            heading_correction = math.atan(self.cross_track_gain * cte)
            heading_correction = clamp(
                heading_correction,
                -self.max_heading_correction,
                self.max_heading_correction,
            )

            desired_yaw = wrap_to_pi(self.segment_yaw_ref - heading_correction)
            yaw_err = wrap_to_pi(desired_yaw - self.yaw)

            v = self.linear_speed

            if remaining < self.slowdown_distance:
                v *= max(0.25, remaining / max(self.slowdown_distance, 1e-6))

            if progress < self.segment_progress_guard:
                v = min(v, 0.6 * self.linear_speed)

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
                    f"progress={progress:.3f} remaining={remaining:.3f} "
                    f"cte={cte:.3f} desired_yaw={desired_yaw:.3f} "
                    f"yaw_err={yaw_err:.3f} v={cmd.linear.x:.3f} "
                    f"w={cmd.angular.z:.3f} settle={self.drive_settle_count}/{self.drive_settle_cycles}"
                )
                self.last_debug_time = now

        elif self.phase == "TURN":
            assert self.turn_target_yaw is not None

            yaw_err = wrap_to_pi(self.turn_target_yaw - self.yaw)

            if abs(yaw_err) <= self.turn_tolerance:
                self.turn_settle_count += 1
            else:
                self.turn_settle_count = 0

            if self.turn_settle_count >= self.turn_settle_cycles:
                self.stop_robot()
                self.side_idx += 1

                self.segment_yaw_ref = self.turn_target_yaw

                self.get_logger().info(
                    f"Finished turn -> next side {self.side_idx + 1}/{self.num_sides} | "
                    f"yaw={self.yaw:.3f}, target={self.turn_target_yaw:.3f}, "
                    f"final_err={yaw_err:.4f}"
                )

                self.start_drive_phase()
                return

            w = self.turn_pid.update(yaw_err, dt)


            slow_zone = math.radians(8.0)
            if abs(yaw_err) > slow_zone:
                if abs(w) < self.min_turn_speed:
                    w = math.copysign(self.min_turn_speed, yaw_err)

            cmd.linear.x = 0.0
            cmd.angular.z = clamp(w, -self.max_angular_speed, self.max_angular_speed)
            self.cmd_pub.publish(cmd)

            dbg_dt = (now - self.last_debug_time).nanoseconds / 1e9
            if dbg_dt > 1.0:
                turned = 0.0 if self.turn_start_yaw is None else wrap_to_pi(
                    self.yaw - self.turn_start_yaw
                )

                self.get_logger().info(
                    f"[TURN] side={self.side_idx + 1}/{self.num_sides} "
                    f"yaw={self.yaw:.3f} turned={turned:.3f} "
                    f"target_delta={self.turn_angle:.3f} "
                    f"yaw_err={yaw_err:.3f} settle={self.turn_settle_count}/{self.turn_settle_cycles} "
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