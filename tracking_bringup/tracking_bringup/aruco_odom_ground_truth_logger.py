import csv
import math
from pathlib import Path

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from message_filters import Subscriber, ApproximateTimeSynchronizer


def norm_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q) -> float:
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return norm_angle(yaw)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class ArucoOdomGroundTruthLogger(Node):
    def __init__(self):
        super().__init__("aruco_odom_ground_truth_logger")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("aruco_topic", "/aruco_pose")
        self.declare_parameter("aruco_marker_center_topic", "/aruco_marker_center")
        self.declare_parameter("ground_truth_topic", "/ground_truth_pose")

        self.declare_parameter("angular_speed", 0.5)
        self.declare_parameter("num_turns", 5.0)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("output_dir", str(Path.home() / "wspace" / "logs"))

        self.declare_parameter("sync_queue_size", 50)
        self.declare_parameter("sync_slop", 0.05)

        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.aruco_topic = str(self.get_parameter("aruco_topic").value)
        self.aruco_marker_center_topic = str(self.get_parameter("aruco_marker_center_topic").value)
        self.gt_topic = str(self.get_parameter("ground_truth_topic").value)

        self.angular_speed = float(self.get_parameter("angular_speed").value)
        self.num_turns = float(self.get_parameter("num_turns").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.output_dir = Path(self.get_parameter("output_dir").value)

        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop = float(self.get_parameter("sync_slop").value)

        if abs(self.angular_speed) < 1e-9:
            raise ValueError("angular_speed must be non-zero")

        self.test_duration = (2.0 * math.pi * self.num_turns) / abs(self.angular_speed)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.odom_sub = Subscriber(self, Odometry, self.odom_topic)
        self.aruco_sub = Subscriber(self, PoseStamped, self.aruco_topic)
        self.marker_center_sub = Subscriber(self, PoseStamped, self.aruco_marker_center_topic)
        self.gt_sub = Subscriber(self, PoseStamped, self.gt_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.odom_sub, self.aruco_sub, self.marker_center_sub, self.gt_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.synced_callback)

        self.cmd_timer = self.create_timer(1.0 / self.publish_rate, self.cmd_timer_callback)

        self.start_time = self.get_clock().now()
        self.finished = False
        self.shutdown_called = False
        self.synced_count = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = self.output_dir / f"aruco_odom_gt_{self.start_time.nanoseconds}.csv"
        self.csv_file = open(filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "t",

            "odom_stamp",
            "aruco_stamp",
            "marker_center_stamp",
            "gt_stamp",

            "dt_odom_gt",
            "dt_aruco_gt",
            "dt_marker_gt",

            "odom_x", "odom_y", "odom_yaw",

            "aruco_frame",
            "aruco_x", "aruco_y", "aruco_z", "aruco_yaw",

            "marker_center_frame",
            "marker_center_x", "marker_center_y", "marker_center_z", "marker_center_yaw",

            "gt_frame",
            "gt_x", "gt_y", "gt_z", "gt_yaw",
        ])
        self.csv_file.flush()

        self.get_logger().info(f"Logging to: {filename}")
        self.get_logger().info(
            f"Target: {self.num_turns:.1f} turns | "
            f"angular_speed={self.angular_speed:.3f} rad/s | "
            f"duration ≈ {self.test_duration:.2f} s"
        )
        self.get_logger().info(
            f"Approx sync enabled: queue_size={self.sync_queue_size}, slop={self.sync_slop:.3f}s"
        )

    def cmd_timer_callback(self):
        if self.finished:
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        cmd = Twist()
        cmd.linear.x = 0.0

        if elapsed < self.test_duration:
            cmd.angular.z = self.angular_speed
        else:
            cmd.angular.z = 0.0
            self.finished = True
            self.get_logger().info("Test finished")
            self.shutdown_node()

        self.cmd_pub.publish(cmd)

    def synced_callback(
        self,
        odom_msg: Odometry,
        aruco_msg: PoseStamped,
        marker_msg: PoseStamped,
        gt_msg: PoseStamped,
    ):
        if self.shutdown_called:
            return

        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9

        odom_stamp = stamp_to_sec(odom_msg.header.stamp)
        aruco_stamp = stamp_to_sec(aruco_msg.header.stamp)
        marker_stamp = stamp_to_sec(marker_msg.header.stamp)
        gt_stamp = stamp_to_sec(gt_msg.header.stamp)

        dt_odom_gt = abs(odom_stamp - gt_stamp)
        dt_aruco_gt = abs(aruco_stamp - gt_stamp)
        dt_marker_gt = abs(marker_stamp - gt_stamp)

        odom_pose = odom_msg.pose.pose
        odom_x = float(odom_pose.position.x)
        odom_y = float(odom_pose.position.y)
        odom_yaw = yaw_from_quat(odom_pose.orientation)

        aruco_pose = aruco_msg.pose
        aruco_x = float(aruco_pose.position.x)
        aruco_y = float(aruco_pose.position.y)
        aruco_z = float(aruco_pose.position.z)
        aruco_yaw = yaw_from_quat(aruco_pose.orientation)

        marker_pose = marker_msg.pose
        marker_center_x = float(marker_pose.position.x)
        marker_center_y = float(marker_pose.position.y)
        marker_center_z = float(marker_pose.position.z)
        marker_center_yaw = yaw_from_quat(marker_pose.orientation)

        gt_pose = gt_msg.pose
        gt_x = float(gt_pose.position.x)
        gt_y = float(gt_pose.position.y)
        gt_z = float(gt_pose.position.z)
        gt_yaw = yaw_from_quat(gt_pose.orientation)

        self.csv_writer.writerow([
            f"{t:.6f}",

            f"{odom_stamp:.9f}",
            f"{aruco_stamp:.9f}",
            f"{marker_stamp:.9f}",
            f"{gt_stamp:.9f}",

            f"{dt_odom_gt:.6f}",
            f"{dt_aruco_gt:.6f}",
            f"{dt_marker_gt:.6f}",

            f"{odom_x:.6f}", f"{odom_y:.6f}", f"{odom_yaw:.6f}",

            aruco_msg.header.frame_id,
            f"{aruco_x:.6f}", f"{aruco_y:.6f}", f"{aruco_z:.6f}", f"{aruco_yaw:.6f}",

            marker_msg.header.frame_id,
            f"{marker_center_x:.6f}", f"{marker_center_y:.6f}", f"{marker_center_z:.6f}", f"{marker_center_yaw:.6f}",

            gt_msg.header.frame_id,
            f"{gt_x:.6f}", f"{gt_y:.6f}", f"{gt_z:.6f}", f"{gt_yaw:.6f}",
        ])
        self.csv_file.flush()

        self.synced_count += 1
        if self.synced_count % 50 == 1:
            self.get_logger().info(
                f"Synced samples={self.synced_count} | "
                f"dt odom-gt={dt_odom_gt:.4f}s, "
                f"aruco-gt={dt_aruco_gt:.4f}s, "
                f"marker-gt={dt_marker_gt:.4f}s"
            )

    def shutdown_node(self):
        if self.shutdown_called:
            return
        self.shutdown_called = True

        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)

        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.close()

        self.get_logger().info("CSV saved")
        self.destroy_node()
        rclpy.shutdown()

    def destroy_node(self):
        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoOdomGroundTruthLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, "csv_file") and not node.csv_file.closed:
            node.csv_file.close()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()