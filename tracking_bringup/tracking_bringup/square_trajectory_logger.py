import csv
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from message_filters import Subscriber, ApproximateTimeSynchronizer


def wrap_to_pi(a: float) -> float:
    import math
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q) -> float:
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return wrap_to_pi(yaw)


def yaw_from_pose_msg(msg: PoseStamped) -> float:
    return yaw_from_quat(msg.pose.orientation)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class SquareTrajectoryLogger(Node):
    def __init__(self):
        super().__init__("square_trajectory_logger")

        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("aruco_topic", "/aruco_pose")
        self.declare_parameter("ground_truth_topic", "/ground_truth_pose")
        self.declare_parameter("output_dir", str(Path.home() / "wspace" / "logs"))
        self.declare_parameter("mode", "odom")  # odom or aruco
        self.declare_parameter("sync_queue_size", 100)
        self.declare_parameter("sync_slop", 0.05)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.aruco_topic = str(self.get_parameter("aruco_topic").value)
        self.gt_topic = str(self.get_parameter("ground_truth_topic").value)
        self.output_dir = Path(self.get_parameter("output_dir").value)
        self.mode = str(self.get_parameter("mode").value)

        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop = float(self.get_parameter("sync_slop").value)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.odom_sub = Subscriber(self, Odometry, self.odom_topic)
        self.aruco_sub = Subscriber(self, PoseStamped, self.aruco_topic)
        self.gt_sub = Subscriber(self, PoseStamped, self.gt_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.odom_sub, self.aruco_sub, self.gt_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.synced_callback)

        self.start_time = self.get_clock().now()

        filename = self.output_dir / f"square_{self.mode}_{self.start_time.nanoseconds}.csv"
        self.csv_file = open(filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "t",
            "odom_stamp", "aruco_stamp", "gt_stamp",
            "dt_odom_gt", "dt_aruco_gt",

            "odom_x", "odom_y", "odom_yaw",
            "aruco_x", "aruco_y", "aruco_yaw",
            "gt_x", "gt_y", "gt_yaw",
        ])
        self.csv_file.flush()

        self.get_logger().info(f"Logging to: {filename}")
        self.get_logger().info(f"Mode: {self.mode}")

    def synced_callback(self, odom_msg: Odometry, aruco_msg: PoseStamped, gt_msg: PoseStamped):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9

        odom_stamp = stamp_to_sec(odom_msg.header.stamp)
        aruco_stamp = stamp_to_sec(aruco_msg.header.stamp)
        gt_stamp = stamp_to_sec(gt_msg.header.stamp)

        dt_odom_gt = abs(odom_stamp - gt_stamp)
        dt_aruco_gt = abs(aruco_stamp - gt_stamp)

        odom_pose = odom_msg.pose.pose
        odom_x = float(odom_pose.position.x)
        odom_y = float(odom_pose.position.y)
        odom_yaw = yaw_from_quat(odom_pose.orientation)

        aruco_x = float(aruco_msg.pose.position.x)
        aruco_y = float(aruco_msg.pose.position.y)
        aruco_yaw = yaw_from_pose_msg(aruco_msg)

        gt_x = float(gt_msg.pose.position.x)
        gt_y = float(gt_msg.pose.position.y)
        gt_yaw = yaw_from_pose_msg(gt_msg)

        self.csv_writer.writerow([
            f"{t:.6f}",
            f"{odom_stamp:.9f}", f"{aruco_stamp:.9f}", f"{gt_stamp:.9f}",
            f"{dt_odom_gt:.6f}", f"{dt_aruco_gt:.6f}",
            f"{odom_x:.6f}", f"{odom_y:.6f}", f"{odom_yaw:.6f}",
            f"{aruco_x:.6f}", f"{aruco_y:.6f}", f"{aruco_yaw:.6f}",
            f"{gt_x:.6f}", f"{gt_y:.6f}", f"{gt_yaw:.6f}",
        ])
        self.csv_file.flush()

    def destroy_node(self):
        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SquareTrajectoryLogger()

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