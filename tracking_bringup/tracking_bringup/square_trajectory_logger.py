import csv
import math
from pathlib import Path
from typing import Optional, Dict

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from tf_transformations import euler_from_quaternion


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_pose(msg: PoseStamped) -> float:
    q = msg.pose.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return wrap_to_pi(yaw)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class RobotLog:

    def __init__(self, node: Node, robot_id: int, csv_path: Path):
        self.robot_id = robot_id
        self.csv_path = csv_path
        self.csv_file = open(csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            "sample_idx",
            "t",
            "source",
            "visual_stamp", "visual_x", "visual_y", "visual_yaw",
            "gt_stamp",     "gt_x",     "gt_y",     "gt_yaw",
        ])
        self.csv_file.flush()
        self.sample_idx = 0

        self.last_visual: Optional[Dict[str, float]] = None
        self.last_gt: Optional[Dict[str, float]] = None

        self.start_time = node.get_clock().now()
        node.get_logger().info(f"Robot {robot_id} -> {csv_path}")

    def update_visual(self, msg: PoseStamped, t_sec: float):
        self.last_visual = {
            "stamp": stamp_to_sec(msg.header.stamp),
            "x": float(msg.pose.position.x),
            "y": float(msg.pose.position.y),
            "yaw": yaw_from_pose(msg),
        }
        self._write_row(t_sec, "visual")

    def update_gt(self, msg: PoseStamped, t_sec: float):
        self.last_gt = {
            "stamp": stamp_to_sec(msg.header.stamp),
            "x": float(msg.pose.position.x),
            "y": float(msg.pose.position.y),
            "yaw": yaw_from_pose(msg),
        }
        self._write_row(t_sec, "gt")

    def _write_row(self, t_sec: float, source: str):
        v = self.last_visual or {"stamp": float("nan"), "x": float("nan"),
                                 "y": float("nan"), "yaw": float("nan")}
        g = self.last_gt or {"stamp": float("nan"), "x": float("nan"),
                             "y": float("nan"), "yaw": float("nan")}
        self.writer.writerow([
            self.sample_idx,
            f"{t_sec:.6f}",
            source,
            f"{v['stamp']:.9f}", f"{v['x']:.6f}", f"{v['y']:.6f}", f"{v['yaw']:.6f}",
            f"{g['stamp']:.9f}", f"{g['x']:.6f}", f"{g['y']:.6f}", f"{g['yaw']:.6f}",
        ])
        self.csv_file.flush()
        self.sample_idx += 1

    def close(self):
        if not self.csv_file.closed:
            self.csv_file.close()


class MultiTrajectoryLogger(Node):
    def __init__(self):
        super().__init__("multi_trajectory_logger")

        self.declare_parameter("output_dir", str(Path.home() / "wspace" / "logs"))
        self.declare_parameter("visual_topic_prefix", "/apriltag_pose_")
        self.declare_parameter("gt_topic_prefix", "/ground_truth_pose_")
        self.declare_parameter("robot_ids", [0, 1, 2])

        self.output_dir = Path(self.get_parameter("output_dir").value)
        self.visual_topic_prefix = str(self.get_parameter("visual_topic_prefix").value)
        self.gt_topic_prefix = str(self.get_parameter("gt_topic_prefix").value)
        self.robot_ids = [int(v) for v in self.get_parameter("robot_ids").value]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = self.get_clock().now()
        ts = self.start_time.nanoseconds

        self.logs: Dict[int, RobotLog] = {}
        for rid in self.robot_ids:
            path = self.output_dir / f"multi_robot_{ts}_robot_{rid}.csv"
            self.logs[rid] = RobotLog(self, rid, path)

        for rid in self.robot_ids:
            v_topic = f"{self.visual_topic_prefix}{rid}"
            g_topic = f"{self.gt_topic_prefix}{rid}"
            self.create_subscription(PoseStamped, v_topic,
                                     self._make_visual_cb(rid), 10)
            self.create_subscription(PoseStamped, g_topic,
                                     self._make_gt_cb(rid), 10)
            self.get_logger().info(f"Robot {rid}: visual={v_topic}, gt={g_topic}")

        self.total_count = 0

    def _make_visual_cb(self, robot_id: int):
        def cb(msg: PoseStamped):
            now = self.get_clock().now()
            t = (now - self.start_time).nanoseconds * 1e-9
            try:
                self.logs[robot_id].update_visual(msg, t)
                self._maybe_log_status()
            except Exception as e:
                self.get_logger().error(f"Visual write failed for robot {robot_id}: {e}")
        return cb

    def _make_gt_cb(self, robot_id: int):
        def cb(msg: PoseStamped):
            now = self.get_clock().now()
            t = (now - self.start_time).nanoseconds * 1e-9
            try:
                self.logs[robot_id].update_gt(msg, t)
                self._maybe_log_status()
            except Exception as e:
                self.get_logger().error(f"GT write failed for robot {robot_id}: {e}")
        return cb

    def _maybe_log_status(self):
        self.total_count += 1
        if self.total_count % 200 == 1:
            counts = ", ".join(
                f"R{rid}={self.logs[rid].sample_idx}" for rid in self.robot_ids
            )
            self.get_logger().info(f"Samples: {counts}")

    def destroy_node(self):
        for rl in self.logs.values():
            rl.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MultiTrajectoryLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()