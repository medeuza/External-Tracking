import math
from typing import Dict, List

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion, quaternion_from_euler


def yaw_from_quat(q) -> float:
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw


class MultiOdomToPose(Node):
    def __init__(self):
        super().__init__("multi_odom_to_pose")

        self.declare_parameter("robot_ids", [0, 1, 2, 3, 4, 5])
        self.declare_parameter("model_prefix", "turtlebot3_burger_aruco_")
        self.declare_parameter("id_pad", 3)
        self.declare_parameter("input_topic_template",
                               "/model/{model}/odom")
        self.declare_parameter("output_topic_prefix", "/odom_pose_")
        self.declare_parameter("output_frame", "world")


        self.declare_parameter("start_poses_x", [4.0, 1.0, -4.25, -4.25, 1.0, 4.0])
        self.declare_parameter("start_poses_y", [9.0, 9.0, 9.0, -9.0, -9.0, -9.0])
        self.declare_parameter("start_poses_yaw", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.robot_ids: List[int] = [
            int(v) for v in self.get_parameter("robot_ids").value
        ]
        self.model_prefix = str(self.get_parameter("model_prefix").value)
        self.id_pad = int(self.get_parameter("id_pad").value)
        self.input_template = str(
            self.get_parameter("input_topic_template").value)
        self.output_prefix = str(
            self.get_parameter("output_topic_prefix").value)
        self.output_frame = str(self.get_parameter("output_frame").value)

        sx = [float(v) for v in self.get_parameter("start_poses_x").value]
        sy = [float(v) for v in self.get_parameter("start_poses_y").value]
        sy_yaw = [float(v) for v in self.get_parameter("start_poses_yaw").value]


        self.start: Dict[int, tuple] = {}
        for i, rid in enumerate(self.robot_ids):
            x = sx[i] if i < len(sx) else 0.0
            y = sy[i] if i < len(sy) else 0.0
            yaw = sy_yaw[i] if i < len(sy_yaw) else 0.0
            self.start[rid] = (x, y, yaw)

        self.pubs: Dict[int, rclpy.publisher.Publisher] = {}
        self.counts: Dict[int, int] = {}
        self.first_logged: Dict[int, bool] = {}

        for rid in self.robot_ids:
            model = f"{self.model_prefix}{str(rid).zfill(self.id_pad)}"
            in_topic = self.input_template.format(model=model)
            out_topic = f"{self.output_prefix}{rid}"

            pub = self.create_publisher(PoseStamped, out_topic, 10)
            self.pubs[rid] = pub
            self.counts[rid] = 0
            self.first_logged[rid] = False

            self.create_subscription(
                Odometry, in_topic, self._make_cb(rid), 10)

            sx_, sy_, syaw_ = self.start[rid]
            self.get_logger().info(
                f"R{rid}: {in_topic}  ->  {out_topic}   "
                f"start=({sx_:+.3f}, {sy_:+.3f}, yaw={syaw_:+.3f})"
            )

    def _make_cb(self, robot_id: int):
        sx, sy, syaw = self.start[robot_id]
        cos_s, sin_s = math.cos(syaw), math.sin(syaw)

        def cb(msg: Odometry):
            ox = msg.pose.pose.position.x
            oy = msg.pose.pose.position.y
            oyaw = yaw_from_quat(msg.pose.pose.orientation)

            wx = sx + cos_s * ox - sin_s * oy
            wy = sy + sin_s * ox + cos_s * oy
            wyaw = syaw + oyaw

            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, wyaw)

            out = PoseStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = self.output_frame
            out.pose.position.x = float(wx)
            out.pose.position.y = float(wy)
            out.pose.position.z = float(msg.pose.pose.position.z)
            out.pose.orientation.x = float(qx)
            out.pose.orientation.y = float(qy)
            out.pose.orientation.z = float(qz)
            out.pose.orientation.w = float(qw)
            self.pubs[robot_id].publish(out)

            self.counts[robot_id] += 1
            if not self.first_logged[robot_id]:
                self.get_logger().info(
                    f"R{robot_id} first pose (world): "
                    f"x={wx:.3f}, y={wy:.3f}, yaw={wyaw:.3f}"
                )
                self.first_logged[robot_id] = True
            elif self.counts[robot_id] % 500 == 1:
                counts_str = " ".join(
                    f"R{rid}={self.counts[rid]}" for rid in self.robot_ids
                )
                self.get_logger().info(f"counts: {counts_str}")
        return cb


def main(args=None):
    rclpy.init(args=args)
    node = MultiOdomToPose()
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