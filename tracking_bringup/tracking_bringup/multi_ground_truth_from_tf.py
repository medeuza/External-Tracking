from typing import Dict, Optional, List

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage


class MultiGroundTruthFromTF(Node):
    def __init__(self):
        super().__init__("multi_ground_truth_from_tf")

        self.declare_parameter("input_topic", "/world/default/dynamic_pose/info")
        self.declare_parameter("output_topic_prefix", "/ground_truth_pose_")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("transform_indices", [0, 1, 2])
        self.declare_parameter("republish_rate_hz", 30.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic_prefix = str(self.get_parameter("output_topic_prefix").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.transform_indices: List[int] = [
            int(v) for v in self.get_parameter("transform_indices").value
        ]
        self.republish_rate_hz = float(self.get_parameter("republish_rate_hz").value)

        self.sub = self.create_subscription(
            TFMessage, self.input_topic, self.callback, 10
        )

        self.pubs = {}
        self.last_pose: Dict[int, Optional[PoseStamped]] = {}
        for robot_id, _ in enumerate(self.transform_indices):
            topic = f"{self.output_topic_prefix}{robot_id}"
            self.pubs[robot_id] = self.create_publisher(PoseStamped, topic, 10)
            self.last_pose[robot_id] = None
            self.get_logger().info(
                f"Robot {robot_id} <- transform[{self.transform_indices[robot_id]}] -> {topic}"
            )

        if self.republish_rate_hz > 0.0:
            self.republish_timer = self.create_timer(
                1.0 / self.republish_rate_hz, self.republish_callback
            )

        self.first_msg_logged = False
        self.published_counts = {i: 0 for i in range(len(self.transform_indices))}
        self.republished_counts = {i: 0 for i in range(len(self.transform_indices))}

        self.get_logger().info(f"Input topic: {self.input_topic}")
        self.get_logger().info(f"Re-publishing at {self.republish_rate_hz:.1f} Hz")

    def callback(self, msg: TFMessage):
        if not self.first_msg_logged:
            self.get_logger().info(
                f"Received first TFMessage with {len(msg.transforms)} transforms"
            )
            for i, tr in enumerate(msg.transforms):
                self.get_logger().info(
                    f"  [{i}] '{tr.header.frame_id}' -> '{tr.child_frame_id}' | "
                    f"x={tr.transform.translation.x:.4f}, "
                    f"y={tr.transform.translation.y:.4f}"
                )
            self.first_msg_logged = True

        if not msg.transforms:
            return

        n = len(msg.transforms)
        for robot_id, tf_idx in enumerate(self.transform_indices):
            if tf_idx < 0 or tf_idx >= n:
                continue
            tr = msg.transforms[tf_idx]
            try:
                pose = PoseStamped()
                if tr.header.stamp.sec == 0 and tr.header.stamp.nanosec == 0:
                    pose.header.stamp = self.get_clock().now().to_msg()
                else:
                    pose.header.stamp = tr.header.stamp
                pose.header.frame_id = self.frame_id

                pose.pose.position.x = float(tr.transform.translation.x)
                pose.pose.position.y = float(tr.transform.translation.y)
                pose.pose.position.z = float(tr.transform.translation.z)
                pose.pose.orientation.x = float(tr.transform.rotation.x)
                pose.pose.orientation.y = float(tr.transform.rotation.y)
                pose.pose.orientation.z = float(tr.transform.rotation.z)
                pose.pose.orientation.w = float(tr.transform.rotation.w)

                self.last_pose[robot_id] = pose
                self.pubs[robot_id].publish(pose)
                self.published_counts[robot_id] += 1

                if self.published_counts[robot_id] % 100 == 1:
                    self.get_logger().info(
                        f"GT robot {robot_id}: x={pose.pose.position.x:.3f}, "
                        f"y={pose.pose.position.y:.3f} "
                        f"(direct: {self.published_counts[robot_id]}, "
                        f"republished: {self.republished_counts[robot_id]})"
                    )
            except Exception as e:
                self.get_logger().error(f"Publish failed for robot {robot_id}: {e}")

    def republish_callback(self):
        for robot_id, last in self.last_pose.items():
            if last is None:
                continue
            try:
                last.header.stamp = self.get_clock().now().to_msg()
                self.pubs[robot_id].publish(last)
                self.republished_counts[robot_id] += 1
            except Exception as e:
                self.get_logger().error(f"Republish failed for robot {robot_id}: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MultiGroundTruthFromTF()
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
