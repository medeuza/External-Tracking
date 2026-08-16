import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage


def normalize_frame_name(name: str) -> str:
    return (name or "").strip().lstrip("/")


class GroundTruthFromTF(Node):
    def __init__(self):
        super().__init__("ground_truth_from_tf")

        self.declare_parameter("input_topic", "/world/default/dynamic_pose/info")
        self.declare_parameter("output_topic", "/ground_truth_pose")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("parent_frame", "world")
        self.declare_parameter("child_frame", "base_link")
        self.declare_parameter("child_frame_contains", "base_link")
        self.declare_parameter("transform_index", 0)
        self.declare_parameter("republish_rate_hz", 30.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.parent_frame = normalize_frame_name(str(self.get_parameter("parent_frame").value))
        self.child_frame = normalize_frame_name(str(self.get_parameter("child_frame").value))
        self.child_frame_contains = normalize_frame_name(
            str(self.get_parameter("child_frame_contains").value)
        )
        self.transform_index = int(self.get_parameter("transform_index").value)
        self.republish_rate_hz = float(self.get_parameter("republish_rate_hz").value)

        self.sub = self.create_subscription(TFMessage, self.input_topic, self.callback, 10)
        self.pub = self.create_publisher(PoseStamped, self.output_topic, 10)

        self.last_pose_msg = None

        if self.republish_rate_hz > 0.0:
            self.republish_timer = self.create_timer(
                1.0 / self.republish_rate_hz,
                self.republish_callback,
            )

        self.first_msg_logged = False
        self.selection_mode_logged = False
        self.published_count = 0
        self.republished_count = 0
        self.warn_counter = 0

        self.get_logger().info(f"Input topic: {self.input_topic}")
        self.get_logger().info(f"Output topic: {self.output_topic}")
        self.get_logger().info(f"Re-publishing last cached pose at {self.republish_rate_hz:.1f} Hz")

    def choose_transform(self, transforms):
        normalized = [
            (normalize_frame_name(tr.header.frame_id), normalize_frame_name(tr.child_frame_id), tr)
            for tr in transforms
        ]

        for parent, child, tr in normalized:
            if not parent and not child:
                break
            if parent == self.parent_frame and child == self.child_frame:
                if not self.selection_mode_logged:
                    self.get_logger().info(
                        f"Using exact frame match: {self.parent_frame} -> {self.child_frame}"
                    )
                    self.selection_mode_logged = True
                return tr

        if self.child_frame_contains:
            matches = [tr for parent, child, tr in normalized
                       if child and self.child_frame_contains in child]
            if len(matches) == 1:
                if not self.selection_mode_logged:
                    self.get_logger().warn(
                        f"Using child contains='{self.child_frame_contains}'."
                    )
                    self.selection_mode_logged = True
                return matches[0]

        if 0 <= self.transform_index < len(transforms):
            if not self.selection_mode_logged:
                self.get_logger().warn(
                    f"Frame names empty or non-matching. Using fallback index={self.transform_index}."
                )
                self.selection_mode_logged = True
            return transforms[self.transform_index]

        return None

    def callback(self, msg: TFMessage):
        if not self.first_msg_logged:
            self.get_logger().info(f"Received first TFMessage with {len(msg.transforms)} transforms")
            for i, tr in enumerate(msg.transforms):
                self.get_logger().info(
                    f"  [{i}] '{tr.header.frame_id}' -> '{tr.child_frame_id}' | "
                    f"x={tr.transform.translation.x:.4f}, y={tr.transform.translation.y:.4f}"
                )
            self.first_msg_logged = True

        if not msg.transforms:
            return

        try:
            target_tf = self.choose_transform(msg.transforms)
        except Exception as e:
            self.get_logger().error(f"choose_transform failed: {e}")
            return

        if target_tf is None:
            return

        try:
            pose_msg = PoseStamped()
            if target_tf.header.stamp.sec == 0 and target_tf.header.stamp.nanosec == 0:
                pose_msg.header.stamp = self.get_clock().now().to_msg()
            else:
                pose_msg.header.stamp = target_tf.header.stamp
            pose_msg.header.frame_id = self.frame_id

            pose_msg.pose.position.x = float(target_tf.transform.translation.x)
            pose_msg.pose.position.y = float(target_tf.transform.translation.y)
            pose_msg.pose.position.z = float(target_tf.transform.translation.z)
            pose_msg.pose.orientation.x = float(target_tf.transform.rotation.x)
            pose_msg.pose.orientation.y = float(target_tf.transform.rotation.y)
            pose_msg.pose.orientation.z = float(target_tf.transform.rotation.z)
            pose_msg.pose.orientation.w = float(target_tf.transform.rotation.w)

            self.last_pose_msg = pose_msg
            self.pub.publish(pose_msg)
            self.published_count += 1

            if self.published_count % 100 == 1:
                self.get_logger().info(
                    f"Published GT: x={pose_msg.pose.position.x:.3f}, "
                    f"y={pose_msg.pose.position.y:.3f} "
                    f"(direct: {self.published_count}, republished: {self.republished_count})"
                )
        except Exception as e:
            self.get_logger().error(f"Failed to publish: {e}")

    def republish_callback(self):
        if self.last_pose_msg is None:
            return
        try:
            self.last_pose_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(self.last_pose_msg)
            self.republished_count += 1
        except Exception as e:
            self.get_logger().error(f"Republish failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthFromTF()
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