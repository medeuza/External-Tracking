import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage


class GroundTruthFromTF(Node):
    def __init__(self):
        super().__init__("ground_truth_from_tf")

        self.declare_parameter("input_topic", "/world/default/dynamic_pose/info")
        self.declare_parameter("output_topic", "/ground_truth_pose")
        self.declare_parameter("frame_id", "world")

        self.declare_parameter("parent_frame", "world")
        self.declare_parameter("child_frame", "base_link")
        self.declare_parameter("transform_index", 0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.parent_frame = str(self.get_parameter("parent_frame").value)
        self.child_frame = str(self.get_parameter("child_frame").value)
        self.transform_index = int(self.get_parameter("transform_index").value)

        self.sub = self.create_subscription(
            TFMessage,
            self.input_topic,
            self.callback,
            10,
        )

        self.pub = self.create_publisher(
            PoseStamped,
            self.output_topic,
            10,
        )

        self.first_msg_logged = False
        self.selection_mode_logged = False
        self.warn_counter = 0

        self.get_logger().info(f"Input topic: {self.input_topic}")
        self.get_logger().info(f"Output topic: {self.output_topic}")
        self.get_logger().info(f"Published frame id: {self.frame_id}")
        self.get_logger().info(
            f"Preferred transform: {self.parent_frame} -> {self.child_frame}, "
            f"fallback index={self.transform_index}"
        )

    def callback(self, msg: TFMessage):
        if not self.first_msg_logged:
            self.get_logger().info(f"Received first TFMessage with {len(msg.transforms)} transforms")
            for i, tr in enumerate(msg.transforms):
                self.get_logger().info(
                    f"[{i}] {tr.header.frame_id} -> {tr.child_frame_id} | "
                    f"x={tr.transform.translation.x:.4f}, "
                    f"y={tr.transform.translation.y:.4f}, "
                    f"z={tr.transform.translation.z:.4f}"
                )
            self.first_msg_logged = True

        if len(msg.transforms) == 0:
            return

        target_tf = None

        # 1) Сначала пробуем найти по именам frame, если они вообще есть
        has_named_frames = any(
            (tr.header.frame_id or tr.child_frame_id) for tr in msg.transforms
        )

        if has_named_frames:
            for tr in msg.transforms:
                if tr.header.frame_id == self.parent_frame and tr.child_frame_id == self.child_frame:
                    target_tf = tr
                    if not self.selection_mode_logged:
                        self.get_logger().info(
                            f"Using frame-based selection: {self.parent_frame} -> {self.child_frame}"
                        )
                        self.selection_mode_logged = True
                    break

        # 2) Если имён нет или нужный transform не найден — fallback на индекс
        if target_tf is None:
            if self.transform_index < 0 or self.transform_index >= len(msg.transforms):
                self.warn_counter += 1
                if self.warn_counter % 50 == 1:
                    self.get_logger().warn(
                        f"transform_index={self.transform_index} out of range, "
                        f"available: 0..{len(msg.transforms)-1}"
                    )
                return

            target_tf = msg.transforms[self.transform_index]
            if not self.selection_mode_logged:
                self.get_logger().warn(
                    f"Frame names are unavailable or target frame not found. "
                    f"Using fallback transform_index={self.transform_index}"
                )
                self.selection_mode_logged = True

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

        self.pub.publish(pose_msg)


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