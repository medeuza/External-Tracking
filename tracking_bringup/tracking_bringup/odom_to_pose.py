import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class OdomToPose(Node):
    def __init__(self):
        super().__init__("odom_to_pose")

        self.declare_parameter("input_topic", "/odom")
        self.declare_parameter("output_topic", "/odom_pose")
        self.declare_parameter("output_frame", "world")

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.output_frame = str(self.get_parameter("output_frame").value)

        self.sub = self.create_subscription(
            Odometry, self.input_topic, self.callback, 10)
        self.pub = self.create_publisher(
            PoseStamped, self.output_topic, 10)

        self.count = 0
        self.first_logged = False

        self.get_logger().info(f"Input  (Odometry):    {self.input_topic}")
        self.get_logger().info(f"Output (PoseStamped): {self.output_topic}")
        self.get_logger().info(f"Output frame_id:      {self.output_frame}")

    def callback(self, msg: Odometry):
        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.output_frame
        out.pose = msg.pose.pose
        self.pub.publish(out)

        self.count += 1
        if not self.first_logged:
            self.get_logger().info(
                f"First pose: x={out.pose.position.x:.3f}, "
                f"y={out.pose.position.y:.3f}"
            )
            self.first_logged = True
        elif self.count % 200 == 1:
            self.get_logger().info(
                f"Published: {self.count}   "
                f"last x={out.pose.position.x:.3f}, "
                f"y={out.pose.position.y:.3f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = OdomToPose()
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