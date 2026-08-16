import json
import math
from pathlib import Path
from typing import Optional, List, Callable

import cv2
import numpy as np
import rclpy
import stag

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from tf_transformations import quaternion_from_euler


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def make_basis_func(basis_labels: List[str]) -> Callable[[float, float], np.ndarray]:
    """Same basis builder as in the AprilTag detector."""
    def basis(x: float, y: float) -> np.ndarray:
        feats = []
        for label in basis_labels:
            if label == "1":
                feats.append(1.0)
            elif label == "x":
                feats.append(x)
            elif label == "y":
                feats.append(y)
            elif label == "x^2":
                feats.append(x * x)
            elif label == "x*y":
                feats.append(x * y)
            elif label == "y^2":
                feats.append(y * y)
            elif label == "x^3":
                feats.append(x * x * x)
            elif label == "x^2*y":
                feats.append(x * x * y)
            elif label == "x*y^2":
                feats.append(x * y * y)
            elif label == "y^3":
                feats.append(y * y * y)
            elif label == "r^2*x":
                feats.append((x * x + y * y) * x)
            elif label == "r^2*y":
                feats.append((x * x + y * y) * y)
            else:
                raise ValueError(f"Unknown basis term: {label}")
        return np.array(feats, dtype=np.float64)
    return basis


class StagDetector(Node):
    def __init__(self):
        super().__init__("stag_detector")

        self.declare_parameter(
            "image_topic",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/image",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/camera_info",
        )
        self.declare_parameter("target_id", 0)
        self.declare_parameter("pose_topic_prefix", "/stag_pose_")

        self.declare_parameter("library_hd", 21)
        self.declare_parameter("error_correction", -1)

        self.declare_parameter("marker_size_m", 0.46)
        self.declare_parameter("x_offset", -0.0376)
        self.declare_parameter("y_offset", -0.0030)
        self.declare_parameter("yaw_offset", 1.5708)
        self.declare_parameter("tag_offset_forward", 0.032)
        self.declare_parameter("tag_offset_lateral", 0.0)

        self.declare_parameter("filter_alpha_pos", 0.7)
        self.declare_parameter("filter_alpha_yaw", 0.5)
        self.declare_parameter("max_position_jump", 0.80)
        self.declare_parameter("max_yaw_jump", 0.75)

        self.declare_parameter(
            "calibration_file",
            str(Path.home() / "stag_correction.json"),
        )

        self.declare_parameter("log_every_n", 60)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.target_id = int(self.get_parameter("target_id").value)
        self.pose_topic_prefix = str(self.get_parameter("pose_topic_prefix").value)

        self.library_hd = int(self.get_parameter("library_hd").value)
        self.error_correction = int(self.get_parameter("error_correction").value)

        self.marker_size_m = float(self.get_parameter("marker_size_m").value)
        self.x_offset = float(self.get_parameter("x_offset").value)
        self.y_offset = float(self.get_parameter("y_offset").value)
        self.yaw_offset = float(self.get_parameter("yaw_offset").value)
        self.tag_offset_forward = float(self.get_parameter("tag_offset_forward").value)
        self.tag_offset_lateral = float(self.get_parameter("tag_offset_lateral").value)

        self.filter_alpha_pos = float(self.get_parameter("filter_alpha_pos").value)
        self.filter_alpha_yaw = float(self.get_parameter("filter_alpha_yaw").value)
        self.max_position_jump = float(self.get_parameter("max_position_jump").value)
        self.max_yaw_jump = float(self.get_parameter("max_yaw_jump").value)
        self.log_every_n = int(self.get_parameter("log_every_n").value)

        calibration_file = str(self.get_parameter("calibration_file").value).strip()
        self.calib_enabled = False
        self.calib_basis = None
        self.calib_coef_x = None
        self.calib_coef_y = None
        if calibration_file:
            path = Path(calibration_file)
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    basis_labels = data["basis"]
                    self.calib_basis = make_basis_func(basis_labels)
                    self.calib_coef_x = np.array(data["coef_x"], dtype=np.float64)
                    self.calib_coef_y = np.array(data["coef_y"], dtype=np.float64)
                    self.calib_enabled = True
                    self.get_logger().info(
                        f"Calibration loaded: {len(basis_labels)} terms"
                    )
                except Exception as e:
                    self.get_logger().error(f"Failed to load calibration: {e}")
            else:
                self.get_logger().warn(
                    f"Calibration file not found: {path}, running without correction"
                )

        self.bridge = CvBridge()
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        s = self.marker_size_m / 2.0
        self.obj_points = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0],
        ], dtype=np.float32)

        topic = f"{self.pose_topic_prefix}{self.target_id}"
        self.publisher = self.create_publisher(PoseStamped, topic, 10)

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10
        )

        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None
        self.last_yaw: Optional[float] = None
        self.published_count = 0
        self.rejected_count = 0
        self.frame_count = 0

        self.get_logger().info(f"STag detector started for id={self.target_id} -> {topic}")
        self.get_logger().info(f"libraryHD={self.library_hd}, marker_size={self.marker_size_m} m")
        self.get_logger().info(
            f"Tag->base compensation: forward={self.tag_offset_forward:+.4f} m, "
            f"lateral={self.tag_offset_lateral:+.4f} m"
        )
        self.get_logger().info(
            f"Post-calibration: {'ENABLED' if self.calib_enabled else 'disabled'}"
        )

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float32)

    def is_outlier(self, x: float, y: float, yaw: float) -> bool:
        if self.last_x is None:
            return False
        dist = math.hypot(x - self.last_x, y - self.last_y)
        dyaw = abs(wrap_to_pi(yaw - self.last_yaw))
        return dist > self.max_position_jump or dyaw > self.max_yaw_jump

    def filter_pose(self, x: float, y: float, yaw: float):
        if self.last_x is None:
            self.last_x, self.last_y, self.last_yaw = x, y, yaw
            return x, y, yaw
        ax = self.filter_alpha_pos
        ayaw = self.filter_alpha_yaw
        x_f = ax * x + (1.0 - ax) * self.last_x
        y_f = ax * y + (1.0 - ax) * self.last_y
        dyaw = wrap_to_pi(yaw - self.last_yaw)
        yaw_f = wrap_to_pi(self.last_yaw + ayaw * dyaw)
        self.last_x, self.last_y, self.last_yaw = x_f, y_f, yaw_f
        return x_f, y_f, yaw_f

    def apply_calibration(self, x: float, y: float):
        if not self.calib_enabled:
            return x, y
        feats = self.calib_basis(x, y)
        err_x = float(np.dot(feats, self.calib_coef_x))
        err_y = float(np.dot(feats, self.calib_coef_y))
        return x - err_x, y - err_y

    def estimate_pose_from_corners(self, corners_one: np.ndarray):

        img_points = np.asarray(corners_one, dtype=np.float32).reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            img_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            raise RuntimeError("solvePnP failed")
        return rvec.flatten(), tvec.flatten()

    def process_one_marker(self, corners_one, msg_stamp):
        rvec, tvec = self.estimate_pose_from_corners(corners_one)

        raw_x = float(tvec[0])
        raw_y = float(tvec[1])

        x_marker = self.x_offset - raw_y
        y_marker = self.y_offset - raw_x

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        raw_yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        yaw = wrap_to_pi(-raw_yaw + self.yaw_offset + math.pi)

        x = x_marker - self.tag_offset_forward * math.cos(yaw) \
                     - self.tag_offset_lateral * math.cos(yaw + math.pi / 2.0)
        y = y_marker - self.tag_offset_forward * math.sin(yaw) \
                     - self.tag_offset_lateral * math.sin(yaw + math.pi / 2.0)

        x, y = self.apply_calibration(x, y)

        if self.is_outlier(x, y, yaw):
            self.rejected_count += 1
            if self.rejected_count % max(1, self.log_every_n) == 1:
                self.get_logger().warn(
                    f"id={self.target_id} outlier rejected "
                    f"(total: {self.rejected_count})"
                )
            return

        x, y, yaw = self.filter_pose(x, y, yaw)

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        pose = PoseStamped()
        pose.header.stamp = msg_stamp
        pose.header.frame_id = "world"
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)

        self.publisher.publish(pose)
        self.published_count += 1

        if self.published_count % max(1, self.log_every_n) == 1:
            self.get_logger().info(
                f"id={self.target_id}: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} "
                f"(published: {self.published_count})"
            )

    def image_callback(self, msg: Image):
        self.frame_count += 1
        if self.camera_matrix is None or self.dist_coeffs is None:
            if self.frame_count % max(1, self.log_every_n) == 1:
                self.get_logger().warn("Waiting for camera_info...")
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        try:
            if self.error_correction >= 0:
                corners, ids, _ = stag.detectMarkers(
                    frame, self.library_hd, errorCorrection=self.error_correction
                )
            else:
                corners, ids, _ = stag.detectMarkers(frame, self.library_hd)
        except Exception as e:
            self.get_logger().error(f"stag.detectMarkers failed: {e}")
            return

        if ids is None or len(ids) == 0:
            return

        ids_flat = np.asarray(ids).flatten()
        matches = np.where(ids_flat == self.target_id)[0]
        if len(matches) == 0:
            return

        idx = int(matches[0])
        try:
            self.process_one_marker(corners[idx], msg.header.stamp)
        except Exception as e:
            self.get_logger().error(f"Processing id={self.target_id} failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = StagDetector()
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