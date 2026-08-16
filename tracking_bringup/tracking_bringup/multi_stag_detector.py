import json
import math
from pathlib import Path
from typing import Optional, Dict, List, Callable

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
    def basis(x: float, y: float) -> np.ndarray:
        feats = []
        for label in basis_labels:
            if   label == "1":      feats.append(1.0)
            elif label == "x":      feats.append(x)
            elif label == "y":      feats.append(y)
            elif label == "x^2":    feats.append(x * x)
            elif label == "x*y":    feats.append(x * y)
            elif label == "y^2":    feats.append(y * y)
            elif label == "x^3":    feats.append(x * x * x)
            elif label == "x^2*y":  feats.append(x * x * y)
            elif label == "x*y^2":  feats.append(x * y * y)
            elif label == "y^3":    feats.append(y * y * y)
            elif label == "r^2*x":  feats.append((x * x + y * y) * x)
            elif label == "r^2*y":  feats.append((x * x + y * y) * y)
            else: raise ValueError(f"Unknown basis term: {label}")
        return np.array(feats, dtype=np.float64)
    return basis


class TagState:
    def __init__(self, node: Node, marker_id: int, pose_topic: str):
        self.marker_id = marker_id
        self.pose_topic = pose_topic
        self.publisher = node.create_publisher(PoseStamped, pose_topic, 10)
        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None
        self.last_yaw: Optional[float] = None
        self.published_count = 0
        self.rejected_count = 0


class MultiStagDetector(Node):
    def __init__(self):
        super().__init__("multi_stag_detector")

        self.declare_parameter(
            "image_topic",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/image",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/camera_info",
        )
        self.declare_parameter("target_ids", [0, 1, 2, 3, 4, 5])
        self.declare_parameter("pose_topic_prefix", "/stag_pose_")

        self.declare_parameter("library_hd", 11)
        self.declare_parameter("error_correction", -1)

        self.declare_parameter("marker_size_m", 0.40)
        self.declare_parameter("x_offset", 0.0)
        self.declare_parameter("y_offset", 0.0)
        self.declare_parameter("yaw_offset", 1.5708)
        self.declare_parameter("tag_offset_forward", -0.032)
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
        self.target_ids: List[int] = [int(v) for v in self.get_parameter("target_ids").value]
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

        # ---- calibration: poly or knn or off ----
        self.calib_mode = "off"
        self.calib_basis = None
        self.calib_coef_x = None
        self.calib_coef_y = None
        self.knn_X = None
        self.knn_err_x = None
        self.knn_err_y = None
        self.knn_k = None

        calibration_file = str(self.get_parameter("calibration_file").value).strip()
        if calibration_file:
            path = Path(calibration_file)
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    model_kind = data.get("model", "")
                    if model_kind == "knn":
                        self.knn_X = np.column_stack([
                            np.array(data["gt_x"], dtype=np.float64),
                            np.array(data["gt_y"], dtype=np.float64),
                        ])
                        self.knn_err_x = np.array(data["err_x"], dtype=np.float64)
                        self.knn_err_y = np.array(data["err_y"], dtype=np.float64)
                        self.knn_k = int(data.get("k_neighbors", 50))
                        self.calib_mode = "knn"
                        self.get_logger().info(
                            f"k-NN calibration loaded: {len(self.knn_X)} points, k={self.knn_k}"
                        )
                    elif model_kind == "cubic_xy_with_radial":
                        self.calib_basis = make_basis_func(data["basis"])
                        self.calib_coef_x = np.array(data["coef_x"], dtype=np.float64)
                        self.calib_coef_y = np.array(data["coef_y"], dtype=np.float64)
                        self.calib_mode = "poly"
                        self.get_logger().info(
                            f"Polynomial calibration loaded: {len(data['basis'])} terms"
                        )
                    else:
                        self.get_logger().warn(
                            f"Unknown calibration model '{model_kind}', running uncalibrated"
                        )
                except Exception as e:
                    self.get_logger().error(f"Failed to load calibration: {e}")
            else:
                self.get_logger().warn(
                    f"Calibration file not found: {path}, running uncalibrated"
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

        self.tag_states: Dict[int, TagState] = {}
        for mid in self.target_ids:
            topic = f"{self.pose_topic_prefix}{mid}"
            self.tag_states[mid] = TagState(self, mid, topic)
            self.get_logger().info(f"Tracking marker id={mid} -> {topic}")

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10
        )

        self.frame_count = 0

        self.get_logger().info(
            f"Multi-STag detector started for ids={self.target_ids}, "
            f"libraryHD={self.library_hd}, marker_size={self.marker_size_m} m"
        )
        self.get_logger().info(
            f"Tag->base compensation: forward={self.tag_offset_forward:+.4f} m, "
            f"lateral={self.tag_offset_lateral:+.4f} m"
        )
        self.get_logger().info(f"Calibration mode: {self.calib_mode.upper()}")

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float32)

    def is_outlier(self, st: TagState, x: float, y: float, yaw: float) -> bool:
        if st.last_x is None:
            return False
        dist = math.hypot(x - st.last_x, y - st.last_y)
        dyaw = abs(wrap_to_pi(yaw - st.last_yaw))
        return dist > self.max_position_jump or dyaw > self.max_yaw_jump

    def filter_pose(self, st: TagState, x: float, y: float, yaw: float):
        if st.last_x is None:
            st.last_x, st.last_y, st.last_yaw = x, y, yaw
            return x, y, yaw
        ax = self.filter_alpha_pos
        ayaw = self.filter_alpha_yaw
        x_f = ax * x + (1.0 - ax) * st.last_x
        y_f = ax * y + (1.0 - ax) * st.last_y
        dyaw = wrap_to_pi(yaw - st.last_yaw)
        yaw_f = wrap_to_pi(st.last_yaw + ayaw * dyaw)
        st.last_x, st.last_y, st.last_yaw = x_f, y_f, yaw_f
        return x_f, y_f, yaw_f

    def apply_calibration(self, x: float, y: float):
        if self.calib_mode == "poly":
            feats = self.calib_basis(x, y)
            err_x = float(np.dot(feats, self.calib_coef_x))
            err_y = float(np.dot(feats, self.calib_coef_y))
            return x - err_x, y - err_y
        if self.calib_mode == "knn":
            d2 = (self.knn_X[:, 0] - x) ** 2 + (self.knn_X[:, 1] - y) ** 2
            k = min(self.knn_k, len(d2))
            idx = np.argpartition(d2, k - 1)[:k]
            dists = np.sqrt(d2[idx])
            w = 1.0 / (dists + 1e-9)
            w_sum = w.sum()
            err_x = float(np.dot(w, self.knn_err_x[idx]) / w_sum)
            err_y = float(np.dot(w, self.knn_err_y[idx]) / w_sum)
            return x - err_x, y - err_y
        return x, y

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

    def process_one_marker(self, st: TagState, corners_one, msg_stamp):
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

        if self.is_outlier(st, x, y, yaw):
            st.rejected_count += 1
            if st.rejected_count % max(1, self.log_every_n) == 1:
                self.get_logger().warn(
                    f"id={st.marker_id} outlier rejected (total: {st.rejected_count})"
                )
            return

        x, y, yaw = self.filter_pose(st, x, y, yaw)

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

        st.publisher.publish(pose)
        st.published_count += 1

        if st.published_count % max(1, self.log_every_n) == 1:
            self.get_logger().info(
                f"id={st.marker_id}: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} "
                f"(published: {st.published_count})"
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
        for tracked_id, st in self.tag_states.items():
            matches = np.where(ids_flat == tracked_id)[0]
            if len(matches) == 0:
                continue
            idx = int(matches[0])
            try:
                self.process_one_marker(st, corners[idx], msg.header.stamp)
            except Exception as e:
                self.get_logger().error(f"Processing id={tracked_id} failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MultiStagDetector()
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