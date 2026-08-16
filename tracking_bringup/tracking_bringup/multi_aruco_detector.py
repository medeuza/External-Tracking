import json
import math
from pathlib import Path
from typing import Optional, Dict, List, Callable

import cv2
import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from tf_transformations import quaternion_from_matrix


def norm_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float):
    half = yaw / 2.0
    return 0.0, 0.0, math.sin(half), math.cos(half)


def pose_from_R_t(R: np.ndarray, t: np.ndarray, stamp, frame_id: str) -> PoseStamped:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    qx, qy, qz, qw = quaternion_from_matrix(T)

    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(t[0, 0])
    msg.pose.position.y = float(t[1, 0])
    msg.pose.position.z = float(t[2, 0])
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg


def make_basis_func(basis_labels: List[str]) -> Callable[[float, float], np.ndarray]:
    """Build a function that maps (x, y) -> basis row using the labels."""
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


class TagState:

    def __init__(self, node: Node, marker_id: int,
                 base_topic: str, marker_center_topic: str):
        self.marker_id = marker_id
        self.base_pub = node.create_publisher(PoseStamped, base_topic, 10)
        self.marker_center_pub = node.create_publisher(
            PoseStamped, marker_center_topic, 10)

        self.last_base_world_msg: Optional[PoseStamped] = None
        self.last_marker_world_msg: Optional[PoseStamped] = None
        self.last_filtered_base_pos: Optional[np.ndarray] = None
        self.last_filtered_base_yaw: Optional[float] = None
        self.last_filtered_marker_pos: Optional[np.ndarray] = None
        self.last_filtered_marker_yaw: Optional[float] = None

        self.published_count = 0
        self.rejected_count = 0
        self.logged_first = False


class MultiArucoDetector(Node):
    def __init__(self):
        super().__init__("multi_aruco_detector")

        self.declare_parameter(
            "image_topic",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/image",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/camera_info",
        )
        self.declare_parameter("target_ids", [0, 1, 2, 3, 4, 5])
        self.declare_parameter("pose_topic_prefix", "/aruco_pose_")
        self.declare_parameter("marker_center_topic_prefix", "/aruco_marker_center_")

        self.declare_parameter("marker_length", 0.4807692308)
        self.declare_parameter("marker_offset_x", -0.032)
        self.declare_parameter("marker_offset_y", 0.0)
        self.declare_parameter("marker_offset_z", 0.235)
        self.declare_parameter("world_bias_x", 0.0)
        self.declare_parameter("world_bias_y", 0.0)
        self.declare_parameter("base_z", 0.01)
        self.declare_parameter("marker_z", 0.235)

        self.declare_parameter("max_position_jump", 0.10)
        self.declare_parameter("max_yaw_jump", 0.40)
        self.declare_parameter("max_reproj_error_px", 2.5)
        self.declare_parameter("filter_alpha_pos", 0.45)
        self.declare_parameter("filter_alpha_yaw", 0.45)


        self.declare_parameter(
            "calibration_file",
            str(Path.home() / "aruco_correction.json"),
        )

        self.declare_parameter("log_every_n", 60)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.target_ids: List[int] = [
            int(v) for v in self.get_parameter("target_ids").value
        ]
        self.pose_topic_prefix = str(self.get_parameter("pose_topic_prefix").value)
        self.marker_center_topic_prefix = str(
            self.get_parameter("marker_center_topic_prefix").value)

        self.marker_length = float(self.get_parameter("marker_length").value)
        self.world_bias_x = float(self.get_parameter("world_bias_x").value)
        self.world_bias_y = float(self.get_parameter("world_bias_y").value)
        self.base_z = float(self.get_parameter("base_z").value)
        self.marker_z = float(self.get_parameter("marker_z").value)

        self.max_position_jump = float(self.get_parameter("max_position_jump").value)
        self.max_yaw_jump = float(self.get_parameter("max_yaw_jump").value)
        self.max_reproj_error_px = float(
            self.get_parameter("max_reproj_error_px").value)
        self.filter_alpha_pos = float(self.get_parameter("filter_alpha_pos").value)
        self.filter_alpha_yaw = float(self.get_parameter("filter_alpha_yaw").value)
        self.log_every_n = int(self.get_parameter("log_every_n").value)

        calibration_file = str(
            self.get_parameter("calibration_file").value).strip()
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
                        f"Calibration loaded: model={data.get('model', '?')}, "
                        f"{len(basis_labels)} terms"
                    )
                    stats = data.get("stats_after", {})
                    if stats:
                        self.get_logger().info(
                            f"  Reported residual: "
                            f"MAE={stats.get('mae_total_cm', '?')} cm, "
                            f"Max={stats.get('max_total_cm', '?')} cm"
                        )
                except Exception as e:
                    self.get_logger().error(
                        f"Failed to load calibration: {e}")
            else:
                self.get_logger().warn(
                    f"Calibration file not found: {path}, "
                    f"running without correction"
                )

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self.parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.parameters = cv2.aruco.DetectorParameters()
        try:
            self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except Exception:
            pass

        self.marker_offset_in_base = np.array(
            [
                [float(self.get_parameter("marker_offset_x").value)],
                [float(self.get_parameter("marker_offset_y").value)],
                [float(self.get_parameter("marker_offset_z").value)],
            ],
            dtype=np.float64,
        )
        self.R_mb = np.eye(3, dtype=np.float64)
        self.R_wc = np.array(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )

        self.tag_states: Dict[int, TagState] = {}
        for mid in self.target_ids:
            base_topic = f"{self.pose_topic_prefix}{mid}"
            mc_topic = f"{self.marker_center_topic_prefix}{mid}"
            self.tag_states[mid] = TagState(self, mid, base_topic, mc_topic)
            self.get_logger().info(f"Tracking marker id={mid} -> {base_topic}")

        self.frame_count = 0
        self.get_logger().info(
            f"Multi-ArUco detector started for ids={self.target_ids}"
        )
        self.get_logger().info(
            f"marker_length={self.marker_length:.4f} m, dict=DICT_4X4_50"
        )
        self.get_logger().info(
            f"Post-calibration: "
            f"{'ENABLED' if self.calib_enabled else 'disabled'}"
        )


    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        if self.frame_count == 0:
            self.get_logger().info(
                f"Camera matrix loaded. fx={self.camera_matrix[0,0]:.1f}, "
                f"fy={self.camera_matrix[1,1]:.1f}, "
                f"cx={self.camera_matrix[0,2]:.1f}, cy={self.camera_matrix[1,2]:.1f}"
            )

    def apply_calibration(self, x: float, y: float):
        """Subtract predicted systematic error using the loaded model."""
        if not self.calib_enabled:
            return x, y
        feats = self.calib_basis(x, y)
        err_x = float(np.dot(feats, self.calib_coef_x))
        err_y = float(np.dot(feats, self.calib_coef_y))
        return x - err_x, y - err_y

    def is_outlier(self, pose_msg: PoseStamped,
                   last_pose_msg: Optional[PoseStamped]) -> bool:
        if last_pose_msg is None:
            return False
        dx = pose_msg.pose.position.x - last_pose_msg.pose.position.x
        dy = pose_msg.pose.position.y - last_pose_msg.pose.position.y
        dist = math.sqrt(dx * dx + dy * dy)
        dyaw = abs(norm_angle(yaw_from_quat(pose_msg.pose.orientation)
                              - yaw_from_quat(last_pose_msg.pose.orientation)))
        return dist > self.max_position_jump or dyaw > self.max_yaw_jump

    def filter_pose(self, pose_msg: PoseStamped,
                    last_pos: Optional[np.ndarray],
                    last_yaw: Optional[float]):
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        z = pose_msg.pose.position.z
        yaw = yaw_from_quat(pose_msg.pose.orientation)

        if last_pos is None or last_yaw is None:
            return pose_msg, np.array([x, y, z], dtype=np.float64), yaw

        filtered_pos = (self.filter_alpha_pos * np.array([x, y, z], dtype=np.float64)
                        + (1.0 - self.filter_alpha_pos) * last_pos)
        dyaw = norm_angle(yaw - last_yaw)
        filtered_yaw = norm_angle(last_yaw + self.filter_alpha_yaw * dyaw)

        pose_msg.pose.position.x = float(filtered_pos[0])
        pose_msg.pose.position.y = float(filtered_pos[1])
        pose_msg.pose.position.z = float(filtered_pos[2])
        qx, qy, qz, qw = yaw_to_quaternion(filtered_yaw)
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        return pose_msg, filtered_pos, filtered_yaw

    def process_one_marker(self, st: TagState, marker_corners, gray, stamp):
        marker_corners = np.asarray(marker_corners, dtype=np.float32)
        if marker_corners.ndim == 3:
            marker_corners = marker_corners[0]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        marker_corners_refined = cv2.cornerSubPix(
            gray,
            marker_corners.reshape(-1, 1, 2),
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=criteria,
        ).reshape(4, 2)

        half = self.marker_length / 2.0
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            marker_corners_refined,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return

        rvec = rvec.reshape(3, 1)
        tvec = tvec.reshape(3, 1)

        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        reproj_err = float(np.mean(np.linalg.norm(
            projected.reshape(-1, 2) - marker_corners_refined, axis=1)))
        if reproj_err > self.max_reproj_error_px:
            st.rejected_count += 1
            return

        R_cm, _ = cv2.Rodrigues(rvec)

        marker_center_world = np.array(
            [
                [-float(tvec[1, 0]) - self.world_bias_x],
                [-float(tvec[0, 0]) - self.world_bias_y],
                [self.marker_z],
            ],
            dtype=np.float64,
        )
        R_wm = self.R_wc @ R_cm
        marker_center_world_msg = pose_from_R_t(R_wm, marker_center_world,
                                                stamp, "world")
        if not self.is_outlier(marker_center_world_msg, st.last_marker_world_msg):
            (marker_center_world_msg,
             st.last_filtered_marker_pos,
             st.last_filtered_marker_yaw) = self.filter_pose(
                marker_center_world_msg,
                st.last_filtered_marker_pos,
                st.last_filtered_marker_yaw,
            )
            st.marker_center_pub.publish(marker_center_world_msg)
            st.last_marker_world_msg = marker_center_world_msg

        R_cb = R_cm @ self.R_mb
        base_pos_cam = tvec - R_cb @ self.marker_offset_in_base
        base_x_world = -float(base_pos_cam[1, 0]) - self.world_bias_x
        base_y_world = -float(base_pos_cam[0, 0]) - self.world_bias_y

        base_x_world, base_y_world = self.apply_calibration(
            base_x_world, base_y_world)

        base_pos_world = np.array(
            [[base_x_world], [base_y_world], [self.base_z]],
            dtype=np.float64,
        )
        R_wb = self.R_wc @ R_cb
        base_world_msg = pose_from_R_t(R_wb, base_pos_world, stamp, "world")

        if self.is_outlier(base_world_msg, st.last_base_world_msg):
            st.rejected_count += 1
            if st.rejected_count % max(1, self.log_every_n) == 1:
                self.get_logger().warn(
                    f"id={st.marker_id} outlier rejected "
                    f"(total: {st.rejected_count})"
                )
            return

        (base_world_msg,
         st.last_filtered_base_pos,
         st.last_filtered_base_yaw) = self.filter_pose(
            base_world_msg,
            st.last_filtered_base_pos,
            st.last_filtered_base_yaw,
        )
        st.base_pub.publish(base_world_msg)
        st.last_base_world_msg = base_world_msg
        st.published_count += 1

        if not st.logged_first:
            self.get_logger().info(
                f"id={st.marker_id} first base WORLD: "
                f"x={base_world_msg.pose.position.x:.3f}, "
                f"y={base_world_msg.pose.position.y:.3f}, "
                f"yaw={yaw_from_quat(base_world_msg.pose.orientation):.3f}"
            )
            st.logged_first = True
        elif st.published_count % max(1, self.log_every_n) == 1:
            self.get_logger().info(
                f"id={st.marker_id}: "
                f"x={base_world_msg.pose.position.x:.3f}, "
                f"y={base_world_msg.pose.position.y:.3f} "
                f"(published: {st.published_count})"
            )


    def image_callback(self, msg: Image):
        self.frame_count += 1
        if self.camera_matrix is None or self.dist_coeffs is None:
            if self.frame_count % max(1, self.log_every_n) == 1:
                self.get_logger().warn("Waiting for camera_info...")
            return

        data = np.frombuffer(msg.data, dtype=np.uint8)
        frame = data.reshape((msg.height, msg.width, 3))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.parameters)
        if ids is None or len(ids) == 0:
            return

        ids_flat = ids.flatten()
        for tracked_id, st in self.tag_states.items():
            matches = np.where(ids_flat == tracked_id)[0]
            if len(matches) == 0:
                continue
            idx = int(matches[0])
            try:
                self.process_one_marker(st, corners[idx], gray, msg.header.stamp)
            except Exception as e:
                self.get_logger().error(f"Processing id={tracked_id} failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MultiArucoDetector()
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