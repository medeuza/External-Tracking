import json
import math
from pathlib import Path
from typing import Optional, List, Callable

import cv2
import numpy as np
import rclpy

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
            if label == "1":      feats.append(1.0)
            elif label == "x":    feats.append(x)
            elif label == "y":    feats.append(y)
            elif label == "x^2":  feats.append(x * x)
            elif label == "x*y":  feats.append(x * y)
            elif label == "y^2":  feats.append(y * y)
            elif label == "x^3":  feats.append(x * x * x)
            elif label == "x^2*y":feats.append(x * x * y)
            elif label == "x*y^2":feats.append(x * y * y)
            elif label == "y^3":  feats.append(y * y * y)
            elif label == "r^2*x":feats.append((x * x + y * y) * x)
            elif label == "r^2*y":feats.append((x * x + y * y) * y)
            else: raise ValueError(f"Unknown basis term: {label}")
        return np.array(feats, dtype=np.float64)
    return basis


ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}

ARUCO_NEW_API = hasattr(cv2.aruco, "ArucoDetector")


def _select_pnp_flag():
    for name in ["SOLVEPNP_SQPNP", "SOLVEPNP_IPPE", "SOLVEPNP_ITERATIVE"]:
        if hasattr(cv2, name):
            return getattr(cv2, name), name
    return cv2.SOLVEPNP_ITERATIVE, "SOLVEPNP_ITERATIVE"

PNP_FLAG, PNP_FLAG_NAME = _select_pnp_flag()


def make_aruco_dictionary(dict_id):
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def make_aruco_detector(aruco_dict):
    if ARUCO_NEW_API:
        params = cv2.aruco.DetectorParameters()
        try:
            if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
            else:
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
            params.cornerRefinementMaxIterations = 50
            params.cornerRefinementMinAccuracy = 0.01
        except Exception:
            pass
        return cv2.aruco.ArucoDetector(aruco_dict, params), params
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()
    try:
        if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        else:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    except Exception:
        pass
    return None, params


def detect_markers(frame, aruco_dict, detector, params):
    if ARUCO_NEW_API:
        return detector.detectMarkers(frame)
    return cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)


def build_board_object_points(grid_cols, grid_rows, marker_size_m,
                               marker_separation_m, board_ids):
    s = marker_size_m / 2.0
    pitch = marker_size_m + marker_separation_m
    cx0 = (grid_cols - 1) / 2.0
    cy0 = (grid_rows - 1) / 2.0
    obj_by_id = {}
    for local_idx, mid in enumerate(board_ids):
        row = local_idx // grid_cols
        col = local_idx % grid_cols
        mx = (col - cx0) * pitch
        my = (cy0 - row) * pitch
        corners = np.array([
            [mx - s, my + s, 0.0],
            [mx + s, my + s, 0.0],
            [mx + s, my - s, 0.0],
            [mx - s, my - s, 0.0],
        ], dtype=np.float32)
        obj_by_id[int(mid)] = corners
    return obj_by_id


def make_grid_board(grid_cols, grid_rows, marker_size_m,
                    marker_separation_m, aruco_dict, board_ids):
    if ARUCO_NEW_API and hasattr(cv2.aruco, "GridBoard"):
        try:
            return cv2.aruco.GridBoard(
                (grid_cols, grid_rows),
                marker_size_m,
                marker_separation_m,
                aruco_dict,
                board_ids,
            )
        except Exception:
            pass
    return build_board_object_points(
        grid_cols, grid_rows, marker_size_m, marker_separation_m, board_ids
    )


class MultiBoardDetector(Node):
    def __init__(self):
        super().__init__("multiboard_detector")

        self.declare_parameter(
            "image_topic",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/image",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/camera_info",
        )
        self.declare_parameter("target_board_id", 0)
        self.declare_parameter("pose_topic_prefix", "/board_pose_")

        self.declare_parameter("dictionary", "DICT_4X4_50")
        self.declare_parameter("grid_rows", 2)
        self.declare_parameter("grid_cols", 2)
        self.declare_parameter("markers_per_board", 4)
        self.declare_parameter("start_id", 0)

        self.declare_parameter("marker_size_m", 0.208333)
        self.declare_parameter("marker_separation_m", 0.041667)

        self.declare_parameter("base_z", 0.01)
        self.declare_parameter("marker_z", 0.235)
        self.declare_parameter("marker_offset_x_base", -0.032)
        self.declare_parameter("marker_offset_y_base", 0.035)
        self.declare_parameter("marker_offset_z_base", 0.235)
        self.declare_parameter("world_bias_x", 0.0)
        self.declare_parameter("world_bias_y", 0.0)

        self.declare_parameter("filter_alpha_pos", 1.0)
        self.declare_parameter("filter_alpha_yaw", 1.0)
        self.declare_parameter("max_position_jump", 0.80)
        self.declare_parameter("max_yaw_jump", 0.75)
        self.declare_parameter("max_reproj_error_px", 5.0)

        self.declare_parameter(
            "calibration_file",
            str(Path.home() / "board_correction.json"),
        )

        self.declare_parameter("log_every_n", 60)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.target_board_id = int(self.get_parameter("target_board_id").value)
        self.pose_topic_prefix = str(self.get_parameter("pose_topic_prefix").value)

        self.dictionary_name = str(self.get_parameter("dictionary").value)
        self.grid_rows = int(self.get_parameter("grid_rows").value)
        self.grid_cols = int(self.get_parameter("grid_cols").value)
        self.markers_per_board = int(self.get_parameter("markers_per_board").value)
        self.start_id = int(self.get_parameter("start_id").value)

        self.marker_size_m = float(self.get_parameter("marker_size_m").value)
        self.marker_separation_m = float(self.get_parameter("marker_separation_m").value)

        self.base_z = float(self.get_parameter("base_z").value)
        self.marker_z = float(self.get_parameter("marker_z").value)
        self.marker_offset_in_base = np.array(
            [
                [float(self.get_parameter("marker_offset_x_base").value)],
                [float(self.get_parameter("marker_offset_y_base").value)],
                [float(self.get_parameter("marker_offset_z_base").value)],
            ],
            dtype=np.float64,
        )

        self.world_bias_x = float(self.get_parameter("world_bias_x").value)
        self.world_bias_y = float(self.get_parameter("world_bias_y").value)

        self.filter_alpha_pos = float(self.get_parameter("filter_alpha_pos").value)
        self.filter_alpha_yaw = float(self.get_parameter("filter_alpha_yaw").value)
        self.max_position_jump = float(self.get_parameter("max_position_jump").value)
        self.max_yaw_jump = float(self.get_parameter("max_yaw_jump").value)
        self.max_reproj_error_px = float(self.get_parameter("max_reproj_error_px").value)
        self.log_every_n = int(self.get_parameter("log_every_n").value)

        self.R_mb = np.eye(3, dtype=np.float64)
        self.R_wc = np.array(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )

        if self.dictionary_name not in ARUCO_DICTS:
            raise ValueError(
                f"Unknown dictionary '{self.dictionary_name}'. "
                f"Known: {sorted(ARUCO_DICTS)}"
            )
        self.aruco_dict = make_aruco_dictionary(ARUCO_DICTS[self.dictionary_name])
        self.aruco_detector, self.detector_params = make_aruco_detector(self.aruco_dict)

        first_id = self.start_id + self.target_board_id * self.markers_per_board
        self.board_ids = np.array(
            [first_id + i for i in range(self.markers_per_board)], dtype=np.int32
        )

        self.grid_board = make_grid_board(
            self.grid_cols, self.grid_rows,
            self.marker_size_m, self.marker_separation_m,
            self.aruco_dict, self.board_ids,
        )

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

        topic = f"{self.pose_topic_prefix}{self.target_board_id}"
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

        self.get_logger().info(
            f"MultiBoard detector started for board_id={self.target_board_id} -> {topic}"
        )
        self.get_logger().info(
            f"PnP solver: {PNP_FLAG_NAME}"
        )
        refine_method = "APRILTAG" if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG") else "SUBPIX"
        self.get_logger().info(
            f"Corner refinement: {refine_method}"
        )
        self.get_logger().info(
            f"Filter: alpha_pos={self.filter_alpha_pos}, alpha_yaw={self.filter_alpha_yaw} "
            f"(1.0 = OFF)"
        )
        self.get_logger().info(
            f"Post-calibration: {'ENABLED' if self.calib_enabled else 'disabled'}"
        )

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def is_outlier(self, x: float, y: float, yaw: float) -> bool:
        if self.last_x is None:
            return False
        dist = math.hypot(x - self.last_x, y - self.last_y)
        dyaw = abs(wrap_to_pi(yaw - self.last_yaw))
        return dist > self.max_position_jump or dyaw > self.max_yaw_jump

    def filter_pose(self, x: float, y: float, yaw: float):
        if self.filter_alpha_pos >= 0.999 and self.filter_alpha_yaw >= 0.999:
            self.last_x, self.last_y, self.last_yaw = x, y, yaw
            return x, y, yaw
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

    def estimate_pose_from_board(self, corners, ids):
        ids_flat = np.asarray(ids).flatten()
        keep = [i for i, mid in enumerate(ids_flat) if mid in self.board_ids]
        if not keep:
            raise RuntimeError("no markers of target board visible")

        board_corners = [corners[i] for i in keep]
        board_marker_ids = ids_flat[keep].reshape(-1, 1)

        if ARUCO_NEW_API:
            obj_points, img_points = self.grid_board.matchImagePoints(
                board_corners, board_marker_ids
            )
            if obj_points is None or len(obj_points) < 4:
                raise RuntimeError("not enough matched points for board pose")

            ok, rvec, tvec = cv2.solvePnP(
                obj_points, img_points,
                self.camera_matrix, self.dist_coeffs,
                flags=PNP_FLAG,
            )
            if not ok:
                raise RuntimeError("solvePnP failed")
            return (rvec.reshape(3, 1), tvec.reshape(3, 1), len(keep),
                    obj_points, img_points)

        obj_pts, img_pts = [], []
        for det_idx in keep:
            mid = int(ids_flat[det_idx])
            if mid not in self.grid_board:
                continue
            marker_img = np.asarray(corners[det_idx]).reshape(4, 2)
            obj_pts.append(self.grid_board[mid])
            img_pts.append(marker_img)

        if len(obj_pts) < 1:
            raise RuntimeError("no usable markers for board pose")

        obj_pts = np.concatenate(obj_pts, axis=0).astype(np.float32)
        img_pts = np.concatenate(img_pts, axis=0).astype(np.float32)

        if obj_pts.shape[0] < 4:
            raise RuntimeError("not enough points for solvePnP")

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts,
            self.camera_matrix, self.dist_coeffs,
            flags=PNP_FLAG,
        )
        if not ok:
            raise RuntimeError("solvePnP failed")
        return (rvec.reshape(3, 1), tvec.reshape(3, 1), len(keep),
                obj_pts, img_pts)

    def process_board(self, corners, ids, msg_stamp):
        rvec, tvec, n_used, obj_pts, img_pts = self.estimate_pose_from_board(
            corners, ids
        )

        projected, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        reproj_err = float(
            np.mean(np.linalg.norm(
                projected.reshape(-1, 2) - img_pts.reshape(-1, 2),
                axis=1,
            ))
        )
        if reproj_err > self.max_reproj_error_px:
            self.rejected_count += 1
            if self.rejected_count % max(1, self.log_every_n) == 1:
                self.get_logger().warn(
                    f"board_id={self.target_board_id} high reprojection "
                    f"error {reproj_err:.2f}px, rejected "
                    f"(total: {self.rejected_count})"
                )
            return

        R_cm, _ = cv2.Rodrigues(rvec)
        R_cb = R_cm @ self.R_mb
        base_pos_cam = tvec - R_cb @ self.marker_offset_in_base
        R_wb = self.R_wc @ R_cb

        x = -float(base_pos_cam[1, 0]) - self.world_bias_x
        y = -float(base_pos_cam[0, 0]) - self.world_bias_y
        yaw = wrap_to_pi(math.atan2(R_wb[1, 0], R_wb[0, 0]))

        x, y = self.apply_calibration(x, y)

        if self.is_outlier(x, y, yaw):
            self.rejected_count += 1
            if self.rejected_count % max(1, self.log_every_n) == 1:
                self.get_logger().warn(
                    f"board_id={self.target_board_id} outlier rejected "
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
        pose.pose.position.z = float(self.base_z)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)

        self.publisher.publish(pose)
        self.published_count += 1

        if self.published_count % max(1, self.log_every_n) == 1:
            self.get_logger().info(
                f"board_id={self.target_board_id}: x={x:.3f}, y={y:.3f}, "
                f"yaw={yaw:.3f}, markers={n_used}, "
                f"reproj={reproj_err:.2f}px "
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
            corners, ids, _ = detect_markers(
                frame, self.aruco_dict, self.aruco_detector, self.detector_params
            )
        except Exception as e:
            self.get_logger().error(f"detectMarkers failed: {e}")
            return

        if ids is None or len(ids) == 0:
            return

        ids_flat = np.asarray(ids).flatten()
        if not np.any(np.isin(ids_flat, self.board_ids)):
            return

        try:
            self.process_board(corners, ids, msg.header.stamp)
        except Exception as e:
            self.get_logger().error(
                f"Processing board_id={self.target_board_id} failed: {e}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = MultiBoardDetector()
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