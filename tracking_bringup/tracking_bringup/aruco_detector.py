import math
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from tf_transformations import quaternion_from_matrix


def norm_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pose_from_R_t(R: np.ndarray, t: np.ndarray, header, frame_id: str) -> PoseStamped:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    qx, qy, qz, qw = quaternion_from_matrix(T)

    msg = PoseStamped()
    msg.header = header
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(t[0, 0])
    msg.pose.position.y = float(t[1, 0])
    msg.pose.position.z = float(t[2, 0])
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg


def yaw_from_pose(msg: PoseStamped) -> float:
    q = msg.pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ArucoDetector(Node):
    def __init__(self):
        super().__init__("aruco_detector")

        self.declare_parameter(
            "image_topic",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/image",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/camera_info",
        )
        self.declare_parameter("marker_length", 0.4807692308)
        self.declare_parameter("marker_id_to_track", 0)
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
        self.declare_parameter("save_debug_image", False)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.marker_length = float(self.get_parameter("marker_length").value)
        self.marker_id_to_track = int(self.get_parameter("marker_id_to_track").value)
        self.world_bias_x = float(self.get_parameter("world_bias_x").value)
        self.world_bias_y = float(self.get_parameter("world_bias_y").value)
        self.base_z = float(self.get_parameter("base_z").value)
        self.marker_z = float(self.get_parameter("marker_z").value)
        self.max_position_jump = float(self.get_parameter("max_position_jump").value)
        self.max_yaw_jump = float(self.get_parameter("max_yaw_jump").value)
        self.max_reproj_error_px = float(self.get_parameter("max_reproj_error_px").value)
        self.filter_alpha_pos = float(self.get_parameter("filter_alpha_pos").value)
        self.filter_alpha_yaw = float(self.get_parameter("filter_alpha_yaw").value)
        self.save_debug_image = bool(self.get_parameter("save_debug_image").value)

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.marker_center_raw_pub = self.create_publisher(PoseStamped, "/aruco_marker_center_raw", 10)
        self.base_raw_pub = self.create_publisher(PoseStamped, "/aruco_pose_raw", 10)
        self.marker_center_pub = self.create_publisher(PoseStamped, "/aruco_marker_center", 10)
        self.base_pub = self.create_publisher(PoseStamped, "/aruco_pose", 10)

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()
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

        self.frame_count = 0
        self.logged_first_debug = False
        self.last_base_world_msg: Optional[PoseStamped] = None
        self.last_marker_world_msg: Optional[PoseStamped] = None
        self.last_filtered_base_pos: Optional[np.ndarray] = None
        self.last_filtered_base_yaw: Optional[float] = None
        self.last_filtered_marker_pos: Optional[np.ndarray] = None
        self.last_filtered_marker_yaw: Optional[float] = None

    def quaternion_to_yaw(self, q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw: float):
        half = yaw / 2.0
        return 0.0, 0.0, math.sin(half), math.cos(half)

    def filter_pose(
        self,
        pose_msg: PoseStamped,
        last_pos: Optional[np.ndarray],
        last_yaw: Optional[float],
    ):
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        z = pose_msg.pose.position.z
        yaw = self.quaternion_to_yaw(pose_msg.pose.orientation)

        if last_pos is None or last_yaw is None:
            return pose_msg, np.array([x, y, z], dtype=np.float64), yaw

        filtered_pos = self.filter_alpha_pos * np.array([x, y, z], dtype=np.float64) + (1.0 - self.filter_alpha_pos) * last_pos
        dyaw = norm_angle(yaw - last_yaw)
        filtered_yaw = norm_angle(last_yaw + self.filter_alpha_yaw * dyaw)

        pose_msg.pose.position.x = float(filtered_pos[0])
        pose_msg.pose.position.y = float(filtered_pos[1])
        pose_msg.pose.position.z = float(filtered_pos[2])
        qx, qy, qz, qw = self.yaw_to_quaternion(filtered_yaw)
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        return pose_msg, filtered_pos, filtered_yaw

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        if self.frame_count == 0:
            self.get_logger().info(
                f"Camera matrix loaded. fx={self.camera_matrix[0,0]:.3f}, fy={self.camera_matrix[1,1]:.3f}, cx={self.camera_matrix[0,2]:.3f}, cy={self.camera_matrix[1,2]:.3f}"
            )

    def image_callback(self, msg: Image):
        self.frame_count += 1
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        data = np.frombuffer(msg.data, dtype=np.uint8)
        frame = data.reshape((msg.height, msg.width, 3))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        if ids is None or len(ids) == 0:
            return

        ids = ids.flatten()
        found = np.where(ids == self.marker_id_to_track)[0]
        if len(found) == 0:
            return

        idx = int(found[0])
        marker_corners = corners[idx].astype(np.float32)
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
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        reproj_err = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - marker_corners_refined, axis=1)))
        if reproj_err > self.max_reproj_error_px:
            return

        R_cm, _ = cv2.Rodrigues(rvec)
        marker_center_raw_msg = pose_from_R_t(R_cm, tvec, msg.header, "overhead_camera")
        self.marker_center_raw_pub.publish(marker_center_raw_msg)

        marker_center_world = np.array(
            [[-float(tvec[1, 0]) - self.world_bias_x], [-float(tvec[0, 0]) - self.world_bias_y], [self.marker_z]],
            dtype=np.float64,
        )
        R_wm = self.R_wc @ R_cm
        marker_center_world_msg = pose_from_R_t(R_wm, marker_center_world, msg.header, "world")
        if not self.is_outlier(marker_center_world_msg, self.last_marker_world_msg):
            marker_center_world_msg, self.last_filtered_marker_pos, self.last_filtered_marker_yaw = self.filter_pose(
                marker_center_world_msg,
                self.last_filtered_marker_pos,
                self.last_filtered_marker_yaw,
            )
            self.marker_center_pub.publish(marker_center_world_msg)
            self.last_marker_world_msg = marker_center_world_msg

        R_cb = R_cm @ self.R_mb
        base_pos_cam = tvec - R_cb @ self.marker_offset_in_base
        base_raw_msg = pose_from_R_t(R_cb, base_pos_cam, msg.header, "overhead_camera")
        self.base_raw_pub.publish(base_raw_msg)

        base_pos_world = np.array(
            [[-float(base_pos_cam[1, 0]) - self.world_bias_x], [-float(base_pos_cam[0, 0]) - self.world_bias_y], [self.base_z]],
            dtype=np.float64,
        )
        R_wb = self.R_wc @ R_cb
        base_world_msg = pose_from_R_t(R_wb, base_pos_world, msg.header, "world")
        if self.is_outlier(base_world_msg, self.last_base_world_msg):
            return

        base_world_msg, self.last_filtered_base_pos, self.last_filtered_base_yaw = self.filter_pose(
            base_world_msg,
            self.last_filtered_base_pos,
            self.last_filtered_base_yaw,
        )
        self.base_pub.publish(base_world_msg)
        self.last_base_world_msg = base_world_msg

        if not self.logged_first_debug:
            self.get_logger().info(
                f"First base WORLD: x={base_world_msg.pose.position.x:.4f}, y={base_world_msg.pose.position.y:.4f}, z={base_world_msg.pose.position.z:.4f}, yaw={yaw_from_pose(base_world_msg):.4f}"
            )
            self.logged_first_debug = True

        if self.save_debug_image and self.frame_count % 20 == 1:
            debug_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
            cv2.aruco.drawDetectedMarkers(
                debug_frame,
                [marker_corners_refined.reshape(1, 4, 2)],
                np.array([[self.marker_id_to_track]], dtype=np.int32),
            )
            cv2.drawFrameAxes(debug_frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
            cv2.imwrite("/tmp/aruco_debug.png", debug_frame)

    def is_outlier(self, pose_msg: PoseStamped, last_pose_msg: Optional[PoseStamped]) -> bool:
        if last_pose_msg is None:
            return False
        dx = pose_msg.pose.position.x - last_pose_msg.pose.position.x
        dy = pose_msg.pose.position.y - last_pose_msg.pose.position.y
        dist = math.sqrt(dx * dx + dy * dy)
        dyaw = abs(norm_angle(yaw_from_pose(pose_msg) - yaw_from_pose(last_pose_msg)))
        return dist > self.max_position_jump or dyaw > self.max_yaw_jump


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
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
