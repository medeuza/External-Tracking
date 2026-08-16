from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf_transformations import quaternion_from_euler

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


DEFAULT_INITIAL_POSITIONS = [
    4.0, 9.0,
    1.0, 9.0,
    -4.25, 9.0,
    -4.25, -9.0,
    1.0, -9.0,
    4.0, -9.0,
]


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Detection:
    x: float
    y: float
    conf: float
    cls: int
    u: float
    v: float
    width_px: float
    height_px: float


class TrackState:
    def __init__(self, node: Node, robot_id: int, topic: str, initial_xy: Optional[Tuple[float, float]]):
        self.robot_id = robot_id
        self.publisher = node.create_publisher(PoseStamped, topic, 10)
        self.initial_xy = initial_xy

        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None
        self.last_yaw: Optional[float] = None
        self.last_stamp_sec: Optional[float] = None

        self.published_count = 0
        self.missed_count = 0

    def reference_xy(self) -> Optional[Tuple[float, float]]:
        if self.last_x is not None and self.last_y is not None:
            return self.last_x, self.last_y
        return self.initial_xy


class MultiYoloDetector(Node):
    def __init__(self):
        super().__init__("multi_yolo_detector")

        self.declare_parameter(
            "image_topic",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/image",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/camera_info",
        )
        self.declare_parameter("model_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("imgsz", 960)
        self.declare_parameter("conf_thres", 0.35)
        self.declare_parameter("iou_thres", 0.50)
        self.declare_parameter("class_ids", [])
        self.declare_parameter("class_names", ["robot"])

        self.declare_parameter("robot_ids", [0, 1, 2, 3, 4, 5])
        self.declare_parameter("pose_topic_prefix", "/yolo_pose_")
        self.declare_parameter("initial_positions", DEFAULT_INITIAL_POSITIONS)

        self.declare_parameter("camera_height_m", 10.0)
        self.declare_parameter("ground_z_m", 0.0)
        self.declare_parameter("x_offset", 0.0)
        self.declare_parameter("y_offset", 0.0)

        self.declare_parameter("max_association_distance", 1.25)
        self.declare_parameter("filter_alpha_pos", 0.70)
        self.declare_parameter("filter_alpha_yaw", 0.45)
        self.declare_parameter("min_yaw_motion_m", 0.04)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_image_topic", "/yolo/debug_image")
        self.declare_parameter("log_every_n", 60)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.model_path = str(self.get_parameter("model_path").value).strip()
        self.device = str(self.get_parameter("device").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.iou_thres = float(self.get_parameter("iou_thres").value)
        self.class_ids = [int(v) for v in self.get_parameter("class_ids").value]
        self.class_names = [str(v) for v in self.get_parameter("class_names").value]

        self.robot_ids = [int(v) for v in self.get_parameter("robot_ids").value]
        self.pose_topic_prefix = str(self.get_parameter("pose_topic_prefix").value)
        self.initial_positions = [float(v) for v in self.get_parameter("initial_positions").value]

        self.camera_height_m = float(self.get_parameter("camera_height_m").value)
        self.ground_z_m = float(self.get_parameter("ground_z_m").value)
        self.x_offset = float(self.get_parameter("x_offset").value)
        self.y_offset = float(self.get_parameter("y_offset").value)

        self.max_association_distance = float(self.get_parameter("max_association_distance").value)
        self.filter_alpha_pos = float(self.get_parameter("filter_alpha_pos").value)
        self.filter_alpha_yaw = float(self.get_parameter("filter_alpha_yaw").value)
        self.min_yaw_motion_m = float(self.get_parameter("min_yaw_motion_m").value)
        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.log_every_n = int(self.get_parameter("log_every_n").value)

        self.bridge = CvBridge()
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.frame_count = 0

        if YOLO is None:
            raise RuntimeError(
                "Python package 'ultralytics' is not installed. Install it with: pip install ultralytics"
            )
        if not self.model_path:
            raise RuntimeError(
                "YOLO model_path is empty. Train/export weights first and pass "
                "--ros-args -p model_path:=/path/to/best.pt"
            )

        self.model = YOLO(self.model_path)
        self.allowed_class_ids = self._resolve_allowed_classes()

        self.tracks: Dict[int, TrackState] = {}
        for idx, rid in enumerate(self.robot_ids):
            initial_xy = self._initial_xy_for_index(idx)
            topic = f"{self.pose_topic_prefix}{rid}"
            self.tracks[rid] = TrackState(self, rid, topic, initial_xy)
            self.get_logger().info(f"YOLO track robot_id={rid}, initial={initial_xy}, topic={topic}")

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10) if self.publish_debug_image else None

        self.get_logger().info(
            f"YOLO detector started. model={self.model_path}, device={self.device}, "
            f"classes={self.allowed_class_ids if self.allowed_class_ids else 'ALL'}"
        )

    def _initial_xy_for_index(self, idx: int) -> Optional[Tuple[float, float]]:
        pos_idx = 2 * idx
        if pos_idx + 1 >= len(self.initial_positions):
            return None
        return self.initial_positions[pos_idx], self.initial_positions[pos_idx + 1]

    def _resolve_allowed_classes(self) -> List[int]:
        allowed = set(self.class_ids)
        names = getattr(self.model, "names", {}) or {}
        if isinstance(names, dict):
            for class_id, name in names.items():
                if str(name) in self.class_names:
                    allowed.add(int(class_id))
        elif isinstance(names, list):
            for class_id, name in enumerate(names):
                if str(name) in self.class_names:
                    allowed.add(int(class_id))
        return sorted(allowed)

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        if self.frame_count == 0:
            self.get_logger().info(
                f"CameraInfo loaded: fx={self.camera_matrix[0,0]:.2f}, "
                f"fy={self.camera_matrix[1,1]:.2f}, cx={self.camera_matrix[0,2]:.2f}, "
                f"cy={self.camera_matrix[1,2]:.2f}"
            )

    def pixel_to_world(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        if self.camera_matrix is None:
            return None
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])
        if fx == 0.0 or fy == 0.0:
            return None

        depth = self.camera_height_m - self.ground_z_m
        cam_x = (u - cx) * depth / fx
        cam_y = (v - cy) * depth / fy

        world_x = self.x_offset - cam_y
        world_y = self.y_offset - cam_x
        return world_x, world_y

    def image_callback(self, msg: Image):
        self.frame_count += 1
        if self.camera_matrix is None:
            if self.frame_count % max(1, self.log_every_n) == 1:
                self.get_logger().warn("Waiting for camera_info...")
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detections = self.detect(frame)
        assignments = self.associate(detections)

        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        for robot_id, det in assignments.items():
            self.publish_track(self.tracks[robot_id], det, msg.header.stamp, stamp_sec)

        for rid, st in self.tracks.items():
            if rid not in assignments:
                st.missed_count += 1

        if self.debug_pub is not None and self.frame_count % 2 == 0:
            debug = self.draw_debug(frame, detections, assignments)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))

        if self.frame_count % max(1, self.log_every_n) == 1:
            counts = ", ".join(
                f"R{rid}:pub={st.published_count},miss={st.missed_count}"
                for rid, st in self.tracks.items()
            )
            self.get_logger().info(
                f"frame={self.frame_count}, detections={len(detections)}, assigned={len(assignments)} | {counts}"
            )

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        detections: List[Detection] = []
        for box, conf, cls_id in zip(xyxy, confs, clss):
            cls_id = int(cls_id)
            if self.allowed_class_ids and cls_id not in self.allowed_class_ids:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            u = 0.5 * (x1 + x2)
            v = 0.5 * (y1 + y2)
            xy = self.pixel_to_world(u, v)
            if xy is None:
                continue
            detections.append(
                Detection(
                    x=xy[0],
                    y=xy[1],
                    conf=float(conf),
                    cls=cls_id,
                    u=u,
                    v=v,
                    width_px=max(0.0, x2 - x1),
                    height_px=max(0.0, y2 - y1),
                )
            )
        return detections

    def associate(self, detections: Sequence[Detection]) -> Dict[int, Detection]:
        """Greedy nearest-neighbour association against last or initial track positions."""
        candidate_pairs = []
        for robot_id, st in self.tracks.items():
            ref = st.reference_xy()
            if ref is None:
                continue
            rx, ry = ref
            for det_idx, det in enumerate(detections):
                dist = math.hypot(det.x - rx, det.y - ry)
                if dist <= self.max_association_distance:
                    candidate_pairs.append((dist, robot_id, det_idx))

        candidate_pairs.sort(key=lambda row: row[0])
        used_tracks = set()
        used_dets = set()
        assignments: Dict[int, Detection] = {}
        for _, robot_id, det_idx in candidate_pairs:
            if robot_id in used_tracks or det_idx in used_dets:
                continue
            assignments[robot_id] = detections[det_idx]
            used_tracks.add(robot_id)
            used_dets.add(det_idx)
        return assignments

    def publish_track(self, st: TrackState, det: Detection, stamp, stamp_sec: float):
        raw_x, raw_y = det.x, det.y

        if st.last_x is None or st.last_y is None:
            x_f, y_f = raw_x, raw_y
            yaw = st.last_yaw if st.last_yaw is not None else 0.0
        else:
            ax = self.filter_alpha_pos
            x_f = ax * raw_x + (1.0 - ax) * st.last_x
            y_f = ax * raw_y + (1.0 - ax) * st.last_y
            dx = x_f - st.last_x
            dy = y_f - st.last_y
            if math.hypot(dx, dy) >= self.min_yaw_motion_m:
                measured_yaw = math.atan2(dy, dx)
                if st.last_yaw is None:
                    yaw = measured_yaw
                else:
                    yaw = wrap_to_pi(st.last_yaw + self.filter_alpha_yaw * wrap_to_pi(measured_yaw - st.last_yaw))
            else:
                yaw = st.last_yaw if st.last_yaw is not None else 0.0

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "world"
        pose.pose.position.x = float(x_f)
        pose.pose.position.y = float(y_f)
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)

        st.publisher.publish(pose)
        st.last_x, st.last_y, st.last_yaw = x_f, y_f, yaw
        st.last_stamp_sec = stamp_sec
        st.published_count += 1

    def draw_debug(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        assignments: Dict[int, Detection],
    ) -> np.ndarray:
        debug = frame.copy()
        assigned_det_ids = {id(det): rid for rid, det in assignments.items()}
        for det in detections:
            w = det.width_px
            h = det.height_px
            x1 = int(det.u - w / 2.0)
            y1 = int(det.v - h / 2.0)
            x2 = int(det.u + w / 2.0)
            y2 = int(det.v + h / 2.0)
            rid = assigned_det_ids.get(id(det))
            label = f"R{rid}" if rid is not None else "unassigned"
            label += f" {det.conf:.2f} ({det.x:.2f},{det.y:.2f})"
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return debug


def main(args=None):
    rclpy.init(args=args)
    node = MultiYoloDetector()
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
