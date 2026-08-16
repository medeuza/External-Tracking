import math
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Twist
from tf_transformations import euler_from_quaternion


def yaw_from_quat(q) -> float:
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw


def norm_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


SHELVES_WORLD = [
    ("shelf1",  2.5,  4.5, 1.0, 7.0),
    ("shelf2", -0.5,  6.5, 1.0, 3.0),
    ("shelf3", -0.5,  2.5, 1.0, 3.0),
    ("shelf4", -3.5,  4.5, 1.0, 7.0),
    ("shelf5",  2.5, -4.5, 1.0, 7.0),
    ("shelf6", -0.5, -6.5, 1.0, 3.0),
    ("shelf7", -0.5, -2.5, 1.0, 3.0),
    ("shelf8", -3.5, -4.5, 1.0, 7.0),
]
WALLS_WORLD = [
    ("wall_east",   5.6,   0.0,  0.4, 20.0),
    ("wall_west",  -5.6,   0.0,  0.4, 20.0),
    ("wall_north",  0.0,  10.1, 11.0,  0.4),
    ("wall_south",  0.0, -10.1, 11.0,  0.4),
]
OBSTACLES_WORLD = SHELVES_WORLD + WALLS_WORLD


def obstacles_to_bounds(margin: float):
    out = []
    for (name, cx, cy, sx, sy) in OBSTACLES_WORLD:
        out.append((name,
                    cx - sx / 2.0 - margin, cx + sx / 2.0 + margin,
                    cy - sy / 2.0 - margin, cy + sy / 2.0 + margin))
    return out


def point_in_any_obstacle(x: float, y: float, boxes) -> Optional[str]:
    for (name, xmin, xmax, ymin, ymax) in boxes:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return name
    return None


def nearest_point_on_polyline(px: float, py: float,
                              waypoints: List[Tuple[float, float]]
                              ) -> Tuple[float, float, int]:
    p = np.array([px, py], dtype=float)
    wps = np.array(waypoints, dtype=float)
    if not np.allclose(wps[0], wps[-1]):
        wps = np.vstack([wps, wps[0]])
    best_d = np.inf
    best_q = wps[0]
    best_end_idx = 0
    for i in range(len(wps) - 1):
        a = wps[i]; b = wps[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            q = a
        else:
            t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
            q = a + t * ab
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best_d = d
            best_q = q
            best_end_idx = (i + 1) % len(waypoints)
    return float(best_q[0]), float(best_q[1]), best_end_idx


def gz_set_pose(world: str, model: str, x: float, y: float,
                yaw: float, z: float = 0.01,
                timeout_ms: int = 2000) -> bool:
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    req = (f'name: "{model}", '
           f'position: {{x: {x}, y: {y}, z: {z}}}, '
           f'orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}')
    cmd = ["gz", "service",
           "-s", f"/world/{world}/set_pose",
           "--reqtype", "gz.msgs.Pose",
           "--reptype", "gz.msgs.Boolean",
           "--timeout", str(timeout_ms),
           "--req", req]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=4.0)
        return r.returncode == 0
    except Exception:
        return False


def gz_pause(world: str, pause: bool, timeout_ms: int = 1500) -> bool:
    req = f'pause: {"true" if pause else "false"}'
    cmd = ["gz", "service",
           "-s", f"/world/{world}/control",
           "--reqtype", "gz.msgs.WorldControl",
           "--reptype", "gz.msgs.Boolean",
           "--timeout", str(timeout_ms),
           "--req", req]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
        return r.returncode == 0
    except Exception:
        return False


STATE_DRIVE = "DRIVE"
STATE_TURN = "TURN"


class RecoveryWaypointController(Node):
    def __init__(self):
        super().__init__("recovery_waypoint_controller")

        self.declare_parameter("robot_id", 0)
        self.declare_parameter("pose_topic", "/odom_pose_0")
        self.declare_parameter("gt_topic", "/ground_truth_pose_0")
        self.declare_parameter("cmd_vel_topic",
                               "/model/turtlebot3_burger_aruco_000/cmd_vel")
        self.declare_parameter("model_name", "turtlebot3_burger_aruco_000")
        self.declare_parameter("world_name", "default")

        self.declare_parameter("other_robot_ids", [1, 2, 3, 4, 5])
        self.declare_parameter("other_gt_topic_prefix", "/ground_truth_pose_")
        self.declare_parameter("robot_collision_dist", 0.30)

        self.declare_parameter("waypoints", [4.0, 9.0, 4.0, 0.5])
        self.declare_parameter("linear_speed", 0.18)
        self.declare_parameter("max_angular_speed", 0.25)
        self.declare_parameter("position_tolerance", 0.10)
        self.declare_parameter("turn_tolerance_rad", 0.10)

        self.declare_parameter("shelf_margin", 0.20)
        self.declare_parameter("post_teleport_pause_s", 0.3)

        self.declare_parameter("log_dir", str(Path.home() / "wspace" / "logs"))

        self.robot_id = int(self.get_parameter("robot_id").value)
        self.model_name = str(self.get_parameter("model_name").value)
        self.world_name = str(self.get_parameter("world_name").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.max_omega = float(self.get_parameter("max_angular_speed").value)
        self.pos_tol = float(self.get_parameter("position_tolerance").value)
        self.yaw_tol = float(self.get_parameter("turn_tolerance_rad").value)
        margin = float(self.get_parameter("shelf_margin").value)
        self.post_teleport_pause = float(
            self.get_parameter("post_teleport_pause_s").value)

        self.other_robot_ids = [int(v) for v
                                in self.get_parameter("other_robot_ids").value
                                if int(v) != self.robot_id]
        self.other_gt_prefix = str(
            self.get_parameter("other_gt_topic_prefix").value)
        self.robot_coll_dist = float(
            self.get_parameter("robot_collision_dist").value)

        wps_flat = list(self.get_parameter("waypoints").value)
        if len(wps_flat) < 4 or len(wps_flat) % 2 != 0:
            raise ValueError("waypoints must be flat list, even length >=4")
        self.waypoints = [(float(wps_flat[i]), float(wps_flat[i + 1]))
                          for i in range(0, len(wps_flat), 2)]
        self.obstacle_boxes = obstacles_to_bounds(margin)

        self.state = STATE_DRIVE
        self.wp_idx = 0
        self.loop_idx = 0
        self.collision_count = 0
        self.was_inside_obstacle = False
        self.was_close_to: Dict[int, bool] = {oid: False
                                              for oid in self.other_robot_ids}

        self.odom_xy: Optional[np.ndarray] = None
        self.odom_yaw: Optional[float] = None
        self.gt_xy: Optional[np.ndarray] = None
        self.gt_yaw: Optional[float] = None

        self.other_gt: Dict[int, Optional[np.ndarray]] = {
            oid: None for oid in self.other_robot_ids
        }

        self.bias_xy = np.zeros(2)
        self.bias_yaw = 0.0

        log_dir = Path(self.get_parameter("log_dir").value)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1e9)
        self.log_path = log_dir / f"collisions_{ts}_robot_{self.robot_id}.csv"
        self._fh = open(self.log_path, "w", buffering=1)
        self._fh.write("t,event,gt_x,gt_y,odom_x,odom_y,obstacle_name,"
                       "target_x,target_y\n")

        cmd_topic = str(self.get_parameter("cmd_vel_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        gt_topic = str(self.get_parameter("gt_topic").value)

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, qos)
        self.create_subscription(PoseStamped, pose_topic, self._odom_cb, qos)
        self.create_subscription(PoseStamped, gt_topic, self._gt_cb, qos)

        for oid in self.other_robot_ids:
            topic = f"{self.other_gt_prefix}{oid}"
            self.create_subscription(PoseStamped, topic,
                                     self._make_other_gt_cb(oid), qos)

        self.timer = self.create_timer(0.05, self._step)

        self.get_logger().info(
            f"R{self.robot_id}: {len(self.waypoints)} wp, INFINITE loops")
        self.get_logger().info(
            f"  model={self.model_name}  world={self.world_name}")
        self.get_logger().info(
            f"  obstacles: {len(SHELVES_WORLD)} shelves + "
            f"{len(WALLS_WORLD)} walls  margin={margin:.2f}m")
        self.get_logger().info(
            f"  other robots: {self.other_robot_ids}  "
            f"collision_dist={self.robot_coll_dist:.2f}m  "
            f"(only smaller-id handles)")
        self.get_logger().info(f"  log: {self.log_path}")

    def _odom_cb(self, msg: PoseStamped):
        x = msg.pose.position.x + self.bias_xy[0]
        y = msg.pose.position.y + self.bias_xy[1]
        yaw = norm_angle(yaw_from_quat(msg.pose.orientation) + self.bias_yaw)
        self.odom_xy = np.array([x, y])
        self.odom_yaw = yaw

    def _gt_cb(self, msg: PoseStamped):
        self.gt_xy = np.array([msg.pose.position.x, msg.pose.position.y])
        self.gt_yaw = yaw_from_quat(msg.pose.orientation)

    def _make_other_gt_cb(self, other_id: int):
        def cb(msg: PoseStamped):
            self.other_gt[other_id] = np.array(
                [msg.pose.position.x, msg.pose.position.y])
        return cb

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _send_cmd(self, v: float, w: float):
        m = Twist(); m.linear.x = float(v); m.angular.z = float(w)
        self.cmd_pub.publish(m)

    def _stop(self):
        self._send_cmd(0.0, 0.0)

    def _log_event(self, event: str, obstacle_name: str = "",
                   target: Optional[Tuple[float, float]] = None):
        gx, gy = (self.gt_xy if self.gt_xy is not None
                  else (math.nan, math.nan))
        ox, oy = (self.odom_xy if self.odom_xy is not None
                  else (math.nan, math.nan))
        tx = ty = math.nan
        if target is not None:
            tx, ty = target
        self._fh.write(
            f"{self._now():.3f},{event},{gx:.4f},{gy:.4f},"
            f"{ox:.4f},{oy:.4f},{obstacle_name},{tx:.4f},{ty:.4f}\n")

    def _do_recovery(self, obstacle_label: str):
        self.collision_count += 1
        tx, ty, new_wp_idx = nearest_point_on_polyline(
            float(self.gt_xy[0]), float(self.gt_xy[1]), self.waypoints)
        next_wp = self.waypoints[new_wp_idx]
        tyaw = math.atan2(next_wp[1] - ty, next_wp[0] - tx)

        self.get_logger().warn(
            f"R{self.robot_id} COLLISION #{self.collision_count} "
            f"with {obstacle_label} at GT=({self.gt_xy[0]:.2f}, "
            f"{self.gt_xy[1]:.2f}).  PAUSE + teleport to ({tx:.2f}, {ty:.2f})")

        self._stop()
        if not gz_pause(self.world_name, True):
            self.get_logger().error(
                f"R{self.robot_id}: gz_pause(True) failed")

        if gz_set_pose(self.world_name, self.model_name, tx, ty, tyaw):
            self.get_logger().info(
                f"R{self.robot_id}: teleported to ({tx:.2f}, {ty:.2f}, "
                f"yaw={math.degrees(tyaw):.0f}°)")
        else:
            self.get_logger().error(f"R{self.robot_id}: gz_set_pose failed")

        # update belief
        if self.odom_xy is not None and self.odom_yaw is not None:
            raw_x = self.odom_xy[0] - self.bias_xy[0]
            raw_y = self.odom_xy[1] - self.bias_xy[1]
            raw_yaw = norm_angle(self.odom_yaw - self.bias_yaw)
            self.bias_xy = np.array([tx - raw_x, ty - raw_y])
            self.bias_yaw = norm_angle(tyaw - raw_yaw)
            self.odom_xy = np.array([tx, ty])
            self.odom_yaw = tyaw

        self.wp_idx = new_wp_idx
        time.sleep(self.post_teleport_pause)

        if not gz_pause(self.world_name, False):
            self.get_logger().error(
                f"R{self.robot_id}: gz_pause(False) failed — sim may stay paused!")

        self._log_event("collision", obstacle_name=obstacle_label,
                        target=(tx, ty))
        self.was_inside_obstacle = False
        self.was_close_to = {oid: False for oid in self.other_robot_ids}

    def _check_robot_robot_collision(self) -> Optional[int]:
        if self.gt_xy is None:
            return None
        for oid in self.other_robot_ids:
            other_xy = self.other_gt[oid]
            if other_xy is None:
                continue
            d = float(np.hypot(self.gt_xy[0] - other_xy[0],
                               self.gt_xy[1] - other_xy[1]))
            if d < self.robot_coll_dist:
                if not self.was_close_to[oid]:
                    self.was_close_to[oid] = True
                    if self.robot_id < oid:
                        return oid
            else:
                self.was_close_to[oid] = False
        return None


    def _step(self):
        if self.odom_xy is None or self.gt_xy is None:
            return

        intruded = point_in_any_obstacle(float(self.gt_xy[0]),
                                         float(self.gt_xy[1]),
                                         self.obstacle_boxes)
        if intruded is not None and not self.was_inside_obstacle:
            self.was_inside_obstacle = True
            self._do_recovery(intruded)
            return
        if intruded is None:
            self.was_inside_obstacle = False

        rr = self._check_robot_robot_collision()
        if rr is not None:
            self._do_recovery(f"robot_R{rr}")
            return
        tgt = np.array(self.waypoints[self.wp_idx])
        dxy = tgt - self.odom_xy
        dist = float(np.linalg.norm(dxy))

        if dist < self.pos_tol:
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
            if self.wp_idx == 0:
                self.loop_idx += 1
                self.get_logger().info(
                    f"R{self.robot_id} loop {self.loop_idx}  "
                    f"(collisions: {self.collision_count})")
            return

        desired_yaw = math.atan2(dxy[1], dxy[0])
        yaw_err = norm_angle(desired_yaw - self.odom_yaw)

        if abs(yaw_err) > self.yaw_tol:
            w = max(-self.max_omega, min(self.max_omega, 1.5 * yaw_err))
            self._send_cmd(0.0, w)
            self.state = STATE_TURN
        else:
            v = self.linear_speed
            w = max(-self.max_omega, min(self.max_omega, 1.0 * yaw_err))
            self._send_cmd(v, w)
            self.state = STATE_DRIVE

    def destroy_node(self):
        try: self._stop()
        except Exception: pass
        try: self._fh.close()
        except Exception: pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecoveryWaypointController()
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