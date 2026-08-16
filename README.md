# External Visual Tracking for Multi-Robot Systems

ROS 2 + Gazebo framework for experimental evaluation of **external visual localization methods in multi-robot warehouse environments**.

The project compares:

* ArUco
* AprilTag
* STag
* MarkerBoard
* YOLO-based markerless tracking
* wheel odometry as a baseline

The experiments are conducted in a simulated warehouse with an overhead RGB camera and up to six TurtleBot3 robots. The main objective is to compare localization accuracy, stability, robustness, and applicability of different external tracking approaches.

## System Architecture

The system uses a centralized external tracking architecture.

```text
Overhead RGB Camera
        │
        ▼
ROS–Gazebo Bridge
        │
        ▼
Tracking Node
ArUco / AprilTag / STag / MarkerBoard / YOLO
        │
        ▼
Estimated Robot Pose
        │
        ├──────────────► Trajectory Logger
        │                     │
        │                     ▼
        │                 CSV Results
        │
        ▼
PID Controller
        │
        ▼
     /cmd_vel
        │
        ▼
   TurtleBot3
```

Gazebo provides the Ground Truth robot pose independently from the tracking pipeline. Estimated positions are compared against Ground Truth and logged to CSV for quantitative evaluation.

The experiments use both simple circle and square trajectories for initial validation and six independent warehouse trajectories for multi-robot experiments.

## Installation

The project was developed using:

```text
ROS 2 Jazzy
Gazebo Sim Harmonic
Python 3
OpenCV
TurtleBot3
ros_gz_bridge
```

For YOLO:

```bash
pip install ultralytics
```

Build the ROS 2 workspace:

```bash
cd ~/wspace

source /opt/ros/jazzy/setup.bash

colcon build --packages-select tracking_bringup

source install/setup.bash
```

## Running the Simulation

### 1. Start Gazebo

```bash
cd ~/wspace/src/tracking_assets

source /opt/ros/jazzy/setup.bash

export GZ_SIM_RESOURCE_PATH="$PWD/models:$PWD/models/generated:$PWD/models/templates:/opt/ros/jazzy/share/turtlebot3_gazebo/models"

gz sim -r worlds/turtlebot3_world.world
```

### 2. Start the ROS–Gazebo Bridge

Open another terminal:

```bash
source /opt/ros/jazzy/setup.bash

ros2 run ros_gz_bridge parameter_bridge \
/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock \
/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry \
/world/default/dynamic_pose/info@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V \
/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image \
/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera_sensor/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo
```

### 3. Start Ground Truth

```bash
cd ~/wspace

source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 run tracking_bringup ground_truth_from_tf --ros-args \
  -p use_sim_time:=true \
  -p transform_index:=0
```

### 4. Example: AprilTag Detector

```bash
ros2 run tracking_bringup apriltag_detector --ros-args \
-p use_sim_time:=true \
-p marker_size_m:=0.46 \
-p yaw_offset:=1.5708 \
-p filter_alpha_pos:=0.35 \
-p filter_alpha_yaw:=0.20
```

### 5. Start the Logger

```bash
rm -f ~/wspace/logs/*.csv

ros2 run tracking_bringup square_trajectory_logger --ros-args \
  -p use_sim_time:=true \
  -p mode:=apriltag_square_pid_3loops \
  -p odom_topic:=/odom \
  -p visual_topic:=/apriltag_pose \
  -p ground_truth_topic:=/ground_truth_pose
```

### 6. Start the PID Controller

```bash
ros2 run tracking_bringup square_pose_pid_controller --ros-args \
-p use_sim_time:=true \
-p pose_source:=apriltag \
-p visual_topic:=/apriltag_pose \
-p cmd_vel_topic:=/cmd_vel \
-p side_length:=0.50 \
-p num_sides:=4 \
-p turn_angle_deg:=90.0 \
-p linear_speed:=0.045 \
-p max_linear_speed:=0.065 \
-p max_angular_speed:=0.25 \
-p min_turn_speed:=0.12 \
-p distance_tolerance:=0.003 \
-p drive_settle_cycles:=6 \
-p turn_tolerance_deg:=3.0 \
-p turn_settle_cycles:=3 \
-p tag_offset_x:=-0.032 \
-p tag_offset_y:=0.02 \
-p pose_timeout:=1.5
```

## Results

The final comparison was performed in the multi-robot warehouse scenario under the same experimental conditions.

| Method              |  MAE, cm | RMSE, cm |      FPS | Calibration | Multi-robot support | Collisions |
| ------------------- | -------: | -------: | -------: | ----------- | ------------------- | ---------: |
| Odometry (baseline) |    26.12 |    30.61 |     50.0 | —           | —                   |         71 |
| ArUco               |     1.65 |     2.18 |     30.0 | Polynomial  | Yes                 |          0 |
| AprilTag            |     6.96 |     7.46 |     0.9* | Polynomial  | Limited             |          0 |
| STag                |     2.18 |     3.00 |     30.3 | k-NN        | Yes                 |          0 |
| **MarkerBoard**     | **0.52** | **1.05** | **30.3** | **k-NN**    | **Yes**             |      **0** |
| YOLO (single robot) |     3.23 |     3.57 |     19.0 | —           | No                  |          — |

* AprilTag update frequency is reported per robot in the six-robot experiment. In the single-robot configuration, the detector reaches approximately 30 Hz.

The results show that **MarkerBoard with k-NN calibration is the best-performing localization method overall**, achieving the lowest localization error while maintaining real-time processing at approximately 30 Hz and stable operation with multiple robots.

ArUco provides the best trade-off between implementation simplicity, accuracy, computational performance, and multi-robot stability. STag also provides reliable real-time localization, although it requires a more complex calibration procedure.

AprilTag showed substantially reduced processing performance in the six-robot scenario, while YOLO achieved good localization accuracy for a single robot but could not provide reliable multi-robot tracking because standard object detection does not maintain persistent identities for visually identical robots.

Therefore, for the current experimental configuration:

**Best localization accuracy:** MarkerBoard + k-NN
**Best practical accuracy/complexity trade-off:** ArUco
**Best markerless result:** YOLO, currently limited to reliable single-robot tracking

