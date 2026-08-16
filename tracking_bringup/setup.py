from setuptools import find_packages, setup

package_name = 'tracking_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='Tracking bringup package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
    'ground_truth_from_tf = tracking_bringup.ground_truth_from_tf:main',
    'aruco_detector = tracking_bringup.aruco_detector:main',
    'square_pose_pid_controller = tracking_bringup.square_pose_pid_controller:main',
    'square_trajectory_logger = tracking_bringup.square_trajectory_logger:main',
    'aruco_odom_ground_truth_logger = tracking_bringup.aruco_odom_ground_truth_logger:main',
    'apriltag_detector = tracking_bringup.apriltag_detector:main',
    'multi_apriltag_detector = tracking_bringup.multi_apriltag_detector:main',
    'lawn_mower_controller = tracking_bringup.lawn_mower_controller:main',
    'waypoint_controller = tracking_bringup.waypoint_controller:main',
    'multi_trajectory_logger = tracking_bringup.multi_trajectory_logger:main',
    'multi_ground_truth_from_tf = tracking_bringup.multi_ground_truth_from_tf:main',
    'stag_detector = tracking_bringup.stag_detector:main',
    'circle_controller = tracking_bringup.circle_controller:main', 
    'multi_stag_detector = tracking_bringup.multi_stag_detector:main',
    'multiboard_detector = tracking_bringup.multiboard_detector:main',
    'multi_aruco_detector = tracking_bringup.multi_aruco_detector:main',
    'odom_to_pose = tracking_bringup.odom_to_pose:main',
    'multi_odom_to_pose = tracking_bringup.multi_odom_to_pose:main',
    'recovery_waypoint_controller = tracking_bringup.recovery_waypoint_controller:main',
    'yolo_dataset_recorder = tracking_bringup.yolo_dataset_recorder:main',
    'yolo_detector = tracking_bringup.yolo_detector:main',
    'tracking_metrics = tracking_bringup.tracking_metrics:main',
],
    },
)