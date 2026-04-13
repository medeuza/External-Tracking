from setuptools import find_packages, setup

package_name = 'tracking_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
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
        ],
},
)