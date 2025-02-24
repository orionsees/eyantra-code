# Hardware: Logistic coBot (LB) for eYRC 2024-25

rm -rf build install log eYantra24-25

ros2 service call /dock_control ebot_docking/srv/DockSw "{linear_dock: False, orientation_dock: True, orientation: 0.0}"

ros2 service call /pass_control ebot_docking/srv/DockSw "{startcmd: False, undocking: False}"

ros2 run tf2_ros static_transform_publisher 3.2 1.9 0.0 0.0 0.0 3.14 odom conv1

ros2 run tf2_ros static_transform_publisher 3.1 -1.15 0.0 0.0 0.0 3.14 odom conv2

ros2 run tf2_ros static_transform_publisher 3.0 -2.5 0.0 0.0 0.0 3.14 odom arm

ros2 run ebot_nav2 duplicate_imu.py

ros2 run ebot_nav2 reset_sensors.py

ros2 run ebot_nav2 record_ebot.py

ros2 run pymoveit2 detect.py

ros2 run pymoveit2 detect_filter1.py

ros2 run pymoveit2 detect_filter2.py

ros2 run pymoveit2 servo.py

ros2 run ebot_docking ebot_docking_boilerplate.py

ros2 run ebot_nav2 vroom.py

ros2 run ebot_nav2 checkimu.py

ros2 run ebot_nav2 calldock.py
pull
