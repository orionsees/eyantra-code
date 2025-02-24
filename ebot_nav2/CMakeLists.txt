cmake_minimum_required(VERSION 3.8)
project(ebot_nav2)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY
  launch
  models
  meshes
  config
  plugins
  maps
  params
  rviz
  DESTINATION
    share/${PROJECT_NAME}/
)


install(DIRECTORY
  scripts/pyarmor_runtime_000000
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
  scripts/pose_detector.py
  scripts/pf_pose_saver.py
  scripts/duplicate_imu.py
  scripts/vroom.py
  scripts/calldock.py
  scripts/checkimu.py
  scripts/reset_sensors.py
  scripts/vroom_without_arm.py
  scripts/callpayload.py
  scripts/record_ebot.py
  scripts/ComplementaryFilter.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
