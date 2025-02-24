cmake_minimum_required(VERSION 3.5)
project(pymoveit2)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(pymoveit2)

# Install examples
set(EXAMPLES_DIR examples)
install(PROGRAMS
    ${EXAMPLES_DIR}/detect.py
    ${EXAMPLES_DIR}/detect_csv.py
    ${EXAMPLES_DIR}/detect_filter1.py
    ${EXAMPLES_DIR}/detect_filter2.py
    ${EXAMPLES_DIR}/servo.py
    ${EXAMPLES_DIR}/servo_old.py
    DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY examples/assets
  DESTINATION lib/${PROJECT_NAME}
)

# Setup the project
ament_package()
