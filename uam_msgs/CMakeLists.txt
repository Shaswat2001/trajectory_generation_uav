cmake_minimum_required(VERSION 3.5)
project(uam_msgs)

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
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

set(msg_files
  "msg/States.msg"
  "msg/Dynamics.msg"
)

set(srv_files
  "srv/Uam.srv"
  "srv/RequestUavPose.srv"
  "srv/RequestUavVel.srv"
  "srv/ResponseUavPose.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/States.msg"
  "msg/Dynamics.msg"
  "srv/Uam.srv"
  "srv/RequestUavPose.srv"
  "srv/RequestUavVel.srv"
  "srv/ResponseUavPose.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
