cmake_minimum_required(VERSION 3.0.2)
project(ddpg_control)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ddpg_control
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs nav_msgs tf2_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(ddpg_control
  src/actor_network.cpp
  src/critic_network.cpp
  src/replay_buffer.cpp
  src/ddpg_agent.cpp
)
target_link_libraries(ddpg_control ${catkin_LIBRARIES})

define_question_package_macro(
  TARGET train_node
)
add_executable(train_node src/train_node.cpp)
target_link_libraries(train_node ddpg_control ${catkin_LIBRARIES})