cmake_minimum_required(VERSION 3.10)

# set the project name
project(slam_pt_filter)

# dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  std_msgs
  rosbag
  tf2
)
find_package(Eigen3)

catkin_package(
 CATKIN_DEPENDS nav_msgs roscpp rospy sensor_msgs std_msg tf2 geometry_msgs eigen_conversions
 DEPENDS EIGEN3
)

# include directories
include_directories(
  src
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# add the executable
add_executable(data_visualize_node src/visualize_laserscan.cpp)
target_link_libraries(data_visualize_node ${catkin_LIBRARIES})

add_executable(particle_function_test src/test.cpp)
target_link_libraries(particle_function_test ${catkin_LIBRARIES})