cmake_minimum_required(VERSION 2.8.3)
project(ball_detection)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  pcl_ros
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)
find_package(PCL)
find_package(Boost REQUIRED system)
find_package(OpenCV 2 REQUIRED)

catkin_package(
  INCLUDE_DIRS
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS pcl_ros roscpp 
  DEPENDS Boost PCL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a cpp library
add_executable(${PROJECT_NAME} src/ball_detection.cpp)
#add_library(${PROJECT_NAME} ${${PROJECT_NAME}_SRCS})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

