cmake_minimum_required(VERSION 3.0.2)
project(heart_pkg)

add_compile_options(-std=c++17)

# Trova i pacchetti ROS necessari
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_srvs
  geometry_msgs
  std_msgs
  tf
  tf2_ros
  tf2_eigen
  image_transport
  cv_bridge
)

# Trova Eigen3
find_package(Eigen3 REQUIRED)

# Trova OpenCV
find_package(OpenCV REQUIRED)

# Configura il pacchetto catkin
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs tf tf2_ros tf2_eigen image_transport cv_bridge
)

# Includi le directory dei pacchetti richiesti
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include
  /home/corbe/wavelib/header
)

# Percorso alla libreria statica di WaveLib
set(WAVELIB_LIB /home/corbe/wavelib/build/Bin/libwavelib.a)

# Aggiungi l'eseguibile
add_executable(opticalflowpose
  src/main.cpp
  src/OpticalFlowPose.cpp
  src/OpticalFlow.cpp
  src/Visualizer.cpp
  src/Feature_extractor.cpp
  src/data.cpp
)

# Collega le librerie necessarie all'eseguibile
target_link_libraries(opticalflowpose
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
  ${OpenCV_LIBS}
  ${WAVELIB_LIB}
)
