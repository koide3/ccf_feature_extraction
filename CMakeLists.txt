cmake_minimum_required(VERSION 2.8.3)
project(ccf_feature_extraction)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

include($ENV{DLIB_ROOT}/dlib/cmake)

###################################
## catkin specific configuration ##
###################################
catkin_package(
   INCLUDE_DIRS include
   LIBRARIES ccf_feature_extraction
   CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_test
  src/ccf_feature_extraction_test.cpp
)
target_link_libraries(${PROJECT_NAME}_test
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  dlib::dlib
)
