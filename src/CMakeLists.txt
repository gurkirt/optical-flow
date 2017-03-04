cmake_minimum_required(VERSION 2.8)
project( optical-flow )
SET(OpenCV_DIR "/mnt/jupiter-alpha/softwares/opencv/build-sun")
find_package( OpenCV REQUIRED )
#echo $OpenCV_INCLUDE_DIRS
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(OF  main.cpp utils.cpp)
target_link_libraries( OF ${OpenCV_LIBS} )
