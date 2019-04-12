cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${ROS_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)


#set (OpenCV_DIR "/usr/locale/opencv/share/OpenCV")  
find_package(OpenCV 3.4.0 REQUIRED)

message(STATUS "OpenCV library status:")  
message(STATUS "    version: ${OpenCV_VERSION}")  
message(STATUS "opencv lib:${OpenCV_LIBS}")
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Boost COMPONENTS system)
find_package( PCL 1.8 REQUIRED )

message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../../
${PROJECT_SOURCE_DIR}/../../../include
${EIGEN3_INCLUDE_DIR}
${PCL_INCLUDE_DIRS}
${Pangolin_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/../../../dynamic/dynamic
${PROJECT_SOURCE_DIR}/../../../dynamic/APF
${PROJECT_SOURCE_DIR}/../../../dynamic/detect/src
${PROJECT_SOURCE_DIR}/../../../Thirdparty/ncnn/include 
${Boost_INCLUDE_DIRS}

)

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PCL_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../../Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../../Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../../lib/libORB_SLAM2.so
${PROJECT_SOURCE_DIR}/../../../lib/libdynamicDetector.so
${PROJECT_SOURCE_DIR}/../../../dynamic/detect/lib/libPersonDetector.so
${NCNN_TARGET_LIB}
${Boost_INCLUDE_DIRS}
#/opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_core3.so.3.3.1
-lboost_system
)

# Node for monocular camera




# Node for RGB-D camera
#rosbuild_add_executable(RGBD_video
#src/ros_rgbd_video.cc
#)
#target_link_libraries(RGBD_video
#${LIBS}
#)

rosbuild_add_executable(RGBD_camera
src/ros_rgbd_camera.cc
)
target_link_libraries(RGBD_camera
${LIBS}
)

# Node for stereo camera  off-line
#rosbuild_add_executable(Stereo_camera
#src/ros_stereo_camera.cc
#)
#target_link_libraries(Stereo_camera
#${LIBS}
#)

# Node for stereo camera on-line
#rosbuild_add_executable(Stereo_video
#src/ros_stereo_video.cc
#)
#target_link_libraries(Stereo_video
#${LIBS}
#)##


#rosbuild_add_executable(ros_detect_pedestrain 
#src/ros_detect_pedestrain.cc)
#target_link_libraries(ros_detect_pedestrain
#${LIBS}
#)

#rosbuild_add_executable(pub_stereo_from_video 
#src/pub_stereo_from_video.cpp)
#target_link_libraries(ros_detect_pedestrain
#${LIBS}
#)


