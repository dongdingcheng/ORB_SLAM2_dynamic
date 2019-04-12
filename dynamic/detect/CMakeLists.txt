cmake_minimum_required(VERSION 3.5)
project(detect)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.4 REQUIRED)
include_directories(
${OpenCV_INCLUDE_DIR}
${PROJECT_SOURCE_DIR}/src
)

message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/)


aux_source_directory(${PROJECT_SOURCE_DIR}/src/ sourcefile)

#set(LIBS 
#${OpenCV_LIBS} 
#${PROJECT_SOURCE_DIR}/lib/libPersonDetector.so
#)

add_library(PersonDetector SHARED
${sourcefile}
)

target_link_libraries(PersonDetector
${OpenCV_LIBS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(ACFTest ACFTest.cpp)
target_link_libraries(ACFTest 
PersonDetector
)

