find_package(GSL REQUIRED)
include_directories(
${GSL_INCLUDE_DIRS}
)

add_library(
dynamicDetector SHARED
dynamic/dynamic.cpp
dynamic/fast.cpp
APF/APF.cpp
APF/kd_tree.cc
)
target_link_libraries(
dynamicDetector
${THIRD_PARTY_LIBS}
${GSL_LIBRARIES}
)

