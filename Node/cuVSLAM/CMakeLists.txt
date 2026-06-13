cmake_minimum_required(VERSION 3.18)
project(cuvslam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-w)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(CUDAToolkit REQUIRED)

set(CUVSLAM_ROOT "/home/jetson/Workspace/FPV/Thirdparty/cuvslam")
set(CUVSLAM_INCLUDE_DIR "${CUVSLAM_ROOT}/include")
set(CUVSLAM_LIB_DIR "${CUVSLAM_ROOT}/cu12/ubuntu22/aarch64")
set(CUVSLAM_LIB "${CUVSLAM_LIB_DIR}/libcuvslam.so")

if(NOT EXISTS "${CUVSLAM_LIB}")
    message(FATAL_ERROR "cuVSLAM library not found: ${CUVSLAM_LIB}")
endif()
if(NOT EXISTS "${CUVSLAM_INCLUDE_DIR}/cuvslam/cuvslam2.h")
    message(FATAL_ERROR "cuVSLAM headers not found: ${CUVSLAM_INCLUDE_DIR}")
endif()

message(STATUS "cuVSLAM: ${CUVSLAM_LIB}")
message(STATUS "CUDA: ${CUDAToolkit_ROOT}")

add_library(cuvslam_lib SHARED IMPORTED)
set_target_properties(cuvslam_lib PROPERTIES
    IMPORTED_LOCATION "${CUVSLAM_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUVSLAM_INCLUDE_DIR}"
)

set(dependencies
    rclcpp
    sensor_msgs
    std_msgs
    geometry_msgs
    message_filters
    tf2_ros
    tf2
    tf2_geometry_msgs
    nav_msgs
    visualization_msgs
)

add_executable(cuvslam src/cuvslam.cpp)

target_include_directories(cuvslam PRIVATE
    ${CUVSLAM_INCLUDE_DIR}
)

target_link_directories(cuvslam PRIVATE
    ${CUVSLAM_LIB_DIR}
    ${CUDAToolkit_LIBRARY_DIR}
)

ament_target_dependencies(cuvslam ${dependencies})

target_link_libraries(cuvslam
    cuvslam_lib
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    pthread
    dl
)

target_compile_options(cuvslam PRIVATE -O3)

set_target_properties(cuvslam PROPERTIES
    BUILD_WITH_INSTALL_RPATH FALSE
    BUILD_RPATH "${CUVSLAM_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN:${CUVSLAM_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(TARGETS cuvslam
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
