cmake_minimum_required(VERSION 3.8)
project(orbslam3)

# 添加CPU架构控制选项
# option(ENABLE_NATIVE_OPTIMIZATION "Enable native CPU optimization (may cause Illegal Instruction)" OFF)
# option(ENABLE_AVX "Enable AVX instructions" OFF)
# option(ENABLE_AVX2 "Enable AVX2 instructions" OFF)

set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

set(CMAKE_SKIP_BUILD_RPATH FALSE)  # 启用构建时RPATH
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)  # 不使用安装时的RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # add_compile_options(-Wall -Wextra -Wpedantic)
  add_compile_options(-w)
endif()

# find dependencies
find_package(OpenCV 4.5 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(SDK_THIRDPARTY_ROOT_DIR ${CMAKE_SOURCE_DIR}/../../Thirdparty)

# ORBBEC-BRIDGE
set(ORBBEC_BRIDGE_ROOT_DIR ${SDK_THIRDPARTY_ROOT_DIR}/OrbbecBridge)
set(ORBBEC_BRIDGE_LIBRARY_DIRS ${ORBBEC_BRIDGE_ROOT_DIR}/lib)
include_directories(${ORBBEC_BRIDGE_ROOT_DIR})
link_directories(${ORBBEC_BRIDGE_LIBRARY_DIRS})

# OrbbecSDK
set(ORBBEC_SDK_ROOT_DIR ${SDK_THIRDPARTY_ROOT_DIR}/OrbbecSDK)
set(ORBBEC_SDK_LIBRARY_DIRS ${ORBBEC_SDK_ROOT_DIR}/lib)
set(ORBBEC_SDK_INCLUDE_DIR ${ORBBEC_SDK_ROOT_DIR}/include)
include_directories(${ORBBEC_SDK_INCLUDE_DIR})
link_directories(${ORBBEC_SDK_LIBRARY_DIRS})

# ORB-SLAM3
set(ORB_SLAM3_ROOT_DIR ${SDK_THIRDPARTY_ROOT_DIR}/ORB_SLAM3)
set(ORB_SLAM3_LIBRARY_DIRS ${ORB_SLAM3_ROOT_DIR}/lib)
include_directories(
  ${ORB_SLAM3_ROOT_DIR}
  ${ORB_SLAM3_ROOT_DIR}/include
  ${ORB_SLAM3_ROOT_DIR}/include/CameraModels
  ${ORB_SLAM3_ROOT_DIR}/Thirdparty
  ${ORB_SLAM3_ROOT_DIR}/Thirdparty/Sophus
  ${ORB_SLAM3_ROOT_DIR}/Thirdparty/DBOW2
  ${ORB_SLAM3_ROOT_DIR}/Thirdparty/g2o
)
link_directories(${ORB_SLAM3_LIBRARY_DIRS})
link_directories(${ORB_SLAM3_ROOT_DIR}/Thirdparty/DBoW2/lib)
link_directories(${ORB_SLAM3_ROOT_DIR}/Thirdparty/g2o/lib)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(orbslam3 src/test.cpp)
target_link_libraries(orbslam3 ${EIGEN3_LIBS} ${Pangolin_LIBRARIES} ${ORB_SLAM3_ROOT_DIR}/lib/libORB_SLAM3.so ${ORB_SLAM3_ROOT_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so ${ORB_SLAM3_ROOT_DIR}/Thirdparty/g2o/lib/libg2o.so ${OpenCV_LIBRARIES} ${ORBBEC_BRIDGE_LIBRARY_DIRS}/liborbbec_bridge.so ${ORBBEC_SDK_LIBRARY_DIRS}/libOrbbecSDK.so)

# 设置RPATH（构建时和安装时）
set_target_properties(orbslam3 PROPERTIES
    BUILD_RPATH "${ORBBEC_SDK_LIBRARY_DIRS}"
    INSTALL_RPATH "$ORIGIN/../lib"
)