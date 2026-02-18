cmake_minimum_required(VERSION 3.8)
project(orb_slam3)

# 添加CPU架构控制选项
option(ENABLE_NATIVE_OPTIMIZATION "Enable native CPU optimization (may cause Illegal Instruction)" OFF)
option(ENABLE_AVX "Enable AVX instructions" OFF)
option(ENABLE_AVX2 "Enable AVX2 instructions" OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-w)
  
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED) 
find_package(builtin_interfaces REQUIRED) 
find_package(geometry_msgs REQUIRED) 
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED) 
# find dependencies
find_package(OpenCV 4.10.0 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
# ORB-SLAM3
set(ORB_SLAM3_ROOT_DIR ${CMAKE_SOURCE_DIR}/../../../Thirdparty/ORB_SLAM3)
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

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

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

if(UNIX)
#linux system set sample rpath
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN:${CMAKE_INSTALL_RPATH}:${ORBBEC_SDK_LIBRARY_DIRS}:${ORBBEC_BRIDGE_LIBRARY_DIRS}")
    if(NOT APPLE)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
    endif()
endif()

ament_package()
add_executable(orb_slam3 src/orb_slam3.cpp)
ament_target_dependencies(orb_slam3 rclcpp sensor_msgs tf2_ros tf2 geometry_msgs std_msgs message_filters nav_msgs visualization_msgs tf2_geometry_msgs)
target_link_libraries(orb_slam3 
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${Pangolin_LIBRARIES}
  DBoW2
  g2o
  ORB_SLAM3
  -lboost_serialization
  -lcrypto
)

# 更完善的RPATH设置
set_target_properties(orb_slam3 PROPERTIES
    BUILD_WITH_INSTALL_RPATH FALSE
    BUILD_RPATH "${ORB_SLAM3_ROOT_DIR}/lib:${ORB_SLAM3_ROOT_DIR}/Thirdparty/DBoW2/lib:${ORB_SLAM3_ROOT_DIR}/Thirdparty/g2o/lib:${OpenCV_LIBS}"
    INSTALL_RPATH "$ORIGIN:${ORB_SLAM3_ROOT_DIR}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# 安装时也安装依赖库
install(TARGETS orb_slam3
    # ${ORB_SLAM3_ROOT_DIR}/lib/libORB_SLAM3.so
    # ${ORB_SLAM3_ROOT_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    # ${ORB_SLAM3_ROOT_DIR}/Thirdparty/g2o/lib/libg2o.so
  DESTINATION lib/${PROJECT_NAME}
)