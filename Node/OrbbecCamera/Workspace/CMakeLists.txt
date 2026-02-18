cmake_minimum_required(VERSION 3.8)
project(orbbec_camera)

# 添加CPU架构控制选项
option(ENABLE_NATIVE_OPTIMIZATION "Enable native CPU optimization (may cause Illegal Instruction)" OFF)
option(ENABLE_AVX "Enable AVX instructions" OFF)
option(ENABLE_AVX2 "Enable AVX2 instructions" OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  
endif()

set(SDK_THIRDPARTY_ROOT_DIR ${CMAKE_SOURCE_DIR}/../../../Thirdparty)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED) 
find_package(builtin_interfaces REQUIRED) 
find_package(geometry_msgs REQUIRED) 
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV 4.5 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
# OrbbecSDK
set(ORBBEC_SDK_ROOT_DIR ${SDK_THIRDPARTY_ROOT_DIR}/OrbbecSDK)
set(ORBBEC_SDK_LIBRARY_DIRS ${ORBBEC_SDK_ROOT_DIR}/lib)
set(ORBBEC_SDK_INCLUDE_DIR ${ORBBEC_SDK_ROOT_DIR}/include)
include_directories(${ORBBEC_SDK_INCLUDE_DIR})
link_directories(${ORBBEC_SDK_LIBRARY_DIRS})
# OrbbecBridge
set(ORBBEC_BRIDGE_ROOT_DIR ${SDK_THIRDPARTY_ROOT_DIR}/OrbbecBridge)
set(ORBBEC_BRIDGE_LIBRARY_DIRS ${ORBBEC_BRIDGE_ROOT_DIR}/lib)
set(ORBBEC_BRIDGE_INCLUDE_DIR ${ORBBEC_BRIDGE_ROOT_DIR})
include_directories(${ORBBEC_BRIDGE_INCLUDE_DIR})
link_directories(${ORBBEC_BRIDGE_LIBRARY_DIRS})

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
add_executable(orbbec_camera src/orbbec_camera.cpp)
ament_target_dependencies(orbbec_camera rclcpp sensor_msgs tf2_ros tf2 geometry_msgs std_msgs)
target_link_libraries(orbbec_camera ${OpenCV_LIBS} ${ORBBEC_BRIDGE_LIBRARY_DIRS}/liborbbec_bridge.so ${ORBBEC_SDK_LIBRARY_DIRS}/libOrbbecSDK.so)                                  

install(TARGETS
  orbbec_camera 
  DESTINATION lib/${PROJECT_NAME}
)
