cmake_minimum_required(VERSION 3.12.1)
project(orbbec_bridge
    VERSION 1.0.0
    DESCRIPTION "Bridge library for Orbbec cameras with OpenCV integration"
    LANGUAGES CXX C
)

# 添加CPU架构控制选项
option(ENABLE_NATIVE_OPTIMIZATION "Enable native CPU optimization (may cause Illegal Instruction)" OFF)
option(ENABLE_AVX "Enable AVX instructions" OFF)
option(ENABLE_AVX2 "Enable AVX2 instructions" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)

# 编译器设在
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(UNIX)
    find_package(Threads REQUIRED)
endif()

# OrbbecSDK
set(ORBBEC_SDK_ROOT_DIR ${CMAKE_SOURCE_DIR}/../OrbbecSDK)
set(ORBBEC_SDK_LIBRARY_DIRS ${ORBBEC_SDK_ROOT_DIR}/lib)
set(ORBBEC_SDK_INCLUDE_DIR ${ORBBEC_SDK_ROOT_DIR}/include)
include_directories(${ORBBEC_SDK_INCLUDE_DIR})
link_directories(${ORBBEC_SDK_LIBRARY_DIRS})

# OpenCV
find_package(OpenCV 4.0 REQUIRED)

# 创建库
add_library(orbbec_bridge SHARED orbbec_bridge.cpp)

# 设置属性
set_target_properties(orbbec_bridge PROPERTIES
    SOVERSION 1
    OUTPUT_NAME "orbbec_bridge"
)

# 包含目录
target_include_directories(orbbec_bridge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${ORBBEC_SDK_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 链接库
target_link_libraries(orbbec_bridge
    PUBLIC
        ${OpenCV_LIBS}
    PRIVATE
        Threads::Threads
)

# Orbbec SDK链接
find_library(ORBBEC_SDK_LIB 
    NAMES OrbbecSDK
    PATHS ${ORBBEC_SDK_LIBRARY_DIRS}
    REQUIRED
)
target_link_libraries(orbbec_bridge PRIVATE ${ORBBEC_SDK_LIB})

# RPATH设置
if(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib")
    file(RELATIVE_PATH relLibDir ${CMAKE_INSTALL_PREFIX}/lib ${ORBBEC_SDK_LIBRARY_DIRS})
    if(NOT relLibDir MATCHES "^\\.\\.")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${ORBBEC_SDK_LIBRARY_DIR}")
    endif()
endif()

# 设置属性
set_target_properties(orbbec_bridge PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "orbbec_bridge"
    # 确保每个配置都输出到正确位置
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)
