# Install script for directory: /home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/Workspace

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/install/orb_slam3")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/orb_slam3")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/orb_slam3")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orb_slam3/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orb_slam3/environment" TYPE FILE FILES "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orb_slam3/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orb_slam3/environment" TYPE FILE FILES "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/ament_cmake_environment_hooks/path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orb_slam3" TYPE FILE FILES "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/ament_cmake_environment_hooks/local_setup.bash")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orb_slam3" TYPE FILE FILES "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/ament_cmake_environment_hooks/local_setup.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orb_slam3" TYPE FILE FILES "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orb_slam3" TYPE FILE FILES "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orb_slam3" TYPE FILE FILES "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/ament_cmake_environment_hooks/package.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/ament_cmake_index/share/ament_index/resource_index/packages/orb_slam3")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orb_slam3/cmake" TYPE FILE FILES
    "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/ament_cmake_core/orb_slam3Config.cmake"
    "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/ament_cmake_core/orb_slam3Config-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orb_slam3" TYPE FILE FILES "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/Workspace/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/orb_slam3/orb_slam3" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/orb_slam3/orb_slam3")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/orb_slam3/orb_slam3"
         RPATH "$ORIGIN:/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/Workspace/../../../Thirdparty/ORB_SLAM3/lib:/opt/ros/humble/lib:/usr/local/opencv-4.10.0/lib:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/orb_slam3" TYPE EXECUTABLE FILES "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/orb_slam3")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/orb_slam3/orb_slam3" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/orb_slam3/orb_slam3")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/orb_slam3/orb_slam3"
         OLD_RPATH "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/Workspace/../../../Thirdparty/ORB_SLAM3/lib:/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/Workspace/../../../Thirdparty/ORB_SLAM3/Thirdparty/DBoW2/lib:/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/Workspace/../../../Thirdparty/ORB_SLAM3/Thirdparty/g2o/lib:opencv_calib3d:opencv_core:opencv_dnn:opencv_features2d:opencv_flann:opencv_gapi:opencv_highgui:opencv_imgcodecs:opencv_imgproc:opencv_ml:opencv_objdetect:opencv_photo:opencv_stitching:opencv_video:opencv_videoio:opencv_alphamat:opencv_aruco:opencv_bgsegm:opencv_bioinspired:opencv_ccalib:opencv_datasets:opencv_dnn_objdetect:opencv_dnn_superres:opencv_dpm:opencv_face:opencv_freetype:opencv_fuzzy:opencv_hdf:opencv_hfs:opencv_img_hash:opencv_intensity_transform:opencv_line_descriptor:opencv_mcc:opencv_optflow:opencv_phase_unwrapping:opencv_plot:opencv_quality:opencv_rapid:opencv_reg:opencv_rgbd:opencv_saliency:opencv_sfm:opencv_shape:opencv_signal:opencv_stereo:opencv_structured_light:opencv_superres:opencv_surface_matching:opencv_text:opencv_tracking:opencv_videostab:opencv_viz:opencv_wechat_qrcode:opencv_xfeatures2d:opencv_ximgproc:opencv_xobjdetect:opencv_xphoto:/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/Workspace/../../../Thirdparty/ORB_SLAM3/lib:/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/Workspace/../../../Thirdparty/ORB_SLAM3/Thirdparty/DBoW2/lib:/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/Workspace/../../../Thirdparty/ORB_SLAM3/Thirdparty/g2o/lib:/opt/ros/humble/lib:/usr/local/opencv-4.10.0/lib:/usr/local/lib:"
         NEW_RPATH "$ORIGIN:/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/Workspace/../../../Thirdparty/ORB_SLAM3/lib:/opt/ros/humble/lib:/usr/local/opencv-4.10.0/lib:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/orb_slam3/orb_slam3")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/build/orb_slam3/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
