# SDK开发环境配置
## 使用脚本进行环境配置
- 运行 `./setup.sh`
## 手动进行环境配置
- 运行 `./setup.sh` 以下载所有所需包并保存至 `Pack` 目录下
### OpenCV编译
- 新建并进入build `mkdir build && cd build`
- 依赖安装（无CUDA）
  ```
  sudo apt update && sudo apt install -y \
      build-essential cmake git pkg-config \
      libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
      libopenjp2-7-dev \
      libavcodec-dev libavformat-dev libswscale-dev \
      libv4l-dev libxvidcore-dev libx264-dev \
      libgtk2.0-dev libgtk-3-dev \
      libcanberra-gtk-module libcanberra-gtk3-module \
      libgdk-pixbuf2.0-dev \
      libtbb-dev libeigen3-dev \
      libatlas-base-dev gfortran \
      python3-dev python3-numpy python3-pip \
      libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
      libdc1394-dev libopenexr-dev \
      libhdf5-dev liblapack-dev \
      libprotobuf-dev protobuf-compiler \
      libgoogle-glog-dev libgflags-dev
      ```
- CMake配置（无CUDA@Opencv4.10.0）
  ```
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-4.10.0 \
        -D CMAKE_CXX_STANDARD=11 \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.10.0/modules \
        -D ENABLE_POPCNT=ON \
        -D ENABLE_FAST_MATH=ON \
        -D WITH_EIGEN=ON \
        -D WITH_TBB=ON \
        -D WITH_OPENMP=ON \
        -D WITH_IPP=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_V4L=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_GTK=ON \
        -D WITH_GTK=ON \
        -D WITH_GTK_2_X=ON \
        -D BUILD_opencv_highgui=ON \
        -D BUILD_opencv_imgcodecs=ON \
        -D BUILD_opencv_videoio=ON \
        -D WITH_OPENGL=ON \
        -D WITH_WEBP=ON \
        -D WITH_JPEG=ON \
        -D WITH_PNG=ON \
        -D WITH_TIFF=ON \
        -D WITH_QT=OFF \
        -D BUILD_EXAMPLES=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_opencv_python3=ON \
        ..
  ```
- CMake配置（JETSON@Opencv4.10.0）
  ```
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-4.10.0 \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.10.0/modules \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D CUDA_ARCH_BIN=8.7 \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_cudacodec=ON \
      -D BUILD_opencv_cudaarithm=ON \
      -D BUILD_opencv_cudaimgproc=ON \
      -D WITH_GTK=ON \
      -D WITH_V4L=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_TBB=ON \
      -D ENABLE_NEON=ON \
      -D WITH_IPP=OFF \
      ..
  ```
  - 编译 `make -j4`
  - 安装 `sudo make install`
  - 创建OpenCV库路径配置文件 `sudo sh -c 'echo "/usr/local/opencv-4.10.0/lib" > /etc/ld.so.conf.d/opencv-4.10.0.conf'`
### Pangolin编译
- 解包完成后进入 `Env/Pangolin-0.6/include/pangolin/gl/colour.h` 在头文件加入 `#include <limits>`
- 新建并进入build `mkdir build && cd build`
- 依赖安装
  ```
  sudo apt update && sudo apt install -y \
      build-essential cmake git pkg-config \
      libjpeg-dev libpng-dev libtiff-dev \
      libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
      libgtk2.0-dev \
      libtbb-dev libeigen3-dev \
      libatlas-base-dev gfortran \
      python3-dev python3-numpy \
      libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
      ```
- CMake配置 `cmake .. -DCMAKE_BUILD_TYPE=Release`
- 编译 `make -j4`
- 安装 `sudo make install`
- ## Sophus编译
- 新建并进入build `mkdir build && cd build`
- CMake配置 `cmake .. -DCMAKE_BUILD_TYPE=Release`
- 编译 `make -j4`
- 安装 `sudo make install`
### sigslot安装
- 新建并进入build `mkdir build && cd build`
- CMake配置 `cmake .. -DCMAKE_BUILD_TYPE=Release`
- 安装 `sudo make install`
### Eigen安装（无CUDA）
- 新建并进入build `mkdir build && cd build`
- CMake配置 `cmake .. -DCMAKE_BUILD_TYPE=Release`
- 安装 `sudo make install`

