// UNIX
#include <iostream>
#include "libobsensor/ObSensor.hpp"
#include <memory>
#include <opencv2/opencv.hpp>

#include "orbbec_bridge.h"
 
constexpr bool FLIP = false;
 
void Camera::init_color(int width, int height, OBFormat format, int fps) {
    // 获取彩色相机的所有流配置，包括流的分辨率，帧率，以及帧的格式
    auto profiles = this->pipe->getStreamProfileList(OB_SENSOR_COLOR);
    // 获取传感器内参（比较狗屎的设计，如果是高分辨率的时候获取内参会报错）
    color_profile = profiles->getVideoStreamProfile(OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_ANY, OB_FPS_ANY);
    color_intrinsics = color_profile->getIntrinsic();
    color_distortion = color_profile->getDistortion();
    try {
        // 根据指定的格式查找对应的Profile,优先选择RGB888格式
        color_profile = profiles->getVideoStreamProfile(width, height, format, fps);
    } catch (const ob::Error &) {
        std::cout << "COLOR ERROR" << std::endl;
        // 没找到RGB888格式后不匹配格式查找对应的Profile进行开流
        color_profile = profiles->getVideoStreamProfile(OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_ANY, OB_FPS_ANY);
    }
    color_intrinsics.width = color_profile->width();
    color_intrinsics.height = color_profile->height();
    // 开启/关闭彩色相机的镜像模式
    if (this->device->isPropertySupported(OB_PROP_COLOR_MIRROR_BOOL, OB_PERMISSION_WRITE)) {
        this->device->setBoolProperty(OB_PROP_COLOR_MIRROR_BOOL, FLIP);
    }
    // 开启彩色流
    this->config->enableStream(color_profile);
}
 
void Camera::init_depth(int width, int height, OBFormat format, int fps) {
    auto profiles = this->pipe->getStreamProfileList(OB_SENSOR_DEPTH);
    // 获取传感器内参（比较狗屎的设计，如果是高分辨率的时候获取内参会报错）
    depth_profile = profiles->getVideoStreamProfile(OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_ANY, OB_FPS_ANY);
    depth_intrinsics = depth_profile->getIntrinsic();
    depth_distortion = depth_profile->getDistortion();
    try {
        // 根据指定的格式查找对应的Profile,优先查找Y16格式
        depth_profile = profiles->getVideoStreamProfile(width, height, format, fps);
    } catch (const ob::Error &) {
        std::cout << "DEPTH ERROR" << std::endl;
        // 没找到Y16格式后不匹配格式查找对应的Profile进行开流
        depth_profile = profiles->getVideoStreamProfile(OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_ANY, OB_FPS_ANY);
    }
    depth_intrinsics.width = depth_profile->width();
    depth_intrinsics.height = depth_profile->height();
    // 开启/关闭深度相机的镜像模式
    if (this->device->isPropertySupported(OB_PROP_DEPTH_MIRROR_BOOL, OB_PERMISSION_WRITE)) {
        this->device->setBoolProperty(OB_PROP_DEPTH_MIRROR_BOOL, FLIP);
    }
    // 开启深度流
    this->config->enableStream(depth_profile);
}
 
void Camera::init_IR(int width, int height, OBFormat format, int fps) {
    // 获取红外相机的所有流配置，包括流的分辨率，帧率，以及帧的格式
    auto profiles = pipe->getStreamProfileList(OB_SENSOR_IR);
    // 获取传感器内参（比较狗屎的设计，如果是高分辨率的时候获取内参会报错）
    ir_profile = profiles->getVideoStreamProfile(OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_ANY, OB_FPS_ANY);
    ir_intrinsics = ir_profile->getIntrinsic();
    ir_distortion = ir_profile->getDistortion();
    try {
        // 根据指定的格式查找对应的Profile,优先查找Y16格式
        ir_profile = profiles->getVideoStreamProfile(width, height, format, fps);
    } catch (const ob::Error &) {
        // 没找到Y16格式后不匹配格式查找对应的Profile进行开流
        ir_profile = profiles->getVideoStreamProfile(OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_ANY, OB_FPS_ANY);
    }
    ir_intrinsics.width = ir_profile->width();
    ir_intrinsics.height = ir_profile->height();
    // 开启/关闭红外相机的镜像模式
    if (this->device->isPropertySupported(OB_PROP_IR_MIRROR_BOOL, OB_PERMISSION_WRITE)) {
        this->device->setBoolProperty(OB_PROP_IR_MIRROR_BOOL, FLIP);
    }
    // 开启红外流
    this->config->enableStream(ir_profile);
}
 
Camera::Camera( bool color, int color_width, int color_height, OBFormat color_format, int color_fps, 
                bool IR, int ir_width, int ir_height, OBFormat ir_format, int ir_fps,
                bool depth, int depth_width, int depth_height, OBFormat depth_format, int depth_fps)
    : pipe(std::make_shared<ob::Pipeline>()), config(std::make_shared<ob::Config>()) {
    this->device = pipe->getDevice();
    if (color) {
        this->init_color(color_width, color_height, color_format, color_fps);
    }
    if (IR) {
        this->init_IR(ir_width, ir_height, ir_format, ir_fps);
    }
    if (depth) {
        this->init_depth(depth_width, depth_height, depth_format, depth_fps);
    }
    // 若摄像头支持多种流，则开启流同步
    if (color + IR + depth > 1) {
        if (device->isPropertySupported(OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL, OB_PERMISSION_READ)) {
            config->setAlignMode(ALIGN_D2C_HW_MODE);
        } else {
            config->setAlignMode(ALIGN_D2C_SW_MODE);
        }
        if(color_profile->width() > 640 || depth_profile->width() > 640){
            config->setAlignMode(ALIGN_DISABLE); // 需要点云时，如果要彩色或深度图像超过640*480的话，必须关闭流同步，但是会卡
        }
    }
    // 如果有新增设备，则开启流；如果有设备拔出，则关闭流。
    ob::Context ctx;
    ctx.setDeviceChangedCallback([this](std::shared_ptr<ob::DeviceList> removed_devices,
                                        std::shared_ptr<ob::DeviceList> added_devices) {
        if (added_devices->deviceCount() > 0) {
            this->pipe = std::make_shared<ob::Pipeline>();
            this->start();
        } else if (removed_devices->deviceCount() > 0) {
            this->stop();
        }
    });
}
 
cv::Mat Camera::frame2mat(const std::shared_ptr<ob::VideoFrame> &frame) {
    if (frame == nullptr || frame->dataSize() < 1024) {
        return {};
    }
 
    const OBFrameType frame_type = frame->type();               // 帧类型（彩色/深度/IR）
    const OBFormat frame_format = frame->format();              // 图像格式
    const int frame_height = static_cast<int>(frame->height()); // 图像高度
    const int frame_width = static_cast<int>(frame->width());   // 图像宽度
    void *const frame_data = frame->data();                     // 帧原始数据首地址
    const int data_size = static_cast<int>(frame->dataSize());  // 帧数据大小
 
    cv::Mat result_mat;
 
    if (frame_type == OB_FRAME_COLOR) {
        // Color image
        if (frame_format == OB_FORMAT_MJPG) {
            const cv::Mat raw_mat(1, data_size, CV_8UC1, frame_data);
            result_mat = cv::imdecode(raw_mat, 1);
        } else if (frame_format == OB_FORMAT_NV21) {
            const cv::Mat raw_mat(frame_height * 3 / 2, frame_width, CV_8UC1, frame_data);
            cv::cvtColor(raw_mat, result_mat, cv::COLOR_YUV2BGR_NV21);
        } else if (frame_format == OB_FORMAT_YUYV || frame_format == OB_FORMAT_YUY2) {
            const cv::Mat raw_mat(frame_height, frame_width, CV_8UC2, frame_data);
            cv::cvtColor(raw_mat, result_mat, cv::COLOR_YUV2BGR_YUY2);
        } else if (frame_format == OB_FORMAT_RGB888) {
            const cv::Mat raw_mat(frame_height, frame_width, CV_8UC3, frame_data);
            cv::cvtColor(raw_mat, result_mat, cv::COLOR_RGB2BGR);
        } else if (frame_format == OB_FORMAT_UYVY) {
            const cv::Mat raw_mat(frame_height, frame_width, CV_8UC2, frame_data);
            cv::cvtColor(raw_mat, result_mat, cv::COLOR_YUV2BGR_UYVY);
        }
    } else if (frame_format == OB_FORMAT_Y16 || frame_format == OB_FORMAT_Y11 || frame_format == OB_FORMAT_Y12 || frame_format == OB_FORMAT_YUYV ||
               frame_format == OB_FORMAT_YUY2) {
        // IR or depth image
        const cv::Mat raw_mat(frame_height, frame_width, CV_16UC1, frame_data);
        result_mat = raw_mat.clone();
        // const double scale =
        //     1 / pow(2, frame->pixelAvailableBitSize() - (frame_type == OB_FRAME_DEPTH ? 10 : 8));
        // cv::convertScaleAbs(raw_mat, result_mat, scale);
    } else if (frame_type == OB_FRAME_IR) {
        // IR image
        if (frame_format == OB_FORMAT_Y8) {
            result_mat = cv::Mat(frame_height, frame_width, CV_8UC1, frame_data);
        } else if (frame_format == OB_FORMAT_MJPG) {
            const cv::Mat raw_mat(1, data_size, CV_8UC1, frame_data);
            result_mat = cv::imdecode(raw_mat, 1);
        }
    }
    return result_mat;
}

OBCameraIntrinsic Camera::get_color_intrinsic(){
    return color_intrinsics;
}

OBCameraIntrinsic Camera::get_depth_intrinsic(){
    return depth_intrinsics;
}

OBCameraIntrinsic Camera::get_ir_intrinsic(){
    return ir_intrinsics;
}

OBCameraDistortion Camera::get_color_distortion(){
    return color_distortion;
}

OBCameraDistortion Camera::get_depth_distortion(){
    return depth_distortion;
}

OBCameraDistortion Camera::get_ir_distortion(){
    return ir_distortion;
}

