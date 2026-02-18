#ifndef CAMERA_H
#define CAMERA_H
 
#include "initializer_list"
#include "libobsensor/ObSensor.hpp"
#include "opencv2/opencv.hpp"
 
class Camera {
  private:
    std::shared_ptr<ob::VideoStreamProfile> color_profile = nullptr;
    std::shared_ptr<ob::VideoStreamProfile> depth_profile = nullptr;
    std::shared_ptr<ob::VideoStreamProfile> ir_profile = nullptr;

    OBCameraIntrinsic color_intrinsics;
    OBCameraIntrinsic depth_intrinsics;
    OBCameraIntrinsic ir_intrinsics;
    OBCameraDistortion color_distortion;
    OBCameraDistortion depth_distortion;
    OBCameraDistortion ir_distortion;

    std::shared_ptr<ob::Device> device;
    std::shared_ptr<ob::Config> config;
    void init_color(int width = OB_WIDTH_ANY, int height = OB_HEIGHT_ANY, OBFormat format = OB_FORMAT_ANY, int fps = OB_FPS_ANY);
    void init_IR(int width = OB_WIDTH_ANY, int height = OB_HEIGHT_ANY, OBFormat format = OB_FORMAT_ANY, int fps = OB_FPS_ANY);
    void init_depth(int width = OB_WIDTH_ANY, int height = OB_HEIGHT_ANY, OBFormat format = OB_FORMAT_ANY, int fps = OB_FPS_ANY);
 
  public:
    std::shared_ptr<ob::Pipeline> pipe;
    
    Camera( bool color = false, int color_width = OB_WIDTH_ANY, int color_height = OB_HEIGHT_ANY, OBFormat color_format = OB_FORMAT_ANY, int color_fps = OB_FPS_ANY, 
            bool IR = false, int ir_width = OB_WIDTH_ANY, int ir_height = OB_HEIGHT_ANY, OBFormat ir_format = OB_FORMAT_ANY, int ir_fps = OB_FPS_ANY,
            bool depth = false, int depth_width = OB_WIDTH_ANY, int depth_height = OB_HEIGHT_ANY, OBFormat depth_format = OB_FORMAT_ANY, int depth_fps = OB_FPS_ANY);
    inline ~Camera() { this->pipe->stop(); }
    inline void start() { this->pipe->start(this->config); }
    inline void stop() { this->pipe->stop(); }
    inline std::shared_ptr<ob::FrameSet> get() { return pipe ? pipe->waitForFrames(400) : nullptr; }
    static cv::Mat frame2mat(const std::shared_ptr<ob::VideoFrame> &frame);
    // 获取相机内参
    OBCameraIntrinsic get_color_intrinsic();    
    OBCameraIntrinsic get_depth_intrinsic();  
    OBCameraIntrinsic get_ir_intrinsic();  
    // 获取相机外参
    OBCameraDistortion get_color_distortion();
    OBCameraDistortion get_depth_distortion();
    OBCameraDistortion get_ir_distortion();
};
 
#endif