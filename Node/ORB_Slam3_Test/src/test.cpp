//UNIX
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
// OPENCV
#include "opencv2/opencv.hpp"
// Orbbec-Beidge
#include "orbbec_bridge.h"
// ORBBEC
#include "libobsensor/ObSensor.hpp"
// ORB-SLAM3
#include "orb_slam3.h"

// 解决Eigen内存对齐问题（关键！）
#include <Eigen/Dense>
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::SE3f)

int main(int argc, char** argv){
    Camera gemini=( true, 640, 480, OB_FORMAT_RGB888, 60, 
                    false, 640, 480, OB_FORMAT_Y8, 60, 
                    true, 640, 480, OB_FORMAT_Y12, 60);
    gemini.start();
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,true);

    while(true){
        auto frame_set = gemini.get();
        if(frame_set != nullptr){
            auto rgb_frame = frame_set->colorFrame();
            auto depth_frame = frame_set->depthFrame();
            if(rgb_frame != nullptr && depth_frame != nullptr){
                cv::Mat rgb_image = gemini.frame2mat(rgb_frame);
                cv::Mat depth_image = gemini.frame2mat(depth_frame);
                auto now = std::chrono::system_clock::now();
                auto duration = now.time_since_epoch();
                double tf_frame = std::chrono::duration<double, std::micro>(duration).count();
                SLAM.TrackRGBD(rgb_image,depth_image,tf_frame);
            }
        }
    }
    
    return 0;
}