# 第三方库（Node中最重要的包）
## 脚本编译安装
- 运行 `./setup.sh` 自动进行编译安装
## OrbbecSDK
### 安装USB规则
- 进入 `OrbbecSDK/Script` ,运行 `sudo ./install_udev_rules.sh`
## ORB-SLAM3
### 版本控制
|库名称|最低版本要求|SDK使用版本|备注|
|:-:|:-:|:-:|:-:|
|<b>OpenCV</b>|<b>3.2</b>|<b>4.10.0</b>|<b>无，不建议使用更高版本</b>|
|<b>Eigen3</b>|<b>3.34</b>|<b>3.37</b>|<b>无</b>|
|<b>Boost</b>|<b>1.65.1</b>|<b>1.83.0</b>|<b>UBUNTU APT安装</b>|
|<b>sigslot</b>|<b>1.0.0</b>|<b>1.0.0</b>|<b>高版本会报`m_slots`错误</b>|
|<b>Pangolin</b>|<b>0.6</b>|<b>0.6</b>|<b>无</b>|
|<b>Sophus</b>|<b>1.22</b>|<b>1.22</b>|<b>无</b>|
|<b>OpenGL</b>|<b>3.0</b>|<b>4.5</b>|<b>无</b>|
- 推荐使用 <b>Ubuntu22.04 jammy</b>
### 依赖编译
- 详见 [依赖编译](../Env/README.md)
### ORB-SLAM3编译（查库路径为ORB-SLAM3工程Thirdparty）
- 运行 `./build.sh`
### ORB-SLAM3测试
- 将 `Thirdparty/Pack/dataset-room4_512_16.tar` 复制到 `Thirdparty/ORB_Slam3/Examples/Monocular` 下并解包
- 在 `Thirdparty/ORB_Slam3/Examples/Monocular` 运行 `./mono_tum_vi ../../Vocabulary/ORBvoc.txt TUM-VI.yaml  dataset-room4_512_16/dso/cam0/images TUM_TimeStamps/dataset-room4_512.txt dataset-corridor1_512_mono`
- 可参考该文章 [ORB-SLAM3测试TUM-VI数据集](https://blog.csdn.net/2402_83452538/article/details/149153262)
