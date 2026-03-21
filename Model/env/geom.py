import genesis
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

import env.util as util
import env.pid as pid

class geom:
    """
        camera_pos:相机位置:(x, y, z)
        camera_lookat:相机方向:(x, y, z)
        camera_fov:视场角:度
        max_FPS:最大FPS
        show_viewer:显示
        dt:步长:s
        device:设备:cpu/gpu/cuda
    """
    def __init__(self, camera_pos, camera_lookat, camera_fov, max_FPS, show_viewer, dt, device):
        if device == "cpu":
            genesis.init(backend=genesis.cpu)
        elif device == "cuda":
            genesis.init(backend=genesis.cuda)
        self._viewer_options = genesis.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=camera_fov,
            max_FPS=max_FPS,
        )
        self._scene = genesis.Scene(
            sim_options=genesis.options.SimOptions(
                dt=dt,
            ),
            viewer_options=self._viewer_options,
            show_viewer=show_viewer,
        )
        self._plane = self._scene.add_entity(
            genesis.morphs.Plane(
                visualization=True,   # 显示地面
                collision=True        # 有碰撞效果
            ),
        )
        self._device = device
        self.drones_init_pos_list = []
        self.drones_init_quat_list = []
        self.drones_list = []
        self.depths_list = []
        self.T_max_list = []
        self.roll_pid_list = []
        self.pitch_pid_list = []
        self.yaw_pid_list = []
        self.cylinders_list = []
        self.boxes_list = []

    """
        urdf_path:URDF路径
        drone_init_pos:无人机初始位置:torch.tensor([x, y, z], dtype=torch.double):m
        drone_init_euler:无人机初始姿态欧拉角:rad
        T_max:无人机最大推力:N
        T_max_att:无人机最大推力衰减:0.0-1.0
        ang_vel_max:角速度最大值:torch.tensor([roll, pitch, yaw], dtype=torch.double):rad/s
        res_W:分辨率像素宽度
        res_H:分辨率像素高度
        depth_pos_offset:深度相机的pos相对于机器人pos的偏置:torch.tensor([x, y, z], dtype=torch.double):m
        depth_euler_offset:深度相机初始姿态欧拉角:rad
        depth_fov_H:水平视场角:度
        depth_fov_V:垂直视场角:度
        num:同一场景实体数量
    """
    def add_drone(self, urdf_path, drone_init_pos, drone_init_euler, T_max, T_max_att, ang_vel_max, res_W, res_H, depth_pos_offset, depth_euler_offset,depth_fov_H , depth_fov_V, num):
        self._num = num
        drone_R = util.euler_to_R(drone_init_euler)
        if self._device == 'cuda':
            # DRONE
            drone_pos = drone_init_pos.cpu().numpy()
            drone_euler = util.rad_to_angle(drone_init_euler).cpu().numpy()
            drone_quat = util.euler_to_quat(drone_init_euler).cpu().numpy()
            # DEPTH
            depth_pos_offset_np = depth_pos_offset.cpu().numpy()
            depth_euler_offset_np = util.rad_to_angle(depth_euler_offset).cpu().numpy()
        else:
            # DRONE
            drone_pos = drone_init_pos.numpy()
            drone_euler = util.rad_to_angle(drone_init_euler).numpy()
            drone_quat = util.euler_to_quat(drone_init_euler).numpy()
            # DEPTH
            depth_pos_offset_np = depth_pos_offset.numpy()
            depth_euler_offset_np = util.rad_to_angle(depth_euler_offset).numpy()
        drone = self._scene.add_entity(
            morph=genesis.morphs.Drone(
                file=urdf_path,
                pos=drone_pos,
                euler=drone_euler,
            ),
        )
        depth_kwargs = dict(
            entity_idx=drone.idx,
            pos_offset=depth_pos_offset_np,
            euler_offset=depth_euler_offset_np,
            return_world_frame=False,
            draw_debug=False,
        )
        depth = self._scene.add_sensor(
            genesis.sensors.DepthCamera(
                pattern=genesis.sensors.DepthCameraPattern(
                    res=(res_W, res_H), # 图像分辨率（宽，高）
                    fov_horizontal=depth_fov_H, # 水平视场角度
                    fov_vertical=depth_fov_V,
                ), 
                **depth_kwargs
            )
        )
        roll_pid = pid.pid(0.1, 0.0, 0.8, 0.0, ang_vel_max[0])
        pitch_pid = pid.pid(0.1, 0.0, 0.8, 0.0, ang_vel_max[1])
        yaw_pid = pid.pid(0.1, 0.0, 0.8, 0.0, ang_vel_max[2])
        self.drones_init_pos_list.append(drone_pos)
        self.drones_init_quat_list.append(drone_quat)
        self.drones_list.append(drone)
        self.depths_list.append(depth)
        self.T_max_list.append(T_max-T_max*T_max_att)
        self.roll_pid_list.append(roll_pid)
        self.pitch_pid_list.append(pitch_pid)
        self.yaw_pid_list.append(yaw_pid)

    """
        添加圆柱障碍
        num:圆柱数量
        param:圆柱参数:torch.tensor([x, y, z, H, R]/[[x, y, z, H, R], ...], dtype=torch.double):m
    """
    def add_cylinders(self, num, param):
        if self._device == 'cuda':
            param_np = param.cpu().numpy()
        else:
            param_np = param.numpy()
        if num == 1:
            cylinder = self._scene.add_entity(
                genesis.morphs.Cylinder(
                    height=param_np[3],
                    radius=param_np[4],
                    pos=(param_np[0], param_np[1], param_np[2]),
                    fixed=True,
                )
            )
            self.cylinders_list.append(cylinder)
        elif num > 1:
            for i in range(num):
                cylinder = self._scene.add_entity(
                    genesis.morphs.Cylinder(
                        height=param_np[i][3],
                        radius=param_np[i][4],
                        pos=(param_np[i][0], param_np[i][1], param_np[i][2]),
                        fixed=True,
                    )
                )
                self.cylinders_list.append(cylinder)

    """
        添加方块障碍
        num:方块数量
        param:方块参数:torch.tensor([x, y, z, L, W, H]/[[x, y, z, L, W, H]], ...], dtype=torch.double):m
    """
    def add_boxes(self, num, param):
        if self._device == 'cuda':
            param_np = param.cpu().numpy()
        else:
            param_np = param.numpy()
        if num == 1:
            box = self._scene.add_entity(
                genesis.morphs.Box(
                    size=(param_np[3], param_np[4], param_np[5]),
                    pos=(param_np[0], param_np[1], param_np[2]),
                    fixed=True,
                )
            )
            self.boxes_list.append(box)
        elif num > 1:
            for i in range(num):
                self._scene.add_entity(
                    genesis.morphs.Box(
                        size=(param_np[i][3], param_np[i][4], param_np[i][5]),
                        pos=(param_np[i][0], param_np[i][1], param_np[i][2]),
                        fixed=True,
                    )
                )
                self.boxes_list.append(box)

    """
        构建场景
    """
    def build(self):
        self._scene.build(n_envs=0)

    """
        继续下一步
        pred_euler:下一步姿态欧拉角
        pred_thrust:下一步推力
    """
    def step(self, pred_euler, pred_thrust):
        for i in range(len(self.drones_list)):
            current_quat = self.drones_list[i].get_quat()
            current_euler = util.quat_to_euler(current_quat)
            # PID控制器计算执行的角速度
            roll_vel = self.roll_pid_list[i].position(current_euler[0], pred_euler[0])
            pitch_vel = self.pitch_pid_list[i].position(current_euler[1], pred_euler[1])
            yaw_vel = self.yaw_pid_list[i].position(current_euler[2], pred_euler[2])
            current_R = util.euler_to_R(current_euler)
            # 施加力
            drone_up = current_R[:, 2]
            T_current = self.T_max_list[i]*pred_thrust*drone_up
            self.drones_list[i].control_dofs_force([T_current[0], T_current[1], T_current[2], roll_vel, pitch_vel, yaw_vel])
            # 显示深度图
            depth_data = self.depths_list[i].read_image().cpu().numpy()
            # 方法1：归一化到 uint8
            # if depth_data.dtype == np.float32 or depth_data.dtype == np.float64:
            #     # 归一化到 0-255
            #     depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            #     depth_display = depth_normalized.astype(np.uint8)
            #     cv2.imshow('Depth', depth_display)
            #     cv2.waitKey(1)
        self._scene.step()
        if len(self.drones_list[i].detect_collision()) > 0:
            return True
        else:
            return False
        
    """
        复位
    """
    def reset(self):
        for i in range(len(self.drones_list)):
            self.drones_list[i].set_pos(self.drones_init_pos_list[i])
            self.drones_list[i].set_quat(self.drones_init_quat_list[i])
            self.roll_pid_list[i].reset()
            self.pitch_pid_list[i].reset()
            self.yaw_pid_list[i].reset()

    """
        获取位置
        返回:ENU位置
    """
    @property
    def pos(self):
        pos_list = []
        for i in range(len(self.drones_list)):
            pos_list.append(self.drones_list[i].get_pos())
        return torch.stack(pos_list)

    """
        获取速度
        返回:FLU线速度
    """
    @property
    def vel(self):
        vel_list = []
        for i in range(len(self.drones_list)):
            vel_list.append(self.drones_list[i].get_vel())
        return torch.stack(vel_list)
    
    """
        获取角速度
        返回:FLU角速度
    """
    @property
    def ang_vel(self):
        ang_vel_list = []
        for i in range(len(self.drones_list)):
            ang_vel_list.append(self.drones_list[i].get_ang())
        return torch.stack(ang_vel_list)
    
    """
        获取姿态四元数
        返回:FLU-ENU四元数
    """
    @property
    def quat(self):
        quat_list = []
        for i in range(len(self.drones_list)):
            quat_list.append(self.drones_list[i].get_quat())
        return torch.stack(quat_list)
    
    """
        获取深度图
        返回:深度图
    """
    @property
    def depth_img(self):
        depth_img_list = []
        for i in range(len(self.depths_list)):
            depth_img_list.append(self.depths_list[i].read_image())
        return torch.stack(depth_img_list)