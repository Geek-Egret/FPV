import numpy as np
import genesis as gs
import genesis.utils.geom as gu
import torch
import math

from drone import pid

class sim_controller:
    """
    paraments:
        drone_mass: drone mass
    """
    def __init__(self, entity, frame="world", dt=0.01, base_rpm=14468.429183500699): 
        self.frame = frame          # 参考坐标系（world/base）
        self.dt = dt                # 每步间隔时间
        self.KF = entity.KF           # 推力系数
        self.KM = entity.KM           # 力矩系数
        self.mass = entity.get_mass()      # 重量
        self.propellers_spin = entity.propellers_spin  # 旋转方向
        self.base_rpm = base_rpm

        # 前后翻滚速度环
        self.kp_vel_pitch = 7.0
        self.ki_vel_pitch = 0.2
        self.kd_vel_pitch = 40.0
        self.sigma_limit_vel_pitch = 0.02
        self.output_limit_vel_pitch = 0.35 # 30*math.pi/180.0=0.5235
        # 前后翻滚角速度环
        self.kp_ang_vel_pitch = 0.015
        self.ki_ang_vel_pitch = 0.002
        self.kd_ang_vel_pitch = 0.02
        self.sigma_limit_ang_vel_pitch = 0.02
        self.output_limit_ang_vel_pitch = 0.0014   # 0.0014
        # 左右翻滚速度环
        self.kp_vel_roll = 7.0
        self.ki_vel_roll = 0.2
        self.kd_vel_roll = 40.0
        self.sigma_limit_vel_roll = 0.02
        self.output_limit_vel_roll = 0.35
        # 左右翻滚角速度环
        self.kp_ang_vel_roll = 0.015
        self.ki_ang_vel_roll = 0.002
        self.kd_ang_vel_roll = 0.025
        self.sigma_limit_ang_vel_roll = 0.02
        self.output_limit_ang_vel_roll = 0.0014
        # 左右旋转速度环
        self.kp_vel_yaw = 0.015
        self.ki_vel_yaw = 0.02
        self.kd_vel_yaw = 0.025
        self.sigma_limit_vel_yaw = 0.02
        self.output_limit_vel_yaw = 0.5 # 0.5
        # 升降速度环
        self.kp_vel_throttle = 3.0
        self.ki_vel_throttle = 1.1
        self.kd_vel_throttle = 1.8
        self.sigma_limit_vel_throttle = 0.02
        self.output_limit_vel_throttle = 1.0    # 0.3

        self.vel_pitch_pid = pid.pid(self.kp_vel_pitch, self.ki_vel_pitch, self.kd_vel_pitch, self.sigma_limit_vel_pitch, self.output_limit_vel_pitch)
        self.ang_vel_pitch_pid = pid.pid(self.kp_ang_vel_pitch, self.ki_ang_vel_pitch, self.kd_ang_vel_pitch, self.sigma_limit_ang_vel_pitch, self.output_limit_ang_vel_pitch)
        self.vel_roll_pid = pid.pid(self.kp_vel_roll, self.ki_vel_roll, self.kd_vel_roll, self.sigma_limit_vel_roll, self.output_limit_vel_roll)
        self.ang_vel_roll_pid = pid.pid(self.kp_ang_vel_roll, self.ki_ang_vel_roll, self.kd_ang_vel_roll, self.sigma_limit_ang_vel_roll, self.output_limit_ang_vel_roll)
        self.vel_yaw_pid = pid.pid(self.kp_vel_yaw, self.ki_vel_yaw, self.kd_vel_yaw, self.sigma_limit_vel_yaw, self.output_limit_vel_yaw)
        self.vel_throttle_pid = pid.pid(self.kp_vel_throttle, self.ki_vel_throttle, self.kd_vel_throttle, self.sigma_limit_vel_throttle, self.output_limit_vel_throttle)

        # self.vel_pitch_pid.set_plot(name="vel pitch")
        # self.ang_vel_pitch_pid.set_plot(name="ang vel pitch")
        # self.vel_roll_pid.set_plot(name="vel roll")
        # self.ang_vel_roll_pid.set_plot(name="ang vel roll")

    def set_control_target(self, entity, exp_vx=0.0, exp_vy=0.0, exp_vz=0.0, yaw_rate=0.0):
        self.exp_yaw_rate = yaw_rate
        self.rt_vel_world = entity.get_vel()    # 以world为参考系
        exp_vel = torch.tensor(np.array([exp_vx, exp_vy, exp_vz]), dtype=torch.float32)
        # 获取base坐标系相对于world的四元数、欧拉角和旋转矩阵
        self.base_quat = entity.get_quat()
        self.base_euler = gu.quat_to_xyz(self.base_quat) 
        self.R_base_world = gu.quat_to_R(self.base_quat)
        # 获取base坐标系相对于world的角速率
        self.base_ang = entity.get_ang() # roll pitch yaw
        # 将实时速度转换到world坐标系下
        self.rt_vel_base = torch.matmul(self.R_base_world.T, self.rt_vel_world)
        # 将 exp_vel 转换到world/base坐标系下
        if(self.frame == "world"):
            self.exp_vel_world = exp_vel.clone()
            self.exp_vel_base = torch.matmul(self.R_base_world.T, exp_vel)
        elif(self.frame == "base"):
            self.exp_vel_base = exp_vel.clone()
            self.exp_vel_world = torch.matmul(self.R_base_world, exp_vel)
        else:
            print(f"[WARNING] Unknown frame name of {self.frame}")
        # print(self.R_base_world)
        print(self.frame)
        print(f"rt_vel_x: {self.rt_vel_base[0].item():.6f} {self.rt_vel_world[0].item():.6f}")
        print(f"rt_vel_y: {self.rt_vel_base[1].item():.6f} {self.rt_vel_world[1].item():.6f}")
        # print(f"rt_vel_z: {self.rt_vel_base[2].item():.6f}")
        # print(f"rt_roll: {self.base_ang[0].item():.6f}")
        # print(f"rt_pitch: {self.base_ang[1].item():.6f}")
        # print(f"rt_roll: {self.base_ang[0].item():.6f}")
        # print(f"rt_yaw: {self.base_ang[2].item():.6f}")
        print(f"exp_vel_x: {self.exp_vel_base[0]} {self.exp_vel_world[0]}")
        print(f"exp_vel_y: {self.exp_vel_base[1]} {self.exp_vel_world[1]}")
        # print(self.exp_vel_world[2])
    
    """
        默认无人机构型：
        1   0       ^
          X         |
        2   3
    """
    def sim_control(self, M0_Mx=0, M1_Mx=1, M2_Mx=2, M3_Mx=3):
        motor_rpm =  np.array([self.base_rpm, self.base_rpm, self.base_rpm, self.base_rpm], dtype=np.float32)
        motor_rpm_mapping =  np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.vel_pitch = self.vel_pitch_pid.incremental(self.rt_vel_base[0].item(), self.exp_vel_base[0].item())
        self.rpm_pitch = self.ang_vel_pitch_pid.incremental(self.base_ang[1].item(), self.vel_pitch)
        motor_rpm[0] -= self.base_rpm*self.rpm_pitch
        motor_rpm[1] -= self.base_rpm*self.rpm_pitch
        motor_rpm[2] += self.base_rpm*self.rpm_pitch
        motor_rpm[3] += self.base_rpm*self.rpm_pitch

        self.vel_roll = -self.vel_roll_pid.incremental(self.rt_vel_base[1].item(), self.exp_vel_base[1].item())
        self.rpm_roll = self.ang_vel_roll_pid.incremental(self.base_ang[0].item(), self.vel_roll)
        motor_rpm[0] -= self.base_rpm*self.rpm_roll
        motor_rpm[1] += self.base_rpm*self.rpm_roll
        motor_rpm[2] += self.base_rpm*self.rpm_roll
        motor_rpm[3] -= self.base_rpm*self.rpm_roll

        self.rpm_yaw = self.vel_yaw_pid.incremental(self.base_ang[2].item(), self.exp_yaw_rate)
        motor_rpm[0] -= self.base_rpm*self.rpm_yaw
        motor_rpm[1] += self.base_rpm*self.rpm_yaw
        motor_rpm[2] -= self.base_rpm*self.rpm_yaw
        motor_rpm[3] += self.base_rpm*self.rpm_yaw

        self.rpm_throttle = self.vel_throttle_pid.incremental(self.rt_vel_base[2].item(), self.exp_vel_base[2].item())
        motor_rpm[0] += self.base_rpm*self.rpm_throttle
        motor_rpm[1] += self.base_rpm*self.rpm_throttle
        motor_rpm[2] += self.base_rpm*self.rpm_throttle
        motor_rpm[3] += self.base_rpm*self.rpm_throttle

        motor_rpm_mapping[M0_Mx] = motor_rpm[0]
        motor_rpm_mapping[M1_Mx] = motor_rpm[1]
        motor_rpm_mapping[M2_Mx] = motor_rpm[2]
        motor_rpm_mapping[M3_Mx] = motor_rpm[3]

        # self.vel_pitch_pid.start_plot(dt=self.dt)
        # self.ang_vel_pitch_pid.start_plot(dt=self.dt)
        # self.vel_roll_pid.start_plot(dt=self.dt)
        # self.ang_vel_roll_pid.start_plot(dt=self.dt)
        # print(f"pitch_output_0: {self.vel_pitch:.6f}")
        # print(f"pitch_output_1: {self.rpm_pitch:.6f}")
        # print(f"roll_output_0: {self.vel_roll:.6f}")
        # print(f"roll_output_1: {self.rpm_roll:.6f}")
        # print(f"roll_output: {self.rpm_roll:.6f}")
        # print(f"yaw_output: {self.rpm_yaw:.6f}")
        # print(f"throttle_output: {self.rpm_throttle:.6f}")
        # print(motor_rpm)
        # print(motor_rpm_mapping)
        
        return motor_rpm_mapping
        # vx_error = exp_vx - current_vx
        # vy_error = exp_vy - current_vy
        # vz_error = exp_vz - current_vz

        # # 通过PID计算期望的俯仰角、滚转角和总推力
        # desired_pitch = pitch_pid(vx_error)  # 前后速度控制俯仰角
        # desired_roll = roll_pid(vy_error)    # 左右速度控制滚转角
        # desired_thrust = thrust_pid(vz_error) # 垂直速度控制总推力
