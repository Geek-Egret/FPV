import torch
import cv2
import numpy as np
import random
import time
import math

import env.util as util
import env.geom as geom
import env.robot as robot
import env.sensor as sensor

device = 'cuda'
episodes = 50000
steps = 300
act = torch.tensor(
    [[[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 0.9]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.2]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]]], 
    dtype=torch.float, device=device, requires_grad=True)
""" GEOM设置 """
batch_size = 16
# 地形域随机化
sphere_dict = {'num':5, 'x_min':1.0, 'x_max':5.0, 'y_min':-3.0, 'y_max':3.0, 'z_min':1.0, 'z_max':2.0, 'R_min':0.3, 'R_max':1.0}
cylinder_dict = {'num':5, 'x_min':1.0, 'x_max':5.0, 'y_min':-3.0, 'y_max':3.0, 'z_min':1.0, 'z_max':2.0, 'R_min':0.3, 'R_max':1.0}
""" 机器人 """
init_pos = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float, device=device, requires_grad=True)
init_euler = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
mass = 0.33
T_max = 0.33*9.81   # 4*0.40*9.81
collision_radius = 0.072
""" 深度相机 """
pos_offset = torch.tensor([0.0425, 0.0, 0.0345], dtype=torch.float, device=device, requires_grad=True)
euler_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
res_W = 40
res_H = 25
fov_H = 67.
fov_V = 45.
min_depth = 0.2
max_depth = 2.
# 传感器域随机化
noise_range = {'min':0.0, 'max':0.0}
black_hole_prob = 0.0

geom = geom.geom(
    batch_size=batch_size,
    device=device,
)
drone_robot = robot.drone(
    device = device,
    init_pos = init_pos, 
    init_euler = init_euler, 
    mass = mass, 
    T_max = T_max, 
    collision_radius = collision_radius
)
cloest_dist = sensor.cloest_dist(
    device = device
)
depth = sensor.depth(
    device = device,
    pos_offset = pos_offset, 
    euler_offset = euler_offset,  
    res_W = res_W, 
    res_H = res_H, 
    fov_H = fov_H,
    fov_V = fov_V,
    min_depth = min_depth,
    max_depth = max_depth,
    noise_range = noise_range,
    black_hole_prob = black_hole_prob
)
drone_robot.sensor_bind(cloest_dist)
drone_robot.sensor_bind(depth)
geom.add_robot(drone_robot)

for idx in range(batch_size):
    for sphere in range(sphere_dict['num']):
        geom.add_sphere(
            torch.tensor(
                [
                    random.uniform(sphere_dict['x_min'], sphere_dict['x_max']), 
                    random.uniform(sphere_dict['y_min'], sphere_dict['y_max']), 
                    random.uniform(sphere_dict['z_min'], sphere_dict['z_max']), 
                    random.uniform(sphere_dict['R_min'], sphere_dict['R_max'])
                ], 
                dtype=torch.float, 
                device=device
            ),
            idx = idx
        )
    for cylinder in range(cylinder_dict['num']):
        z = random.uniform(sphere_dict['z_min'], sphere_dict['z_max'])
        geom.add_cylinder(
            torch.tensor(
                [
                    random.uniform(sphere_dict['x_min'], sphere_dict['x_max']), 
                    random.uniform(sphere_dict['y_min'], sphere_dict['y_max']), 
                    z, 
                    random.uniform(sphere_dict['R_min'], sphere_dict['R_max']),
                    z*2
                ], 
                dtype=torch.float, 
                device=device
            ),
            idx = idx
        )
geom.build()

for episode in range(episodes):
    geom.reset()
    geom.build()
    for step in range(steps):
        start = time.perf_counter()
        obs = geom.step(
            mode = 'euler+T_rate',
            T_att_range = {'min':0.0, 'max':0.0},
            act = act,
            alpha_1_range = {'min':0.00, 'max':0.00},
            dt = 0.01
        )

        """ 深度图可视化 """
        img_list = []
        for idx in range(batch_size):
            img = 255 * obs['depth'][idx, 0, ...].detach().cpu().numpy() / max_depth
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            img_norm_np = img_norm.astype(np.uint8)
            img_list.append(img_norm_np)
        # 计算网格布局
        num_images = len(img_list)
        cols = min(4, num_images)
        rows = math.ceil(num_images / cols)
        # 获取单张图像尺寸
        h, w = img_list[0].shape
        line_width = 2  # 分割线宽度
        # 创建空白画布（考虑分割线）
        canvas_h = h * rows + line_width * (rows - 1)
        canvas_w = w * cols + line_width * (cols - 1)
        canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255
        # 填充图像
        for idx, img in enumerate(img_list):
            row = idx // cols
            col = idx % cols
            # 计算图像位置（考虑分割线）
            y_start = row * (h + line_width)
            y_end = y_start + h
            x_start = col * (w + line_width)
            x_end = x_start + w
            canvas[y_start:y_end, x_start:x_end] = img
        # 缩放画布到合适大小（可选：放大显示）
        scale = 3  # 放大1.5倍
        canvas_resized = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("All Depth Images", canvas_resized)
        cv2.waitKey(1)

        end = time.perf_counter()
        elapsed = end - start
        print(obs['pos'])
        print(f"{1/elapsed}\n")


