import torch
import genesis
import cv2
import numpy as np
import time

import env.util as util
import env.geom as geom
import env.robot as robot
import env.sensor as sensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
init_pos = torch.tensor([0.0, 0.0, 0.05], dtype=torch.float, device=device, requires_grad=True)
init_euler = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
act = torch.tensor(
    [[[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0]]], 
    dtype=torch.float, device=device, requires_grad=True)
# act = torch.tensor(
#     [[[0.0, 30.0, 0.0, 0.90], [0.0, 30.0, 0.0, 0.90]],
#      [[0.0, 30.0, 0.0, 0.90], [0.0, 30.0, 0.0, 0.90]]], 
#     dtype=torch.float, device=device, requires_grad=True)
pos_offset = torch.tensor([0.0425, 0.0, 0.0345], dtype=torch.float, device=device, requires_grad=True)
euler_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)

if device == 'cuda':
    genesis.init(
        seed                = None,
        precision           = '32',
        debug               = False,
        eps                 = 1e-12,
        logging_level       = 'warning',
        backend             = genesis.cuda,
        theme               = 'dark',
        logger_verbose_time = False
    )
else:
    genesis.init(
        seed                = None,
        precision           = '32',
        debug               = False,
        eps                 = 1e-12,
        logging_level       = 'warning',
        backend             = genesis.cpu,
        theme               = 'dark',
        logger_verbose_time = False
    )
viewer_options = genesis.options.ViewerOptions(
    camera_pos=(1.0, 1.0, 1.0),
    camera_lookat=(0.0, 0.0, 0.0),
    camera_fov=90,
    max_FPS=120,
)
scene = genesis.Scene(
    sim_options=genesis.options.SimOptions(
        dt=0.01,
    ),
    viewer_options=viewer_options,
    show_viewer=True,
)
plane = scene.add_entity(
    genesis.morphs.Plane(
        visualization=True,   # 显示地面
        collision=False        # 有碰撞效果
    ),
)
# drone = scene.add_entity(
#     morph=genesis.morphs.Drone(
#         file="urdf/ge_fpv.urdf",
#         pos=init_pos.clone().detach().to('cpu').numpy(),
#         euler=init_euler.clone().detach().to('cpu').numpy(),
#     ),
# )
# scene.add_entity(
#     genesis.morphs.Sphere(
#         pos=(2.0, 0.0, 1.0),
#         radius=0.2,
#         fixed=True,
#     )
# )
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.3,
        pos=(-5.0, 0.0, 1.0),
        fixed=True,
    )
)
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.4,
        pos=(-2.0, 1.0, 1.0),
        fixed=True,
    )
)
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.2,
        pos=(2.0, 3.0, 1.0),
        fixed=True,
    )
)
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.6,
        pos=(3.0, 2.0, 1.0),
        fixed=True,
    )
)
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.2,
        pos=(-3.0, -2.0, 1.0),
        fixed=True,
    )
)
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.6,
        pos=(5.0, 0.0, 1.0),
        fixed=True,
    )
)
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.5,
        pos=(5.0, 2.0, 1.0),
        fixed=True,
    )
)
scene.build()

geom = geom.geom(
    batch_size=17,
    device=device,
)
for i in range(1):
    drone_robot = robot.drone(
        device = device,
        init_pos = init_pos, 
        init_euler = init_euler, 
        mass = 0.33, 
        T_max = 0.33*9.81, 
        collision_radius = 0.4
    )
    cloest_dist = sensor.cloest_dist(
        device = device
    )
    depth = sensor.depth(
        device = device,
        pos_offset = pos_offset, 
        euler_offset = euler_offset,  
        res_W = 40, 
        res_H = 25, 
        fov_H = 67.9,
        fov_V = 45.3,
        distance_range = {'min': 0.25, 'max': 2.5},
        noise_range = {'min':0.0, 'max':0.0},
        black_hole_prob = 0.0
    )
    lidar_2D = sensor.lidar_2D(
        device = device,
        pos_offset = pos_offset, 
        euler_offset = euler_offset,  
        angle_range = {'start':0.0, 'end':360.0},
        angular_res = 1.0,
        distance_range = {'min': 0.25, 'max': 8.0},
        noise_range = {'min':0.0, 'max':0.0}
    )
    drone_robot.sensor_bind(cloest_dist)
    drone_robot.sensor_bind(depth)
    drone_robot.sensor_bind(lidar_2D)
    geom.add_robot(drone_robot)
for idx in range(17):
    # geom.add_sphere(
    #     torch.tensor([2.0, 0.0, 1.0, 0.2], dtype=torch.float, device=device),
    #     idx = idx
    # )
    # geom.add_sphere(
    #     torch.tensor([0.0, 0.0, 0.0, 0.8], dtype=torch.float, device=device),
    #     idx = idx
    # )
    geom.add_cylinder(
        torch.tensor([-5.0, 0.0, 1.0, 0.3, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([-2.0, 1.0, 1.0, 0.4, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([2.0, 3.0, 1.0, 0.2, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([3.0, 2.0, 1.0, 0.6, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([-3.0, -2.0, 1.0, 0.2, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([5.0, 0.0, 1.0, 0.6, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([5.0, 2.0, 1.0, 0.5, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
geom.build()
count = 0

def draw_lidar_points(distance, angle_deg, img_size=300, max_distance=10):
    """
    只显示点云点，没有任何辅助线
    
    Args:
        distance: [num] 距离张量
        angle_deg: [num] 角度张量（度数）
        img_size: 图像大小
        max_distance: 最大距离（米）
    """
    # 转换为 numpy
    if torch.is_tensor(distance):
        distance = distance.cpu().numpy()
    if torch.is_tensor(angle_deg):
        angle_deg = angle_deg.cpu().numpy()
    
    # 创建黑色背景
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    center = (img_size // 2, img_size // 2)
    
    # 极坐标转直角坐标并绘制
    for dist, ang in zip(distance, angle_deg):
        if dist <= 0 or dist > max_distance:
            continue
            
        rad = np.deg2rad(-ang-90.0)
        r = (dist / max_distance) * (img_size // 2 - 10)
        x = int(center[0] + r * np.cos(rad))
        y = int(center[1] + r * np.sin(rad))
        
        # 只画点（白色）
        cv2.circle(img, (x, y), 1, (255, 255, 255), -1)
    
    return img

while True:
    start = time.perf_counter()
    obs = geom.step(
        mode = 'euler+T_rate',
        T_att_range = {'min':0.0, 'max':0.0},
        act = act,
        alpha_1_range = {'min':0.01, 'max':0.01},
        dt = 0.01
    )

    """ 深度图 """
    # img = 255*obs['depth'][0, 0, ...].detach().cpu().numpy()/2.5
    # # 2. 归一化到 0~255（深度图必须做这步）
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv2.imshow("DEPTH VIEWER", img.astype(np.uint8))
    # cv2.waitKey(1)
    """ 点云 """
    img = draw_lidar_points(obs['cloud_point'][0, 0, :, 0], obs['cloud_point'][0, 0, :, 1], img_size=300, max_distance=7.0)
    cv2.imshow("Lidar Points", img)
    cv2.waitKey(1)

    end = time.perf_counter()
    elapsed = end - start
    # print(obs['cloud_point'][0, 0, :, :])
    print(f"pos {obs['pos'][0, 0, ...]}")
    print(f"dist {obs['distance'][0, 0, ...]}")
    print(obs['pos'].shape)
    print(obs['distance'].shape)
    print(obs['depth'].shape)
    print(f"{1/elapsed}\n")

    scene.draw_debug_sphere(pos=obs['pos'][0, 0, ...].clone().detach().cpu(), radius=0.1, color=(1, 0, 0))
    # drone.set_pos(obs['pos'][0, 0, ...].clone().detach())
    # drone.set_quat(util.euler_to_quat(util.deg_to_rad(obs['ang'][0, 0, ...]).clone().detach()))
    scene.step()
    # break

# drone_robot.reset()
# while True:
#     drone_robot.solver(
#         'euler+T_rate', 
#         {'min':0.0, 'max':0.0}, 
#         act, 
#         {'min':0.0, 'max':0.0}, 
#         0.01
#     )
#     drone.set_pos(drone_robot.pos.clone().detach())
#     drone.set_quat(util.euler_to_quat(drone_robot.euler.clone().detach()))
#     scene.step()

    # drone_robot.sensor_list[0]['sensor_class'].render(init_euler, drone_robot.euler)
    # print(drone_robot.sensor_list[0]['sensor_class'].pos)