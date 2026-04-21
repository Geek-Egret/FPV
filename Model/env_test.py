import torch
import genesis
import cv2
import numpy as np
import time

import env.util as util
import env.geom as geom
import env.robot as robot
import env.sensor as sensor

device = 'cuda'
init_pos = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float, device=device, requires_grad=True)
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

if device == device:
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
drone = scene.add_entity(
    morph=genesis.morphs.Drone(
        file="urdf/ge_fpv.urdf",
        pos=init_pos.clone().detach().to('cpu').numpy(),
        euler=init_euler.clone().detach().to('cpu').numpy(),
    ),
)
# scene.add_entity(
#     genesis.morphs.Sphere(
#         pos=(2.0, 0.0, 1.0),
#         radius=0.2,
#         fixed=True,
#     )
# )
# scene.add_entity(
#     genesis.morphs.Sphere(
#         pos=(0.0, 0.0, 0.0),
#         radius=0.8,
#         fixed=True,
#     )
# )
scene.add_entity(
    genesis.morphs.Cylinder(
        height=0.2,
        radius=1.2,
        pos=(0.0, 0.0, 0.1),
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
        min_depth = 0.25,
        max_depth = 2.5,
        noise_range = {'min':0.0, 'max':0.0},
        black_hole_prob = 0.0
    )
    drone_robot.sensor_bind(cloest_dist)
    drone_robot.sensor_bind(depth)
    geom.add_robot(drone_robot)
for idx in range(17):
    geom.add_sphere(
        torch.tensor([2.0, 0.0, 1.0, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_sphere(
        torch.tensor([0.0, 0.0, 0.0, 0.8], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([0.0, 0.0, 0.1, 1.2, 0.2], dtype=torch.float, device=device),
        idx = idx
    )

    geom.add_sphere(
        torch.tensor([2.0, 0.0, 1.0, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_sphere(
        torch.tensor([0.0, 0.0, 0.0, 0.8], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([0.0, 0.0, 0.1, 1.2, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_sphere(
        torch.tensor([2.0, 0.0, 1.0, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_sphere(
        torch.tensor([0.0, 0.0, 0.0, 0.8], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([0.0, 0.0, 0.1, 1.2, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_sphere(
        torch.tensor([2.0, 0.0, 1.0, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_sphere(
        torch.tensor([0.0, 0.0, 0.0, 0.8], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([0.0, 0.0, 0.1, 1.2, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_sphere(
        torch.tensor([2.0, 0.0, 1.0, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_sphere(
        torch.tensor([0.0, 0.0, 0.0, 0.8], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([0.0, 0.0, 0.1, 1.2, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_sphere(
        torch.tensor([2.0, 0.0, 1.0, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_sphere(
        torch.tensor([0.0, 0.0, 0.0, 0.8], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([0.0, 0.0, 0.1, 1.2, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_sphere(
        torch.tensor([2.0, 0.0, 1.0, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_sphere(
        torch.tensor([0.0, 0.0, 0.0, 0.8], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([0.0, 0.0, 0.1, 1.2, 0.2], dtype=torch.float, device=device),
        idx = idx
    )
geom.build()
count = 0
while True:
    start = time.perf_counter()
    obs = geom.step(
        mode = 'euler+T_rate',
        T_att_range = {'min':0.0, 'max':0.0},
        act = act,
        alpha_1_range = {'min':0.01, 'max':0.01},
        dt = 0.01
    )

    img = 255*obs['depth'][0, 0, ...].detach().cpu().numpy()/2.5
    # 2. 归一化到 0~255（深度图必须做这步）
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow("DEPTH VIEWER", img.astype(np.uint8))
    cv2.waitKey(1)

    end = time.perf_counter()
    elapsed = end - start
    print(f"pos {obs['pos'][0, 0, ...]}")
    print(f"dist {obs['distance'][0, 0, ...]}")
    print(obs['pos'].shape)
    print(obs['distance'].shape)
    print(obs['depth'].shape)
    print(f"{1/elapsed}\n")

    drone.set_pos(obs['pos'][0, 0, ...].clone().detach())
    drone.set_quat(util.euler_to_quat(util.deg_to_rad(obs['ang'][0, 0, ...]).clone().detach()))
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