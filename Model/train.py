import torch
import math
import os
import torch.optim as optim


import env.util as util
import env.geom as geom
import model

device = 'cuda'
dt = 0.005
drone_init_pos = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double, device=device)
drone_init_euler = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double, device=device)
T_max = 4*0.4315*9.81
T_max_att = 0.0
ang_vel_max = torch.tensor([0.4, 0.4, 0.4], dtype=torch.double, device=device)
depth_pos_offset = torch.tensor([0.0425, 0.0, 0.0345], dtype=torch.double, device=device)
init_forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.double, device=device)
init_up_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double, device=device)
num = 1
num_cylinders = 1
cylinders = torch.tensor([2.0, 1.0, 0.0, 3.0, 0.5], dtype=torch.double, device=device)
num_boxes = 1
boxes = torch.tensor([1.0, 0.0, 0.0, 0.3, 0.5, 3.0], dtype=torch.double, device=device)


wind_dir = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.double, device=device)
wind_speed = torch.tensor([0.0], dtype=torch.double, device=device)


scene = geom.geom(camera_pos=(3.5, 3.5, 3.5), camera_lookat=(0.0, 0.0, 0.0), camera_fov=35, 
                    max_FPS=1000, show_viewer=True, dt=dt, device=device)
scene.add_cylinders(num_cylinders, cylinders)
scene.add_boxes(num_boxes, boxes)
scene.add_drone(urdf_path="urdf/ge_fpv.urdf", drone_init_pos=drone_init_pos, drone_init_euler=drone_init_euler, 
                T_max=T_max, T_max_att=T_max_att, ang_vel_max=ang_vel_max, res_W=640, res_H=400, 
                depth_pos_offset=depth_pos_offset, depth_euler_offset=drone_init_euler, 
                depth_fov_H=67.9, depth_fov_V=45.3, num=num) 
scene.build()

fpv_model = model.Model().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
optimizer = optim.Adam(fpv_model.parameters(), lr=0.001) # 优化器
target_act = torch.tensor([0,0,0.0,0.5], device=device)
penalty_act = torch.tensor([0,0,0,0.5], device=device)
best_loss = 100000.0
loss = torch.tensor(0.0, device=device, requires_grad=True)
time = 0.0
z_max = 1.0
z_min = 1.0

while True:
    time += dt
    fpv_model.train()
    depth_img = scene.depth_img.unsqueeze(0).to(device)
    euler = util.quat_to_euler(scene.quat)
    # print(depth_img.shape)
    # print(euler.shape)
    act, _, hidden_state = fpv_model.forward(depth_img, euler)
    # print(act.squeeze(0).shape)

    if torch.abs(scene.pos[0, 2]-1.0) < 0.1:
        loss = loss+torch.mean(torch.square(act - target_act))*30.0 # 鼓励稳定悬停
    elif torch.abs(scene.pos[0, 2]-1.0) >= 0.1 and torch.abs(scene.pos[0, 2]-1.0) < 0.6:
        loss = loss+torch.mean(torch.square(act + penalty_act))*5.0  # 惩罚
    else:
        loss = loss+torch.mean(torch.square(act + penalty_act))*4.0  # 惩罚
    # if act.squeeze(0)[3] > 1.0-0.19119351:
    #     loss = loss+torch.mean(torch.square(act + penalty_act))*0.5  # 惩罚
    # if act.squeeze(0)[3] < -0.19119351:
    #     loss = loss+torch.mean(torch.square(act + penalty_act))*2.5 # 惩罚
    # if act.squeeze(0)[3] > 0.0:
    #     loss = loss+torch.mean(torch.square(act - target_act))*4.0  # 鼓励

    if scene.pos[0, 2] > z_max:
        z_max = scene.pos[0, 2].item()
    if scene.pos[0, 2] < z_min:
        z_min = scene.pos[0, 2].item()

    if abs(z_max-z_min) < 0.2:
        loss = loss+torch.mean(torch.square(act - target_act))*8.0 # 鼓励稳定悬停
    else:
        loss = loss+torch.mean(torch.square(act + penalty_act))*7.0  # 惩罚
    
    print(act)
    print(scene.pos)
    print(abs(z_max-z_min))
    print(best_loss)
    print(loss)
    print(time)
    print("\n")

    if act.squeeze(0)[3] < 0.0:
        act.squeeze(0)[3] = 0.0
    if act.squeeze(0)[3] > 1.0:
        act.squeeze(0)[3] = 1.0

    if scene.step(torch.tensor([act.squeeze(0)[0], act.squeeze(0)[1], act.squeeze(0)[2]], dtype=torch.double, device=device), act.squeeze(0)[3]) or scene.pos[0, 2] < 0 or abs(z_max-z_min) > 1.0:    # 0.33*9.81  0.19119351
        print("retry")
        scene.reset()
        loss = loss+torch.mean(torch.square(act + penalty_act))*30.0  # 惩罚
        print(loss)
        optimizer.zero_grad()  # 清空上一轮梯度（必须！）
        loss.backward()        # 反向传播，计算梯度
        optimizer.step()       # 更新参数
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        time = 0.0
        z_max = 1.0
        z_min = 1.0
    
    if time >= 3:
        scene.reset()
        loss = loss+torch.mean(torch.square(act - target_act))*6.0 # 鼓励稳定悬停
        print(loss)
        optimizer.zero_grad()  # 清空上一轮梯度（必须！）
        loss.backward()        # 反向传播，计算梯度
        optimizer.step()       # 更新参数
        time = 0.0
        z_max = 1.0
        z_min = 1.0

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(fpv_model.state_dict(), "best.pth")
        
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    

    # print(scene.quat)
    
