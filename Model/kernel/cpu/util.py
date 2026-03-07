import torch
import torch.nn.functional as F

def euler_to_R(euler, convention='zyx'):
    shape = euler.shape[:-1]
    euler = euler.reshape(-1, 3)
    
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    
    if convention == 'zyx':  # 航空顺序: yaw-pitch-roll
        R = torch.stack([
            cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr,
            sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr,
            -sp,   cp*sr,             cp*cr
        ], dim=-1).reshape(-1, 3, 3)
    
    elif convention == 'xyz':  # 机器人顺序
        R = torch.stack([
            cp*cy, -cp*sy, sp,
            cr*sy + sr*sp*cy, cr*cy - sr*sp*sy, -sr*cp,
            sr*sy - cr*sp*cy, sr*cy + cr*sp*sy, cr*cp
        ], dim=-1).reshape(-1, 3, 3)
    
    return R.reshape(*shape, 3, 3)

def R_to_euler(R, convention='zyx'):
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    if convention == 'zyx':
        # 检查万向锁
        sy = torch.sqrt(R[..., 0, 0]**2 + R[..., 1, 0]**2)
        singular = sy < 1e-6
        
        roll = torch.zeros_like(sy)
        pitch = torch.zeros_like(sy)
        yaw = torch.zeros_like(sy)
        
        # 非奇异情况
        mask = ~singular
        roll[mask] = torch.atan2(R[mask, 2, 1], R[mask, 2, 2])
        pitch[mask] = torch.atan2(-R[mask, 2, 0], sy[mask])
        yaw[mask] = torch.atan2(R[mask, 1, 0], R[mask, 0, 0])
        
        # 奇异情况 (万向锁)
        roll[singular] = torch.atan2(R[singular, 0, 1], R[singular, 0, 2])
        pitch[singular] = torch.atan2(-R[singular, 2, 0], sy[singular])
        yaw[singular] = 0
        
        euler = torch.stack([roll, pitch, yaw], dim=-1)
    
    elif convention == 'xyz':
        sy = torch.sqrt(R[..., 1, 2]**2 + R[..., 2, 2]**2)
        singular = sy < 1e-6
        
        roll = torch.zeros_like(sy)
        pitch = torch.zeros_like(sy)
        yaw = torch.zeros_like(sy)
        
        mask = ~singular
        roll[mask] = torch.atan2(-R[mask, 1, 2], R[mask, 2, 2])
        pitch[mask] = torch.atan2(R[mask, 0, 2], sy[mask])
        yaw[mask] = torch.atan2(-R[mask, 0, 1], R[mask, 0, 0])
        
        roll[singular] = torch.atan2(-R[singular, 1, 2], R[singular, 2, 2])
        pitch[singular] = torch.atan2(R[singular, 0, 2], sy[singular])
        yaw[singular] = 0
        
        euler = torch.stack([roll, pitch, yaw], dim=-1)
    
    return euler.reshape(*shape, 3)

def quat_to_R(q):
    shape = q.shape[:-1]
    q = q.reshape(-1, 4)
    
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R = torch.stack([
        1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w,
        2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
        2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y
    ], dim=-1).reshape(-1, 3, 3)
    
    return R.reshape(*shape, 3, 3)

def R_to_quat(R):
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    batch_size = R.shape[0]
    q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    mask1 = trace > 0
    if mask1.any():
        s = 0.5 / torch.sqrt(trace[mask1] + 1.0)
        q[mask1, 0] = 0.25 / s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) * s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) * s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) * s
    
    mask2 = ~mask1
    if mask2.any():
        # 找到最大对角元素
        diag = torch.stack([
            R[mask2, 0, 0],
            R[mask2, 1, 1],
            R[mask2, 2, 2]
        ], dim=-1)
        max_diag, max_idx = torch.max(diag, dim=-1)
        
        for i in range(3):
            mask_i = (max_idx == i) & mask2
            if not mask_i.any():
                continue
            
            if i == 0:  # R[0,0] 最大
                s = 2.0 * torch.sqrt(1.0 + R[mask_i, 0, 0] - R[mask_i, 1, 1] - R[mask_i, 2, 2])
                q[mask_i, 0] = (R[mask_i, 2, 1] - R[mask_i, 1, 2]) / s
                q[mask_i, 1] = 0.25 * s
                q[mask_i, 2] = (R[mask_i, 0, 1] + R[mask_i, 1, 0]) / s
                q[mask_i, 3] = (R[mask_i, 0, 2] + R[mask_i, 2, 0]) / s
            
            elif i == 1:  # R[1,1] 最大
                s = 2.0 * torch.sqrt(1.0 + R[mask_i, 1, 1] - R[mask_i, 0, 0] - R[mask_i, 2, 2])
                q[mask_i, 0] = (R[mask_i, 0, 2] - R[mask_i, 2, 0]) / s
                q[mask_i, 1] = (R[mask_i, 0, 1] + R[mask_i, 1, 0]) / s
                q[mask_i, 2] = 0.25 * s
                q[mask_i, 3] = (R[mask_i, 1, 2] + R[mask_i, 2, 1]) / s
            
            else:  # R[2,2] 最大
                s = 2.0 * torch.sqrt(1.0 + R[mask_i, 2, 2] - R[mask_i, 0, 0] - R[mask_i, 1, 1])
                q[mask_i, 0] = (R[mask_i, 1, 0] - R[mask_i, 0, 1]) / s
                q[mask_i, 1] = (R[mask_i, 0, 2] + R[mask_i, 2, 0]) / s
                q[mask_i, 2] = (R[mask_i, 1, 2] + R[mask_i, 2, 1]) / s
                q[mask_i, 3] = 0.25 * s

    q = F.normalize(q, p=2, dim=-1)
    
    return q.reshape(*shape, 4)

def quat_to_euler(q, convention='zyx'):
    R = quat_to_R(q)
    return R_to_euler(R, convention)

def euler_to_quat(euler, convention='zyx'):
    R = euler_to_R(euler, convention)
    return R_to_quat(R)

def quat_multi(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=-1)

def quat_conjugate(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def quat_rotate_vector(q, v):
    shape = v.shape
    if q.dim() == 1:
        q = q.unsqueeze(0)
    if v.dim() == 1:
        v = v.unsqueeze(0)
    
    # 将向量扩展为四元数 [0, v]
    v_q = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    
    # 旋转: q * v * q^-1
    q_inv = quat_conjugate(q)
    rotated_q = quat_multi(quat_multi(q, v_q), q_inv)
    
    return rotated_q[..., 1:].reshape(shape)

def get_forward_vector(R_or_q):
    if R_or_q.shape[-1] == 4:  # 是四元数
        R = quat_to_R(R_or_q)
    else:
        R = R_or_q
    return R[..., :, 0]  # 第一列

def get_up_vector(R_or_q):
    if R_or_q.shape[-1] == 4:
        R = quat_to_R(R_or_q)
    else:
        R = R_or_q
    return R[..., :, 2]  # 第三列

def get_right_vector(R_or_q):
    if R_or_q.shape[-1] == 4:
        R = quat_to_R(R_or_q)
    else:
        R = R_or_q
    return R[..., :, 1]  # 第二列