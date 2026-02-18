import math
import torch
import quadsim_cuda

# 设置设备
device = torch.device('cuda')
dtype = torch.float32  # 使用float32代替double以提高性能

# 创建测试数据
batch_size = 64
R = torch.randn((batch_size, 3, 3), dtype=dtype, device=device)
dg = torch.randn((batch_size, 3), dtype=dtype, device=device)
z_drag_coef = torch.randn((batch_size, 1), dtype=dtype, device=device)
drag_2 = torch.randn((batch_size, 2), dtype=dtype, device=device)
pitch_ctl_delay = torch.randn((batch_size, 1), dtype=dtype, device=device)
g_std = torch.tensor([[0, 0, -9.80665]], dtype=dtype, device=device)
act_pred = torch.randn((batch_size, 3), dtype=dtype, device=device, requires_grad=True)
act = torch.randn((batch_size, 3), dtype=dtype, device=device, requires_grad=True)
p = torch.randn((batch_size, 3), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((batch_size, 3), dtype=dtype, device=device, requires_grad=True)
v_wind = torch.randn((batch_size, 3), dtype=dtype, device=device, requires_grad=True)
a = torch.randn((batch_size, 3), dtype=dtype, device=device, requires_grad=True)

grad_decay = 0.4
ctl_dt = 1/15

class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply

def run_forward_pytorch(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt):
    alpha = torch.exp(-pitch_ctl_delay * ctl_dt)
    act_next = act_pred * (1 - alpha) + act * alpha
    
    # 计算相对速度
    v_rel = v - v_wind
    v_body = torch.einsum('bij,bj->bi', R, v_rel)
    v_fwd_s, v_left_s, v_up_s = v_body[:, 0:1], v_body[:, 1:2], v_body[:, 2:3]
    
    # 计算阻力
    drag = drag_2[:, :1] * (v_fwd_s.abs() * v_fwd_s * R[..., 0] + 
                            v_left_s.abs() * v_left_s * R[..., 1] + 
                            v_up_s.abs() * v_up_s * R[..., 2] * z_drag_coef)
    drag += drag_2[:, 1:] * (v_fwd_s * R[..., 0] + 
                             v_left_s * R[..., 1] + 
                             v_up_s * R[..., 2] * z_drag_coef)
    
    # 计算下一时刻的加速度
    a_next = act_next + dg - drag.sum(dim=1)
    
    # 更新位置和速度
    p_next = g_decay(p, grad_decay ** ctl_dt) + v * ctl_dt + 0.5 * a * ctl_dt**2
    v_next = g_decay(v, grad_decay ** ctl_dt) + (a + a_next) / 2 * ctl_dt
    
    return act_next, p_next, v_next, a_next

print("Testing forward pass...")
# 运行CUDA版本
act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, 0)

# 运行PyTorch版本
_act_next, _p_next, _v_next, _a_next = run_forward_pytorch(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt)

# 验证结果
assert torch.allclose(act_next, _act_next, rtol=1e-4, atol=1e-4)
assert torch.allclose(a_next, _a_next, rtol=1e-4, atol=1e-4)
assert torch.allclose(p_next, _p_next, rtol=1e-4, atol=1e-4)
assert torch.allclose(v_next, _v_next, rtol=1e-4, atol=1e-4)
print("Forward pass passed!")

print("Testing backward pass...")
# 创建随机梯度
d_act_next = torch.randn_like(act_next)
d_p_next = torch.randn_like(p_next)
d_v_next = torch.randn_like(v_next)
d_a_next = torch.randn_like(a_next)

# 清除之前的梯度
act_pred.grad = None
act.grad = None
p.grad = None
v.grad = None
a.grad = None

# PyTorch反向传播
torch.autograd.backward(
    (_act_next, _p_next, _v_next, _a_next),
    (d_act_next, d_p_next, d_v_next, d_a_next),
)

# 保存PyTorch梯度
act_pred_grad_pytorch = act_pred.grad.clone()
act_grad_pytorch = act.grad.clone()
p_grad_pytorch = p.grad.clone()
v_grad_pytorch = v.grad.clone()
a_grad_pytorch = a.grad.clone()

# 清除梯度
act_pred.grad = None
act.grad = None
p.grad = None
v.grad = None
a.grad = None

# CUDA反向传播
d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, 
    d_act_next, d_p_next, d_v_next, d_a_next, grad_decay, ctl_dt)

# 验证梯度
assert torch.allclose(d_act_pred, act_pred_grad_pytorch, rtol=1e-4, atol=1e-4)
assert torch.allclose(d_act, act_grad_pytorch, rtol=1e-4, atol=1e-4)
assert torch.allclose(d_p, p_grad_pytorch, rtol=1e-4, atol=1e-4)
assert torch.allclose(d_v, v_grad_pytorch, rtol=1e-4, atol=1e-4)
assert torch.allclose(d_a, a_grad_pytorch, rtol=1e-4, atol=1e-4)

print("Backward pass passed!")
print("All tests passed successfully!")