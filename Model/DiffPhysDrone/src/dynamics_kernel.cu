#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// 定义宏以确保兼容性
#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() \
  do { \
    cudaError_t err = cudaGetLastError(); \
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err)); \
  } while(0)
#endif

namespace {

template <typename scalar_t>
__global__ void update_state_vec_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> R_new,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> R,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> a_thr,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> v_pred,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> alpha,
    float yaw_inertia) {
    
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (b >= B) return;
    
    // a_thr = a_thr - self.g_std;
    scalar_t ax = a_thr[b][0];
    scalar_t ay = a_thr[b][1];
    scalar_t az = a_thr[b][2] + 9.80665;
    
    // thrust = torch.norm(a_thr, 2, -1, True);
    scalar_t thrust = sqrt(ax*ax + ay*ay + az*az);
    
    // self.up_vec = a_thr / thrust;
    scalar_t ux = ax / thrust;
    scalar_t uy = ay / thrust;
    scalar_t uz = az / thrust;
    
    // forward_vec = self.forward_vec * yaw_inertia + v_pred;
    scalar_t fx = R[b][0][0] * yaw_inertia + v_pred[b][0];
    scalar_t fy = R[b][1][0] * yaw_inertia + v_pred[b][1];
    scalar_t fz = R[b][2][0] * yaw_inertia + v_pred[b][2];
    
    // forward_vec = F.normalize(forward_vec, 2, -1);
    // forward_vec = (1-alpha) * forward_vec + alpha * self.forward_vec
    scalar_t t = sqrt(fx * fx + fy * fy + fz * fz);
    fx = (1 - alpha[b][0]) * (fx / t) + alpha[b][0] * R[b][0][0];
    fy = (1 - alpha[b][0]) * (fy / t) + alpha[b][0] * R[b][1][0];
    fz = (1 - alpha[b][0]) * (fz / t) + alpha[b][0] * R[b][2][0];
    
    // forward_vec[2] = (forward_vec[0] * self_up_vec[0] + forward_vec[1] * self_up_vec[1]) / -self_up_vec[2]
    fz = (fx * ux + fy * uy) / -uz;
    
    // self.forward_vec = F.normalize(forward_vec, 2, -1);
    t = sqrt(fx * fx + fy * fy + fz * fz);
    fx /= t;
    fy /= t;
    fz /= t;
    
    // self.left_vec = torch.cross(self.up_vec, self.forward_vec);
    R_new[b][0][0] = fx;
    R_new[b][0][1] = uy * fz - uz * fy;
    R_new[b][0][2] = ux;
    R_new[b][1][0] = fy;
    R_new[b][1][1] = uz * fx - ux * fz;
    R_new[b][1][2] = uy;
    R_new[b][2][0] = fz;
    R_new[b][2][1] = ux * fy - uy * fx;
    R_new[b][2][2] = uz;
}

template <typename scalar_t>
__global__ void run_forward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> R,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dg,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> z_drag_coef,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> drag_2,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> pitch_ctl_delay,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> act_pred,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> act,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> p,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> v_wind,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> act_next,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> p_next,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> v_next,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> a_next,
    float ctl_dt, float airmode_av2a) {
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (i >= B) return;
    
    // alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
    scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);
    
    // self.act = act_pred * (1 - alpha) + self.act * alpha
    for (int j = 0; j < 3; j++)
        act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;
    
    // v_rel = v - v_wind
    scalar_t v_rel_x = v[i][0] - v_wind[i][0];
    scalar_t v_rel_y = v[i][1] - v_wind[i][1];
    scalar_t v_rel_z = v[i][2] - v_wind[i][2];
    
    // Project velocities to body frame
    scalar_t v_fwd_s = v_rel_x * R[i][0][0] + v_rel_y * R[i][1][0] + v_rel_z * R[i][2][0];
    scalar_t v_left_s = v_rel_x * R[i][0][1] + v_rel_y * R[i][1][1] + v_rel_z * R[i][2][1];
    scalar_t v_up_s = v_rel_x * R[i][0][2] + v_rel_y * R[i][1][2] + v_rel_z * R[i][2][2];
    
    scalar_t v_up_2 = v_up_s * abs(v_up_s);
    scalar_t v_fwd_2 = v_fwd_s * abs(v_fwd_s);
    scalar_t v_left_2 = v_left_s * abs(v_left_s);
    
    scalar_t a_drag_2[3], a_drag_1[3];
    for (int j = 0; j < 3; j++) {
        a_drag_2[j] = v_up_2 * R[i][j][2] * z_drag_coef[i][0] + 
                      v_left_2 * R[i][j][1] + v_fwd_2 * R[i][j][0];
        a_drag_1[j] = v_up_s * R[i][j][2] * z_drag_coef[i][0] + 
                      v_left_s * R[i][j][1] + v_fwd_s * R[i][j][0];
    }
    
    // Compute airmode contribution
    scalar_t dot = act[i][0] * act_next[i][0] + act[i][1] * act_next[i][1] + 
                  (act[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
    scalar_t n1 = act[i][0] * act[i][0] + act[i][1] * act[i][1] + 
                 (act[i][2] + 9.80665) * (act[i][2] + 9.80665);
    scalar_t n2 = act_next[i][0] * act_next[i][0] + act_next[i][1] * act_next[i][1] + 
                 (act_next[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
    
    scalar_t av = acos(fmax(-1., fmin(1., dot / fmax(1e-8, sqrt(n1) * sqrt(n2))))) / ctl_dt;
    
    scalar_t ax = act[i][0];
    scalar_t ay = act[i][1];
    scalar_t az = act[i][2] + 9.80665;
    scalar_t thrust = sqrt(ax*ax + ay*ay + az*az);
    
    scalar_t airmode_a[3] = {
        ax / thrust * av * airmode_av2a,
        ay / thrust * av * airmode_av2a,
        az / thrust * av * airmode_av2a
    };
    
    // Compute next acceleration
    for (int j = 0; j < 3; j++)
        a_next[i][j] = act_next[i][j] + dg[i][j] - 
                      a_drag_2[j] * drag_2[i][0] - 
                      a_drag_1[j] * drag_2[i][1] + 
                      airmode_a[j];
    
    // Update position
    for (int j = 0; j < 3; j++)
        p_next[i][j] = p[i][j] + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;
    
    // Update velocity
    for (int j = 0; j < 3; j++)
        v_next[i][j] = v[i][j] + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;
}

template <typename scalar_t>
__global__ void run_backward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> R,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dg,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> z_drag_coef,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> drag_2,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> pitch_ctl_delay,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> v_wind,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> act_next,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_act_pred,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_act,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_p,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_v,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_a,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> _d_act_next,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_p_next,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_v_next,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> _d_a_next,
    float grad_decay,
    float ctl_dt) {
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (i >= B) return;
    
    // alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
    scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);
    
    // Load gradients from next step
    scalar_t d_act_next_val[3] = {_d_act_next[i][0], _d_act_next[i][1], _d_act_next[i][2]};
    scalar_t d_a_next_val[3] = {_d_a_next[i][0], _d_a_next[i][1], _d_a_next[i][2]};
    
    // Backward through velocity update
    for (int j = 0; j < 3; j++) {
        d_v[i][j] = d_v_next[i][j] * pow(grad_decay, ctl_dt);
        d_a[i][j] = 0.5 * ctl_dt * d_v_next[i][j];
        d_a_next_val[j] += 0.5 * ctl_dt * d_v_next[i][j];
    }
    
    // Backward through position update
    for (int j = 0; j < 3; j++) {
        d_p[i][j] = d_p_next[i][j] * pow(grad_decay, ctl_dt);
        d_v[i][j] += ctl_dt * d_p_next[i][j];
        d_a[i][j] += 0.5 * ctl_dt * ctl_dt * d_p_next[i][j];
    }
    
    scalar_t d_a_drag_2[3];
    scalar_t d_a_drag_1[3];
    
    for (int j = 0; j < 3; j++) {
        d_act_next_val[j] += d_a_next_val[j];
        d_a_drag_2[j] = -d_a_next_val[j] * drag_2[i][0];
        d_a_drag_1[j] = -d_a_next_val[j] * drag_2[i][1];
    }
    
    // Compute relative velocity
    scalar_t v_rel_x = v[i][0] - v_wind[i][0];
    scalar_t v_rel_y = v[i][1] - v_wind[i][1];
    scalar_t v_rel_z = v[i][2] - v_wind[i][2];
    
    scalar_t v_fwd_s = v_rel_x * R[i][0][0] + v_rel_y * R[i][1][0] + v_rel_z * R[i][2][0];
    scalar_t v_left_s = v_rel_x * R[i][0][1] + v_rel_y * R[i][1][1] + v_rel_z * R[i][2][1];
    scalar_t v_up_s = v_rel_x * R[i][0][2] + v_rel_y * R[i][1][2] + v_rel_z * R[i][2][2];
    
    scalar_t d_v_fwd_s = 0;
    scalar_t d_v_left_s = 0;
    scalar_t d_v_up_s = 0;
    
    for (int j = 0; j < 3; j++) {
        d_v_fwd_s += d_a_drag_2[j] * 2 * abs(v_fwd_s) * R[i][j][0];
        d_v_left_s += d_a_drag_2[j] * 2 * abs(v_left_s) * R[i][j][1];
        d_v_up_s += d_a_drag_2[j] * 2 * abs(v_up_s) * R[i][j][2] * z_drag_coef[i][0];
        d_v_fwd_s += d_a_drag_1[j] * R[i][j][0];
        d_v_left_s += d_a_drag_1[j] * R[i][j][1];
        d_v_up_s += d_a_drag_1[j] * R[i][j][2] * z_drag_coef[i][0];
    }
    
    // Backpropagate to velocity
    for (int j = 0; j < 3; j++) {
        d_v[i][j] += R[i][j][0] * d_v_fwd_s;
        d_v[i][j] += R[i][j][1] * d_v_left_s;
        d_v[i][j] += R[i][j][2] * d_v_up_s;
    }
    
    // Backward through act update
    for (int j = 0; j < 3; j++) {
        d_act_pred[i][j] = (1 - alpha) * d_act_next_val[j];
        d_act[i][j] = alpha * d_act_next_val[j];
    }
}

} // namespace

// Forward function
std::vector<torch::Tensor> run_forward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor act_pred,
    torch::Tensor act,
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor a,
    float ctl_dt,
    float airmode_av2a) {
    
    // Check tensor devices and shapes
    TORCH_CHECK(R.device().is_cuda(), "R must be a CUDA tensor");
    TORCH_CHECK(dg.device().is_cuda(), "dg must be a CUDA tensor");
    
    auto batch_size = R.size(0);
    
    torch::Tensor act_next = torch::empty_like(act);
    torch::Tensor p_next = torch::empty_like(p);
    torch::Tensor v_next = torch::empty_like(v);
    torch::Tensor a_next = torch::empty_like(a);
    
    // Configure kernel launch
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(R.scalar_type(), "run_forward_cuda", ([&] {
        run_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            dg.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            z_drag_coef.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            drag_2.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            pitch_ctl_delay.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            act_pred.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            act.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            p.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            v_wind.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            a.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            act_next.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            p_next.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            v_next.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            a_next.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            ctl_dt, airmode_av2a);
    }));
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return {act_next, p_next, v_next, a_next};
}

// Backward function
std::vector<torch::Tensor> run_backward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor act_next,
    torch::Tensor _d_act_next,
    torch::Tensor d_p_next,
    torch::Tensor d_v_next,
    torch::Tensor _d_a_next,
    float grad_decay,
    float ctl_dt) {
    
    auto batch_size = R.size(0);
    
    torch::Tensor d_act_pred = torch::empty_like(dg);
    torch::Tensor d_act = torch::empty_like(dg);
    torch::Tensor d_p = torch::empty_like(dg);
    torch::Tensor d_v = torch::empty_like(dg);
    torch::Tensor d_a = torch::empty_like(dg);
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(R.scalar_type(), "run_backward_cuda", ([&] {
        run_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            dg.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            z_drag_coef.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            drag_2.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            pitch_ctl_delay.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            v_wind.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            act_next.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            d_act_pred.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            d_act.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            d_p.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            d_v.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            d_a.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            _d_act_next.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            d_p_next.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            d_v_next.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            _d_a_next.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            grad_decay, ctl_dt);
    }));
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return {d_act_pred, d_act, d_p, d_v, d_a};
}

// Update state function
torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia) {
    
    auto batch_size = a_thr.size(0);
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    torch::Tensor R_new = torch::empty_like(R);
    
    AT_DISPATCH_FLOATING_TYPES(a_thr.scalar_type(), "update_state_vec_cuda", ([&] {
        update_state_vec_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R_new.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            R.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            a_thr.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            v_pred.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            alpha.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            yaw_inertia);
    }));
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return R_new;
}

// Module definition - 注意这里不能有重复的PYBIND11_MODULE